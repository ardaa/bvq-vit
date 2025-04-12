import argparse
import torch
import torch.optim as optim
from model import VisionTransformer, train_one_epoch, validate
from utils import get_imagenet_dataloaders, post_training_quantize_model, compute_metrics, save_metrics
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import shutil
from datetime import datetime
from tqdm import tqdm

class Logger:
    """Class to log console output to a file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def create_backup(output_dir, epoch, model, optimizer, train_losses, train_accs, val_losses, val_accs):
    """Create a backup of the current training state."""
    backup_dir = os.path.join(output_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create a timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f'backup_epoch_{epoch}_{timestamp}')
    os.makedirs(backup_path, exist_ok=True)
    
    # Save model and optimizer state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(backup_path, 'model_optimizer.pth'))
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    with open(os.path.join(backup_path, 'metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    # Save a copy of the current log file
    log_files = [f for f in os.listdir(output_dir) if f.startswith('training_log_')]
    if log_files:
        latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        shutil.copy2(os.path.join(output_dir, latest_log), os.path.join(backup_path, 'training_log.txt'))
    
    # Save a copy of the current metrics plots
    if os.path.exists(os.path.join(output_dir, 'training_metrics.png')):
        shutil.copy2(os.path.join(output_dir, 'training_metrics.png'), os.path.join(backup_path, 'training_metrics.png'))
    
    print(f"Backup created at epoch {epoch}: {backup_path}")
    return backup_path

def compute_metrics_with_progress(model, dataloader, criterion, device, class_names=None):
    """Compute comprehensive metrics for model evaluation with progress bar."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(total=len(dataloader), desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            batch_correct = (preds == targets).sum().item()
            batch_total = targets.size(0)
            correct += batch_correct
            total += batch_total
            
            # Update progress bar with current accuracy
            current_acc = batch_correct / batch_total
            pbar.set_postfix({"batch_acc": f"{current_acc:.4f}", "total_acc": f"{correct/total:.4f}"})
            pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    accuracy = correct / total
    avg_loss = running_loss / total
    
    # Import necessary functions from sklearn
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_targets)))]
    
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    # ROC curves for each class
    roc_data = {}
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(all_targets == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[class_names[i]] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc
        }
    
    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_targets)):
        top5_preds = np.argsort(all_probs[i])[-5:][::-1]
        if all_targets[i] in top5_preds:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_targets)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_targets == i
        if np.sum(class_mask) > 0:
            per_class_accuracy[class_name] = np.mean(all_preds[class_mask] == all_targets[class_mask])
        else:
            per_class_accuracy[class_name] = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_data': roc_data,
        'per_class_accuracy': per_class_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on ImageNet with post-training quantization")
    parser.add_argument('--data-path', type=str, default='./drive/MyDrive/BT-ViT/data',
                      help='Path to ImageNet data directory')
    parser.add_argument('--output-dir', type=str, default='./drive/MyDrive/BT-ViT/outputs',
                      help='Directory to save model outputs')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--embed-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer blocks')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--post-quantize', action='store_true', help='Enable post-training quantization')
    parser.add_argument('--quant-bits', type=int, default=8, choices=[8, 16], help='Bit width for post-training quantization (8 or 16)')
    parser.add_argument('--backup-interval', type=int, default=10, help='Create a backup every N epochs')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to backup directory to resume training from')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only (no training)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for evaluation')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'training_log_{timestamp}.txt')
    sys.stdout = Logger(log_file)
    
    # Print configuration
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Check if in evaluation mode
    if args.eval:
        if args.checkpoint is None:
            print("Error: Checkpoint must be specified in evaluation mode. Use --checkpoint PATH")
            return
        print(f"Running in evaluation mode with checkpoint: {args.checkpoint}")
    
    # Check GPU availability and set device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            # Print GPU information
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # Set default CUDA device
            torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("Using CPU for training")
    
    # Create the Vision Transformer model.
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    # If using GPU, wrap model with DataParallel if multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Load model for evaluation or resume training
    if args.eval:
        # Load model from checkpoint for evaluation
        print(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
    elif args.resume_from:
        # Resume training from backup
        print(f"Resuming training from backup: {args.resume_from}")
        backup_path = args.resume_from
        
        # Load model and optimizer state
        checkpoint = torch.load(os.path.join(backup_path, 'model_optimizer.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # Load training metrics
        with open(os.path.join(backup_path, 'metrics.json'), 'r') as f:
            import json
            metrics = json.load(f)
            train_losses = metrics['train_losses']
            train_accs = metrics['train_accs']
            val_losses = metrics['val_losses']
            val_accs = metrics['val_accs']
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Previous metrics: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
    
    # Obtain dataloaders
    train_loader, val_loader, test_loader = get_imagenet_dataloaders(args.data_path, args.batch_size, args.img_size)
    
    # Get class names if available
    class_names = None
    if hasattr(train_loader.dataset, 'classes'):
        class_names = train_loader.dataset.classes
    
    # If in evaluation mode, skip training and go straight to evaluation
    if args.eval:
        print("\nRunning evaluation on test set...")
        test_metrics = compute_metrics_with_progress(model, test_loader, criterion, device, class_names)
        save_metrics(test_metrics, args.output_dir, prefix='test_eval_')
        
        # Print evaluation results
        print(f"\nEvaluation Results:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
        
        print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to {args.output_dir}")
        return  # Exit function early, skipping all training code
    
    # Initialize variables for training (only if not in eval mode)
    start_epoch = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Set up optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load optimizer state if resuming training
    if args.resume_from:
        checkpoint = torch.load(os.path.join(args.resume_from, 'model_optimizer.pth'))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Main training loop.
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Create backup every N epochs
        if (epoch + 1) % args.backup_interval == 0:
            backup_path = create_backup(args.output_dir, epoch + 1, model, optimizer, 
                                       train_losses, train_accs, val_losses, val_accs)
        
        # Compute and save detailed metrics for validation set
        val_metrics = compute_metrics_with_progress(model, val_loader, criterion, device, class_names)
        save_metrics(val_metrics, args.output_dir, prefix=f'val_epoch_{epoch+1}_')
        
        # Clear GPU cache after each epoch if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/3600:.2f} hours")
    
    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    
    # If post-training quantization is enabled, quantize the model weights.
    if args.post_quantize:
        print(f"Applying post-training quantization with {args.quant_bits}-bit precision.")
        model, scale, zero_point = post_training_quantize_model(model, args.quant_bits)
    
    # Save the final model.
    model_path = os.path.join(args.output_dir, "vit_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save the quantization parameters.
    if args.post_quantize:
        quant_params_path = os.path.join(args.output_dir, "quant_params.pth")
        torch.save({
            'scale': scale,
            'zero_point': zero_point
        }, quant_params_path)
        print(f"Quantization parameters saved to {quant_params_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = compute_metrics_with_progress(model, test_loader, criterion, device, class_names)
    save_metrics(test_metrics, args.output_dir, prefix='test_final_')
    
    # Print final test results
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
    
    print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs saved to {args.output_dir}")

def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training and validation loss and accuracy."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save the figure
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(metrics_path)
    print(f"Training metrics plot saved as '{metrics_path}'")
    plt.close()

if __name__ == '__main__':
    main()
