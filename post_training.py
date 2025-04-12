import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import VisionTransformer
from utils import (
    get_imagenet_dataloaders, 
    compute_metrics, 
    save_metrics,
    post_training_quantize_model,
    verify_quantization,
    dequantize
)

def load_model(model_path, model_config=None):
    """Load the model from a checkpoint file."""
    print(f"Loading model from {model_path}...")
    
    # First load the checkpoint to inspect its structure
    checkpoint = torch.load(model_path, map_location='cuda')
    
    # If model_config is not provided, try to infer it from the checkpoint
    if model_config is None:
        # Default ImageNet configuration
        model_config = {
            'img_size': 32,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 10,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'attn_dropout': 0.0
        }
        
        # Try to infer configuration from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Infer image size and patch size from pos_embed
        if 'pos_embed' in state_dict:
            pos_embed_shape = state_dict['pos_embed'].shape
            num_patches = pos_embed_shape[1] - 1  # -1 for cls token
            img_size = int(np.sqrt(num_patches) * model_config['patch_size'])
            model_config['img_size'] = img_size
            
        # Infer number of classes from head weight
        if 'head.weight' in state_dict:
            num_classes = state_dict['head.weight'].shape[0]
            model_config['num_classes'] = num_classes
            
        # Infer patch size from patch_embed.proj.weight
        if 'patch_embed.proj.weight' in state_dict:
            patch_size = state_dict['patch_embed.proj.weight'].shape[2]
            model_config['patch_size'] = patch_size
    
    print(f"Using model configuration: {model_config}")
    
    # Create model with the configuration
    model = VisionTransformer(**model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, model_config

def quantize_model(model, quant_bits):
    """Quantize the model using post-training quantization."""
    print(f"Quantizing model to {quant_bits} bits...")
    quantized_model, inference_model, quant_info = post_training_quantize_model(model, quant_bits)
    print(f"Quantized model created with {len(quant_info)} quantized layers")
    return quantized_model, inference_model, quant_info

def evaluate_model(model, dataloader, criterion, device, num_classes):
    """Evaluate the model on the given dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
    
    accuracy = correct / total
    avg_loss = running_loss / total
    
    # Calculate per-class accuracies
    per_class_accuracies = []
    for i in range(num_classes):
        class_correct = confusion[i, i]
        class_total = np.sum(confusion[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        per_class_accuracies.append(class_accuracy)

    # Calculate top-5 accuracy - only compute if there are at least 5 classes
    top5_accuracy = None
    if num_classes >= 5:
        top5_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Computing Top-5 Accuracy"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # Make sure we don't request more classes than we have
                k = min(5, num_classes)
                _, top5_preds = torch.topk(outputs, k, dim=1)
                for i, target in enumerate(targets):
                    if target in top5_preds[i]:
                        top5_correct += 1
        top5_accuracy = top5_correct / total
        print(f"Top-{k} Accuracy: {top5_accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'top5_accuracy': top5_accuracy,
        'confusion_matrix': confusion.tolist(),
        'per_class_accuracies': per_class_accuracies
    }

def measure_inference_time(model, dataloader, device, num_batches=100):
    """Measure the inference time of the model."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Measuring inference time")):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Warm-up
            if i == 0:
                for _ in range(10):
                    _ = model(inputs)
            
            # Measure inference time
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    
    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'throughput': 1.0 / avg_time  # samples per second
    }

def collect_activations(model, dataloader, device, num_batches=5):
    """
    Collect activations from a model with hooks.
    
    Args:
        model: PyTorch model with hooks
        dataloader: DataLoader to get inputs
        device: Device to run on
        num_batches: Number of batches to collect activations from
    
    Returns:
        Dict mapping layer names to activation tensors
    """
    activations = {}
    activation_hooks = []
    
    # Register hooks to collect activations
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            # Convert to numpy for easier storage and processing
            if isinstance(output, torch.Tensor):
                # Store a sample of activations (first 10 elements of first item in batch)
                # Avoiding storing all activations to prevent memory issues
                act = output[0].detach().cpu().flatten()[:1000].numpy()
                activations[name].append(act)
            return output
        return hook
    
    # Add hooks to the model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            activation_hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Collect activations
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Remove hooks
    for hook in activation_hooks:
        hook.remove()
    
    # Concatenate activations from different batches
    for name in activations:
        activations[name] = np.concatenate(activations[name])
    
    return activations

def plot_activation_histograms(original_activations, quantized_activations, output_dir, quant_bits):
    """
    Plot histograms comparing original and quantized activations.
    
    Args:
        original_activations: Dict mapping layer names to original activations
        quantized_activations: Dict mapping layer names to quantized activations
        output_dir: Directory to save plots
        quant_bits: Number of bits used for quantization
    """
    os.makedirs(os.path.join(output_dir, f'activation_histograms_{quant_bits}bit'), exist_ok=True)
    
    # Select a subset of layers to plot if there are many
    layers_to_plot = list(original_activations.keys())
    if len(layers_to_plot) > 10:
        # Take a representative sample: first, middle, and last few layers
        layers_to_plot = layers_to_plot[:3] + \
                          layers_to_plot[len(layers_to_plot)//2-1:len(layers_to_plot)//2+2] + \
                          layers_to_plot[-3:]
    
    for layer_name in layers_to_plot:
        if layer_name in quantized_activations:
            plt.figure(figsize=(12, 6))
            
            # Plot histograms
            plt.subplot(1, 2, 1)
            plt.hist(original_activations[layer_name], bins=50, alpha=0.7, label='Original')
            plt.hist(quantized_activations[layer_name], bins=50, alpha=0.7, label='Quantized')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.title(f'Activation Histograms: {layer_name} ({quant_bits}-bit)')
            plt.legend()
            
            # Plot difference
            plt.subplot(1, 2, 2)
            plt.hist(original_activations[layer_name] - quantized_activations[layer_name], bins=50)
            plt.xlabel('Difference (Original - Quantized)')
            plt.ylabel('Frequency')
            plt.title('Quantization Error Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'activation_histograms_{quant_bits}bit', f'{layer_name.replace(".", "_")}_hist.png'))
            plt.close()
    
    # Create a summary plot showing quantization effects across layers
    plt.figure(figsize=(12, 6))
    
    # Calculate mean absolute error for each layer
    mae_values = []
    layer_names = []
    
    for layer_name in original_activations:
        if layer_name in quantized_activations:
            mae = np.mean(np.abs(original_activations[layer_name] - quantized_activations[layer_name]))
            mae_values.append(mae)
            layer_names.append(layer_name.split('.')[-1])  # Simplify names for readability
    
    # Sort by MAE
    sorted_indices = np.argsort(mae_values)[::-1]  # Descending order
    sorted_mae = [mae_values[i] for i in sorted_indices]
    sorted_names = [layer_names[i] for i in sorted_indices]
    
    # Plot top 15 layers by quantization error
    plt.bar(range(min(15, len(sorted_mae))), sorted_mae[:15])
    plt.xticks(range(min(15, len(sorted_mae))), sorted_names[:15], rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Quantization Error by Layer ({quant_bits}-bit)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'activation_histograms_{quant_bits}bit', f'quantization_error_by_layer_{quant_bits}bit.png'))
    plt.close()

def compare_models(original_model, quantized_model, dataloader, device, output_dir, num_classes):
    """Compare the performance of the original and quantized models."""
    criterion = nn.CrossEntropyLoss()
    
    # Get the quantization bit depth from the model
    quant_bits = getattr(quantized_model, 'quant_bits', 8)  # Default to 8 if not found
    
    # Collect activations for a few batches
    print("Collecting activations from original model...")
    original_activations = collect_activations(original_model, dataloader, device)
    
    print("Collecting activations from quantized model...")
    quantized_activations = collect_activations(quantized_model, dataloader, device)
    
    # Plot activation histograms
    print("Plotting activation histograms...")
    plot_activation_histograms(
        original_activations, 
        quantized_activations, 
        output_dir,
        quant_bits
    )
    
    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, dataloader, criterion, device, num_classes)
    
    # Evaluate quantized model
    print("Evaluating quantized model...")
    quantized_metrics = evaluate_model(quantized_model, dataloader, criterion, device, num_classes)
    
    # Measure inference time
    print("Measuring inference time for original model...")
    original_time = measure_inference_time(original_model, dataloader, device)
    
    print("Measuring inference time for quantized model...")
    quantized_time = measure_inference_time(quantized_model, dataloader, device)
    
    # Combine metrics
    comparison = {
        'quant_bits': quant_bits,
        'original': {
            'accuracy': original_metrics['accuracy'],
            'loss': original_metrics['loss'],
            'inference_time': original_time,
            'per_class_accuracies': original_metrics['per_class_accuracies']
        },
        'quantized': {
            'accuracy': quantized_metrics['accuracy'],
            'loss': quantized_metrics['loss'],
            'inference_time': quantized_time,
            'per_class_accuracies': quantized_metrics['per_class_accuracies']
        }
    }
    
    # Calculate differences
    comparison['differences'] = {
        'accuracy_diff': quantized_metrics['accuracy'] - original_metrics['accuracy'],
        'loss_diff': quantized_metrics['loss'] - original_metrics['loss'],
        'speedup': original_time['avg_inference_time'] / quantized_time['avg_inference_time'],
        'per_class_accuracy_diffs': [q - o for q, o in zip(
            quantized_metrics['per_class_accuracies'], 
            original_metrics['per_class_accuracies']
        )]
    }
    
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'quantization_comparison_{quant_bits}bit.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Plot comparison
    plot_comparison(comparison, output_dir, quant_bits)
    
    # Plot confusion matrices
    plot_confusion_matrices(original_metrics['confusion_matrix'], 
                          quantized_metrics['confusion_matrix'], 
                          output_dir,
                          quant_bits)
    
    # Plot per-class accuracies
    plot_per_class_accuracies(
        original_metrics['per_class_accuracies'],
        quantized_metrics['per_class_accuracies'], 
        comparison['differences']['per_class_accuracy_diffs'],
        output_dir,
        quant_bits
    )
    
    return comparison

def plot_comparison(comparison, output_dir, quant_bits):
    """Plot the comparison between original and quantized models."""
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ['Original', f'Quantized ({quant_bits}-bit)']
    accuracies = [comparison['original']['accuracy'], comparison['quantized']['accuracy']]
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{quant_bits}bit.png'))
    
    # Inference time comparison
    plt.figure(figsize=(10, 6))
    times = [comparison['original']['inference_time']['avg_inference_time'], 
             comparison['quantized']['inference_time']['avg_inference_time']]
    plt.bar(models, times, color=['blue', 'green'])
    plt.ylabel('Average Inference Time (s)')
    plt.title('Model Inference Time Comparison')
    for i, v in enumerate(times):
        plt.text(i, v + 0.001, f'{v:.6f}', ha='center')
    plt.savefig(os.path.join(output_dir, f'inference_time_comparison_{quant_bits}bit.png'))
    
    # Throughput comparison
    plt.figure(figsize=(10, 6))
    throughputs = [comparison['original']['inference_time']['throughput'], 
                  comparison['quantized']['inference_time']['throughput']]
    plt.bar(models, throughputs, color=['blue', 'green'])
    plt.ylabel('Throughput (samples/s)')
    plt.title('Model Throughput Comparison')
    for i, v in enumerate(throughputs):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.savefig(os.path.join(output_dir, f'throughput_comparison_{quant_bits}bit.png'))
    
    plt.close('all')

def plot_confusion_matrices(original_confusion, quantized_confusion, output_dir, quant_bits):
    """Plot confusion matrices for original and quantized models."""
    # Original model confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(original_confusion, cmap='Blues')
    plt.colorbar()
    plt.title('Original Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'original_confusion_matrix_{quant_bits}bit.png'))
    
    # Quantized model confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(quantized_confusion, cmap='Blues')
    plt.colorbar()
    plt.title(f'Quantized Model Confusion Matrix ({quant_bits}-bit)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'quantized_confusion_matrix_{quant_bits}bit.png'))
    
    plt.close('all')

def plot_per_class_accuracies(original_accs, quantized_accs, diffs, output_dir, quant_bits):
    """Plot per-class accuracies for original and quantized models."""
    num_classes = len(original_accs)
    
    # Plot accuracies
    plt.figure(figsize=(15, 8))
    x = np.arange(num_classes)
    width = 0.35
    
    plt.bar(x - width/2, original_accs, width, label='Original')
    plt.bar(x + width/2, quantized_accs, width, label=f'Quantized ({quant_bits}-bit)')
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, [str(i) for i in range(num_classes)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add accuracy values on top of bars
    for i, (orig, quant) in enumerate(zip(original_accs, quantized_accs)):
        plt.text(i - width/2, orig + 0.01, f'{orig:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, quant + 0.01, f'{quant:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(os.path.join(output_dir, f'per_class_accuracy_{quant_bits}bit.png'))
    
    # Plot differences
    plt.figure(figsize=(15, 8))
    colors = ['red' if d < 0 else 'green' for d in diffs]
    bars = plt.bar(x, diffs, color=colors)
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy Difference (Quantized - Original)')
    plt.title(f'Per-Class Accuracy Differences ({quant_bits}-bit)')
    plt.xticks(x, [str(i) for i in range(num_classes)])
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Add difference values on top of bars
    for i, d in enumerate(diffs):
        plt.text(i, d + (0.01 if d >= 0 else -0.02), 
                f'{d:.3f}', ha='center', va='bottom' if d >= 0 else 'top', 
                fontsize=9, color='black')
    
    plt.savefig(os.path.join(output_dir, f'per_class_accuracy_diffs_{quant_bits}bit.png'))
    
    # Add a heatmap visualization of per-class accuracy changes
    plt.figure(figsize=(12, 6))
    data = np.vstack((original_accs, quantized_accs))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Accuracy')
    plt.yticks([0, 1], ['Original', f'Quantized ({quant_bits}-bit)'])
    plt.xticks(range(num_classes), [str(i) for i in range(num_classes)])
    plt.xlabel('Class')
    plt.title(f'Accuracy Heatmap by Class ({quant_bits}-bit)')
    
    # Add text annotations
    for i in range(2):
        for j in range(num_classes):
            value = data[i, j]
            text_color = 'white' if value < 0.7 else 'black'
            plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)
    
    plt.savefig(os.path.join(output_dir, f'accuracy_heatmap_{quant_bits}bit.png'))
    
    # Save per-class accuracies to CSV
    with open(os.path.join(output_dir, f'per_class_accuracies_{quant_bits}bit.csv'), 'w') as f:
        f.write('Class,Original,Quantized,Difference\n')
        for i in range(num_classes):
            f.write(f'{i},{original_accs[i]:.4f},{quantized_accs[i]:.4f},{diffs[i]:.4f}\n')
    
    plt.close('all')

def detect_dataset(data_path):
    """Detect the dataset being used and return its properties."""
    if data_path is None:
        print("No data path provided, using CIFAR-10 as fallback")
        return {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'img_size': 32
        }
    else:
        # Check if ImageNet directory structure exists
        if os.path.exists(os.path.join(data_path, 'train')) and os.path.exists(os.path.join(data_path, 'val')):
            print("ImageNet dataset detected")
            return {
                'name': 'ImageNet',
                'num_classes': 10,
                'img_size': 224
            }
        else:
            print("Unknown dataset structure, using CIFAR-10 as fallback")
            return {
                'name': 'CIFAR-10',
                'num_classes': 10,
                'img_size': 32
            }

def calculate_model_size(model, bits_per_param=32):
    """
    Calculate the model size in bytes and MB.
    
    Args:
        model: PyTorch model
        bits_per_param: Number of bits per parameter (32 for float32, 8 for int8, etc.)
    
    Returns:
        tuple: (size_bytes, size_mb)
    """
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = num_params * (bits_per_param / 8)  # Convert bits to bytes
    size_mb = size_bytes / (1024 * 1024)  # Convert bytes to MB
    return size_bytes, size_mb

def plot_model_size_comparison(original_size, quantized_size, quant_bits, output_dir):
    """Plot model size comparison between original and quantized models."""
    plt.figure(figsize=(8, 6))
    sizes = [original_size, quantized_size]
    labels = ['Original (FP32)', f'Quantized ({quant_bits}-bit)']
    colors = ['blue', 'green']
    
    plt.bar(labels, sizes, color=colors)
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    
    # Add reduction percentage
    reduction = (1 - quantized_size/original_size) * 100
    plt.text(1, quantized_size/2, f"{reduction:.1f}% reduction", 
             ha='center', va='center', fontweight='bold')
    
    # Add size labels on top of bars
    for i, size in enumerate(sizes):
        plt.text(i, size + (original_size * 0.02), f"{size:.2f} MB", 
                 ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, f'model_size_comparison_{quant_bits}bit.png'))
    plt.close()

def compare_bit_depths(bit_depth_results, output_dir):
    """
    Create visualizations comparing results across different bit depths.
    
    Args:
        bit_depth_results: Dict mapping bit depths to their results
        output_dir: Directory to save comparative visualizations
    """
    accuracy_data = []
    inference_time_data = []
    throughput_data = []
    model_size_data = []
    bit_depths = sorted(bit_depth_results.keys())
    
    for bits in bit_depths:
        result = bit_depth_results[bits]
        accuracy_data.append(result['accuracy'])
        inference_time_data.append(result['inference_time'])
        throughput_data.append(result['throughput'])
        model_size_data.append(result['model_size'])
    
    # Comparative plots directory
    comp_dir = os.path.join(output_dir, 'comparative_analysis')
    os.makedirs(comp_dir, exist_ok=True)
    
    # Plot accuracy vs bit depth
    plt.figure(figsize=(10, 6))
    plt.plot(bit_depths, [acc['original'] for acc in accuracy_data], 'bo-', label='Original (FP32)')
    plt.plot(bit_depths, [acc['quantized'] for acc in accuracy_data], 'go-', label='Quantized')
    plt.xlabel('Bit Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Bit Depth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(bit_depths)
    plt.savefig(os.path.join(comp_dir, 'accuracy_vs_bit_depth.png'))
    plt.close()
    
    # Plot inference time vs bit depth
    plt.figure(figsize=(10, 6))
    plt.plot(bit_depths, [t['original'] for t in inference_time_data], 'bo-', label='Original (FP32)')
    plt.plot(bit_depths, [t['quantized'] for t in inference_time_data], 'go-', label='Quantized')
    plt.xlabel('Bit Depth')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs. Bit Depth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(bit_depths)
    plt.savefig(os.path.join(comp_dir, 'inference_time_vs_bit_depth.png'))
    plt.close()
    
    # Plot throughput vs bit depth
    plt.figure(figsize=(10, 6))
    plt.plot(bit_depths, [t['original'] for t in throughput_data], 'bo-', label='Original (FP32)')
    plt.plot(bit_depths, [t['quantized'] for t in throughput_data], 'go-', label='Quantized')
    plt.xlabel('Bit Depth')
    plt.ylabel('Throughput (samples/s)')
    plt.title('Throughput vs. Bit Depth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(bit_depths)
    plt.savefig(os.path.join(comp_dir, 'throughput_vs_bit_depth.png'))
    plt.close()
    
    # Plot model size vs bit depth
    plt.figure(figsize=(10, 6))
    # Plot horizontal line for original model size (which is constant)
    plt.axhline(y=model_size_data[0]['original'], color='b', linestyle='-', label='Original (FP32)')
    plt.plot(bit_depths, [size['quantized'] for size in model_size_data], 'go-', label='Quantized')
    plt.xlabel('Bit Depth')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size vs. Bit Depth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(bit_depths)
    plt.savefig(os.path.join(comp_dir, 'model_size_vs_bit_depth.png'))
    plt.close()
    
    # Plot accuracy vs model size (Pareto frontier)
    plt.figure(figsize=(10, 6))
    sizes = [size['quantized'] for size in model_size_data]
    accuracies = [acc['quantized'] for acc in accuracy_data]
    
    for i, bits in enumerate(bit_depths):
        plt.scatter(sizes[i], accuracies[i], s=100, label=f'{bits}-bit')
    
    plt.scatter(model_size_data[0]['original'], accuracy_data[0]['original'], 
               s=100, marker='*', color='red', label='Original (FP32)')
    
    # Add annotations
    for i, bits in enumerate(bit_depths):
        plt.annotate(f'{bits}-bit', 
                   (sizes[i], accuracies[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Model Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(comp_dir, 'accuracy_vs_model_size.png'))
    plt.close()
    
    # Create a comprehensive comparison table (CSV)
    with open(os.path.join(comp_dir, 'bit_depth_comparison.csv'), 'w') as f:
        f.write('Bit Depth,Original Accuracy,Quantized Accuracy,Accuracy Drop,Original Size (MB),'
               'Quantized Size (MB),Size Reduction %,Original Inference (s),Quantized Inference (s),'
               'Speedup,Original Throughput,Quantized Throughput\n')
        
        for i, bits in enumerate(bit_depths):
            acc_orig = accuracy_data[i]['original']
            acc_quant = accuracy_data[i]['quantized']
            acc_drop = acc_orig - acc_quant
            
            size_orig = model_size_data[i]['original']
            size_quant = model_size_data[i]['quantized']
            size_reduction = (1 - size_quant/size_orig) * 100
            
            infer_orig = inference_time_data[i]['original']
            infer_quant = inference_time_data[i]['quantized']
            speedup = infer_orig / infer_quant
            
            tput_orig = throughput_data[i]['original']
            tput_quant = throughput_data[i]['quantized']
            
            f.write(f'{bits},{acc_orig:.4f},{acc_quant:.4f},{acc_drop:.4f},{size_orig:.2f},'
                   f'{size_quant:.2f},{size_reduction:.2f},{infer_orig:.6f},{infer_quant:.6f},'
                   f'{speedup:.2f},{tput_orig:.2f},{tput_quant:.2f}\n')
    
    # Create a summary JSON file
    summary = {
        'bit_depths': bit_depths,
        'accuracy': {
            'original': [acc['original'] for acc in accuracy_data],
            'quantized': [acc['quantized'] for acc in accuracy_data],
            'difference': [acc['original'] - acc['quantized'] for acc in accuracy_data]
        },
        'model_size': {
            'original': [size['original'] for size in model_size_data],
            'quantized': [size['quantized'] for size in model_size_data],
            'reduction_percent': [(1 - size['quantized']/size['original']) * 100 for size in model_size_data]
        },
        'inference_time': {
            'original': [t['original'] for t in inference_time_data],
            'quantized': [t['quantized'] for t in inference_time_data],
            'speedup': [t['original']/t['quantized'] for t in inference_time_data]
        },
        'throughput': {
            'original': [t['original'] for t in throughput_data],
            'quantized': [t['quantized'] for t in throughput_data],
            'improvement_percent': [(t['quantized']/t['original'] - 1) * 100 for t in throughput_data]
        }
    }
    
    with open(os.path.join(comp_dir, 'comparative_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Post-training quantization and performance comparison")
    parser.add_argument('--model-path', type=str, default='model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs/quantization',
                        help='Directory to save output files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Input image size (will be auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to use for inference time measurement')
    args = parser.parse_args()
    
    # Define bit depths to analyze
    bit_depths = [4, 6, 8, 10, 12, 16]
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model once
    model, model_config = load_model(args.model_path)
    
    # Calculate original model size once (FP32)
    _, original_size_mb = calculate_model_size(model)
    print(f"Original model size: {original_size_mb:.2f} MB (FP32)")
    
    # Detect dataset once
    dataset_info = detect_dataset(args.data_path)
    img_size = model_config['img_size']
    
    # Set number of classes to match the model
    num_classes = 100
    print(f"Model configuration: {num_classes} classes, image size {img_size}")
    print(f"Using dataset: {dataset_info['name']}")
    
    # Get dataloader once
    _, _, test_loader = get_imagenet_dataloaders(args.data_path, args.batch_size, img_size)
    
    # Dictionary to store results for each bit depth
    bit_depth_results = {}
    
    # Run analysis for each bit depth
    for bits in bit_depths:
        print(f"\n{'='*50}")
        print(f"Starting analysis for {bits}-bit quantization")
        print(f"{'='*50}")
        
        # Create bit-specific output directory
        bit_output_dir = os.path.join(args.output_dir, f'{bits}bit')
        os.makedirs(bit_output_dir, exist_ok=True)
        
        # Create a fresh copy of the model for each bit depth
        current_model = load_model(args.model_path)[0]
        current_model = current_model.to(args.device)
        
        # Quantize model
        print(f"Quantizing model to {bits} bits...")
        quantized_model, inference_model, quant_info = quantize_model(current_model, bits)
        
        # Calculate quantized model size
        _, quantized_size_mb = calculate_model_size(quantized_model, bits_per_param=bits)
        print(f"Quantized model size: {quantized_size_mb:.2f} MB ({bits}-bit)")
        print(f"Size reduction: {(1 - quantized_size_mb/original_size_mb) * 100:.2f}%")
        
        # Move models to device
        quantized_model = quantized_model.to(args.device)
        inference_model = inference_model.to(args.device)
        
        # Verify quantization
        print("\nVerifying quantization...")
        is_quantized = verify_quantization(quantized_model, bits)
        if not is_quantized:
            print(f"\nWARNING: The {bits}-bit model does not appear to be properly quantized!")
        else:
            print(f"\nQuantization verification passed for {bits}-bit model.")
        
        # Save quantization information
        with open(os.path.join(bit_output_dir, f'quantization_info_{bits}bit.json'), 'w') as f:
            # Convert tensors to float/int for JSON serialization
            serializable_info = {}
            for k, v in quant_info.items():
                serializable_info[k] = {
                    'scale': float(v['scale']),
                    'zero_point': float(v['zero_point']),
                    'bits': int(v['bits'])
                }
            json.dump(serializable_info, f, indent=4)
        
        # Add model size information to the output
        with open(os.path.join(bit_output_dir, f'model_size_comparison_{bits}bit.json'), 'w') as f:
            size_info = {
                'original': {
                    'size_mb': float(original_size_mb),
                    'bits_per_param': 32
                },
                'quantized': {
                    'size_mb': float(quantized_size_mb),
                    'bits_per_param': bits
                },
                'reduction_percent': float((1 - quantized_size_mb/original_size_mb) * 100)
            }
            json.dump(size_info, f, indent=4)
        
        # Plot model size comparison
        plot_model_size_comparison(original_size_mb, quantized_size_mb, bits, bit_output_dir)
        
        # Compare models - use inference_model for evaluation
        comparison = compare_models(current_model, inference_model, test_loader, args.device, bit_output_dir, num_classes)
        
        # Store the results for this bit depth
        bit_depth_results[bits] = {
            'accuracy': {
                'original': comparison['original']['accuracy'],
                'quantized': comparison['quantized']['accuracy']
            },
            'inference_time': {
                'original': comparison['original']['inference_time']['avg_inference_time'],
                'quantized': comparison['quantized']['inference_time']['avg_inference_time']
            },
            'throughput': {
                'original': comparison['original']['inference_time']['throughput'],
                'quantized': comparison['quantized']['inference_time']['throughput']
            },
            'model_size': {
                'original': original_size_mb,
                'quantized': quantized_size_mb
            }
        }
        
        # Print results
        print("\nQuantization Results:")
        print(f"Original Accuracy: {comparison['original']['accuracy']:.4f}")
        print(f"Quantized Accuracy: {comparison['quantized']['accuracy']:.4f}")
        print(f"Accuracy Difference: {comparison['differences']['accuracy_diff']:.4f}")
        print(f"Original Inference Time: {comparison['original']['inference_time']['avg_inference_time']:.6f} s")
        print(f"Quantized Inference Time: {comparison['quantized']['inference_time']['avg_inference_time']:.6f} s")
        print(f"Speedup: {comparison['differences']['speedup']:.2f}x")
        
        # Print model size comparison
        print(f"\nModel Size Comparison:")
        print(f"Original Model (FP32): {original_size_mb:.2f} MB")
        print(f"Quantized Model ({bits}-bit): {quantized_size_mb:.2f} MB")
        print(f"Size Reduction: {(1 - quantized_size_mb/original_size_mb) * 100:.2f}%")
        
        print(f"\nResults saved to {bit_output_dir}")
    
    # Create comparative visualizations across bit depths
    print("\nCreating comparative analysis across bit depths...")
    compare_bit_depths(bit_depth_results, args.output_dir)
    
    print(f"\nComparative analysis saved to {os.path.join(args.output_dir, 'comparative_analysis')}")
    print("\nQuantization analysis complete!")

if __name__ == "__main__":
    main()
