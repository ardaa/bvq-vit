import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import copy

def get_imagenet_dataloaders(data_path, batch_size, img_size):
    # Data augmentation and normalization for training.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Normalization for validation and test.
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=eval_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def compute_metrics(model, dataloader, criterion, device, class_names=None):
    """Compute comprehensive metrics for model evaluation."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    running_loss = 0.0
    correct = 0
    total = 0
    
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
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    accuracy = correct / total
    avg_loss = running_loss / total
    
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

def load_checkpoint(model, checkpoint_path):
    """Load a checkpoint from a file."""
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def get_dataloader(data_path, batch_size, img_size):
    train_loader, val_loader, test_loader = get_imagenet_dataloaders(data_path, batch_size, img_size)
    return train_loader, val_loader

def save_metrics(metrics, output_dir, prefix=''):
    """Save metrics to files in the output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = os.path.join(metrics_dir, f'{prefix}metrics_{timestamp}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(metrics['confusion_matrix'], cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(metrics_dir, f'{prefix}confusion_matrix_{timestamp}.png'))
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for class_name, roc_info in metrics['roc_data'].items():
        plt.plot(roc_info['fpr'], roc_info['tpr'], 
                 label=f'{class_name} (AUC = {roc_info["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(metrics_dir, f'{prefix}roc_curves_{timestamp}.png'))
    plt.close()
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    classes = list(metrics['per_class_accuracy'].keys())
    accuracies = list(metrics['per_class_accuracy'].values())
    plt.bar(classes, accuracies)
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f'{prefix}per_class_accuracy_{timestamp}.png'))
    plt.close()
    
    print(f"Metrics saved to {metrics_dir}")
    return metrics_file

#####################
# Quantization Utils
#####################

def compute_quant_params(x, num_bits):
    """
    Compute the scale and zero point for quantization.
    For simplicity, we use the minimum value as the zero point.
    """
    qmin = 0
    qmax = 2**num_bits - 1
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / (qmax - qmin + 1e-8)
    zero_point = x_min
    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits):
    """
    Quantize the tensor x using the computed scale and zero point.
    """
    qmin = 0
    qmax = 2**num_bits - 1
    x_q = torch.clamp(torch.round((x - zero_point) / scale), qmin, qmax)
    return x_q

def dequantize_tensor(x_q, scale, zero_point):
    """
    Dequantize the tensor from its quantized representation.
    """
    return x_q * scale + zero_point

def quantize(x, num_bits):
    """
    Quantization by computing quantization parameters, quantizing x, and dequantizing it.
    """
    scale, zero_point = compute_quant_params(x, num_bits)
    x_q = quantize_tensor(x, scale, zero_point, num_bits)
    return x_q, scale, zero_point

def dequantize(x_q, scale, zero_point):
    """
    Dequantize the tensor from its quantized representation.
    """
    return dequantize_tensor(x_q, scale, zero_point)

def post_training_quantize_model(model, quant_bits):
    """
    Quantize the model weights to lower precision.
    
    Returns:
        - quantized_model: The model with quantized weights
        - inference_model: A model that simulates quantized inference
        - quant_info: A dictionary containing quantization parameters for each layer
    """
    # Create a deep copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    quant_info = {}
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Quantize weights
            weight_q, scale, zero_point = quantize(module.weight.data, quant_bits)
            
            # Store quantized weights (important: we don't dequantize)
            module.weight.data = weight_q
            
            # Store quantization parameters for inference
            quant_info[f"{name}.weight"] = {
                'scale': scale,
                'zero_point': zero_point,
                'bits': quant_bits
            }
            
            # Also quantize bias if it exists
            if module.bias is not None:
                bias_q, bias_scale, bias_zero_point = quantize(module.bias.data, quant_bits)
                module.bias.data = bias_q
                quant_info[f"{name}.bias"] = {
                    'scale': bias_scale,
                    'zero_point': bias_zero_point,
                    'bits': quant_bits
                }
    
    # Create a model for quantized inference (with hooks)
    inference_model = copy.deepcopy(model)
    
    # Store the quantization parameters in the inference model for access in hooks
    inference_model.quant_info = quant_info
    inference_model.quant_bits = quant_bits
    
    # Define a hook that simulates quantized inference
    def add_quantization_hook(name):
        def hook(module, input, output):
            # Simulate activation quantization
            activation_q, act_scale, act_zero_point = quantize(output, quant_bits)
            
            # Simulate dequantization (converting back to floating point)
            # This simulates what would happen in hardware when reading quantized values
            dequantized_output = dequantize(activation_q, act_scale, act_zero_point)
            
            # Store quantization parameters for analysis
            if not hasattr(inference_model, 'activation_quant_info'):
                inference_model.activation_quant_info = {}
            
            inference_model.activation_quant_info[name] = {
                'scale': act_scale,
                'zero_point': act_zero_point,
                'bits': quant_bits
            }
            
            return dequantized_output
        
        return hook
    
    # Add hooks to the inference model
    for name, module in inference_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Replace weights with the quantized weights
            weight_info = quant_info.get(f"{name}.weight")
            if weight_info:
                # Get the quantized weights from the quantized model
                quantized_weight = quantized_model.get_submodule(name).weight.data
                # Dequantize for the inference model
                dequantized_weight = dequantize(
                    quantized_weight, 
                    weight_info['scale'], 
                    weight_info['zero_point']
                )
                module.weight.data = dequantized_weight
                
                # Similarly for bias
                if module.bias is not None:
                    bias_info = quant_info.get(f"{name}.bias")
                    if bias_info:
                        quantized_bias = quantized_model.get_submodule(name).bias.data
                        dequantized_bias = dequantize(
                            quantized_bias,
                            bias_info['scale'],
                            bias_info['zero_point']
                        )
                        module.bias.data = dequantized_bias
            
            # Register hook to simulate activation quantization
            module.register_forward_hook(add_quantization_hook(name))
    
    return quantized_model, inference_model, quant_info

def verify_quantization(model, quant_bits):
    """
    Verify that the model is properly quantized by checking the values in the weights.
    For a model quantized to N bits, we expect to find at most 2^N unique values.
    
    Returns:
        bool: True if the model appears to be quantized correctly, False otherwise
    """
    expected_max_unique = 2**quant_bits
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Count unique values in the weight tensor
            unique_values = torch.unique(module.weight.data)
            num_unique = len(unique_values)
            
            print(f"Layer {name}: {num_unique} unique values (max expected: {expected_max_unique})")
            
            # If we find any layer with more unique values than expected, it's not properly quantized
            if num_unique > expected_max_unique:
                print(f"  WARNING: Layer {name} has {num_unique} unique values, which is more than the expected {expected_max_unique} for {quant_bits}-bit quantization")
                return False
    
    return True