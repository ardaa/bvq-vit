import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import VisionTransformer
from utils import load_checkpoint, get_dataloader, get_imagenet_dataloaders
import seaborn as sns
import os

def evaluate_model_with_ablated_layer(model, dataloader, device, layer_idx=None):
    """
    Evaluate the model with a specific layer ablated (zeroed out).
    If layer_idx is None, evaluates the original model.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Store original weights if we're ablating a layer
    original_weights = None
    if layer_idx is not None:
        # Zero out the specified layer
        original_weights = model.blocks[layer_idx].state_dict()
        for param in model.blocks[layer_idx].parameters():
            param.data.zero_()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Restore original weights if we ablated a layer
    if layer_idx is not None:
        model.blocks[layer_idx].load_state_dict(original_weights)
    
    return 100. * correct / total

def perform_ablation_study(model, dataloader, device):
    """
    Perform ablation study on all layers of the model.
    """
    # Get baseline performance
    baseline_acc = evaluate_model_with_ablated_layer(model, dataloader, device)
    
    # Test each layer
    num_layers = len(model.blocks)
    accuracies = []
    
    for layer_idx in tqdm(range(num_layers), desc="Performing ablation study"):
        acc = evaluate_model_with_ablated_layer(model, dataloader, device, layer_idx)
        accuracies.append(acc)
    
    return baseline_acc, accuracies

def plot_ablation_results(baseline_acc, layer_accuracies):
    """
    Create a visualization of the ablation study results.
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    x = np.arange(len(layer_accuracies))
    accuracies = np.array(layer_accuracies)
    
    # Calculate importance as drop in accuracy
    importance = baseline_acc - accuracies
    
    # Create the plot
    sns.set_style("whitegrid")
    plt.bar(x, importance, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add baseline reference line
    plt.axhline(y=baseline_acc, color='g', linestyle='--', alpha=0.5, 
                label=f'Baseline Accuracy: {baseline_acc:.2f}%')
    
    # Customize the plot
    plt.xlabel('Layer Index')
    plt.ylabel('Performance Drop (%)')
    plt.title('Layer Importance Analysis\n(Performance Drop When Layer is Ablated)')
    plt.xticks(x)
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(importance):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('layer_importance.png')
    plt.close()

def infer_model_config(state_dict):
    """Infer model configuration from the state dict."""
    # Default configuration
    model_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
        'attn_dropout': 0.0
    }
    
    # Infer image size and patch size from pos_embed
    if 'pos_embed' in state_dict:
        pos_embed_shape = state_dict['pos_embed'].shape
        num_patches = pos_embed_shape[1] - 1  # -1 for cls token
        embed_dim = pos_embed_shape[2]
        
        # Update embed_dim
        model_config['embed_dim'] = embed_dim
        
        # Try to infer patch size from state_dict
        if 'patch_embed.proj.weight' in state_dict:
            patch_size = state_dict['patch_embed.proj.weight'].shape[2]
            model_config['patch_size'] = patch_size
            
            # Infer image size from number of patches and patch size
            img_size = int(np.sqrt(num_patches) * patch_size)
            model_config['img_size'] = img_size
    
    # Infer number of classes from head weight
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
        model_config['num_classes'] = num_classes
    
    # Infer depth (number of transformer blocks)
    depth = 0
    while f'blocks.{depth}.norm1.weight' in state_dict:
        depth += 1
    if depth > 0:
        model_config['depth'] = depth
    
    # Infer number of heads from the first block
    if 'blocks.0.attn.in_proj_weight' in state_dict:
        in_proj_weight = state_dict['blocks.0.attn.in_proj_weight']
        embed_dim = model_config['embed_dim']
        # Each head typically has dimension embed_dim // num_heads
        # This is a heuristic based on common implementations
        for num_heads in [8, 12, 16]:
            if embed_dim % num_heads == 0 and embed_dim // num_heads * 3 == in_proj_weight.shape[0]:
                model_config['num_heads'] = num_heads
                break
    
    return model_config

def load_model(model_path, device):
    """Load a model from a file."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's already a model (should not happen, but just in case)
        if isinstance(state_dict, torch.nn.Module):
            print("Loaded object is already a model")
            return state_dict
        
        # If it's a checkpoint with model_state_dict, extract it
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            print("Found model_state_dict in checkpoint")
            state_dict = state_dict['model_state_dict']
        
        # Infer model configuration from the state dict
        model_config = infer_model_config(state_dict)
        print(f"Inferred model configuration: {model_config}")
        
        # Create the model and load state dict
        model = VisionTransformer(**model_config)
        model.load_state_dict(state_dict)
        model.to(device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model correctly using our load_model function
    checkpoint_path = 'model.pth'
    model = load_model(checkpoint_path, device)
    
    # if dataset does not exist, download it
    if not os.path.exists('./data'):
        print("Downloading ImageNet dataset...")
        get_imagenet_dataloaders('./data', 128, 224)
    # Get validation dataloader
    _, val_loader = get_dataloader('./data', 128, 224)
    
    # Perform ablation study
    print("Starting ablation study...")
    baseline_acc, layer_accuracies = perform_ablation_study(model, val_loader, device)
    
    # Plot results
    print("Creating visualization...")
    plot_ablation_results(baseline_acc, layer_accuracies)
    print("Results saved as 'layer_importance.png'")

if __name__ == '__main__':
    main()
