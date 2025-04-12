import torch
import torchviz
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from model import VisionTransformer
import numpy as np
from torchinfo import summary
import os
import sys
import matplotlib.patches as patches

def create_model_summary():
    # Create a sample model
    model = VisionTransformer(
        img_size=32,
        patch_size=16,
        in_chans=3,
        num_classes=100,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    # Generate model summary
    model_stats = summary(model, input_size=(1, 3, 32, 32), verbose=0)
    return model, model_stats

def visualize_model_architecture(model):
    try:
        # Create a sample input
        x = torch.randn(1, 3, 32, 32)
        
        # Generate model graph
        dot = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
        
        dot.render("model_architecture", format="png", cleanup=True)
        print("Model architecture visualization saved as 'model_architecture.png'")
        return True
    except Exception as e:
        print(f"Error generating model visualization: {str(e)}")
        print("Falling back to alternative visualization method...")
        return False

def visualize_model_alternative(model):
    """Alternative visualization method using matplotlib"""
    # Create a simple layer visualization
    plt.figure(figsize=(15, 10))
    
    # Get layer information
    layers = []
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
            layers.append(name)
            params.append(sum(p.numel() for p in module.parameters()))
    
    # Plot layer sizes
    plt.subplot(2, 1, 1)
    plt.bar(layers, params)
    plt.xticks(rotation=45, ha='right')
    plt.title('Number of Parameters per Layer')
    plt.ylabel('Parameters')
    
    # Plot layer types distribution
    plt.subplot(2, 1, 2)
    layer_types = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    plt.bar(layer_types.keys(), layer_types.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Layer Types')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('model_architecture_alternative.png')
    print("Alternative model visualization saved as 'model_architecture_alternative.png'")

def analyze_parameters(model):
    # Collect parameter statistics
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item(),
            'shape': param.data.shape
        }
    
    # Count how many weight matrices we have (parameters with shape > 1)
    weight_params = [name for name, stats in param_stats.items() if len(stats['shape']) > 1]
    num_weight_params = len(weight_params)
    
    # Limit to 64 subplots (matplotlib's maximum)
    max_subplots = 64
    if num_weight_params > max_subplots:
        print(f"Warning: Model has {num_weight_params} weight parameters, but only showing the first {max_subplots} due to matplotlib limitations.")
        weight_params = weight_params[:max_subplots]
        num_weight_params = max_subplots
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_weight_params)))
    
    # Create parameter distribution plots
    plt.figure(figsize=(15, 15))
    for i, name in enumerate(weight_params):
        param = model.state_dict()[name]
        stats = param_stats[name]
        plt.subplot(grid_size, grid_size, i+1)
        sns.histplot(param.flatten().cpu().numpy(), bins=50)
        plt.title(f'{name}\nμ={stats["mean"]:.3f}, σ={stats["std"]:.3f}')
        plt.tight_layout()
    
    plt.savefig('parameter_distributions.png')
    print("Parameter distribution plots saved as 'parameter_distributions.png'")
    
    return param_stats

def create_simplistic_plot(model):
    """Create a simplistic plot showing parameter distribution across layers"""
    plt.figure(figsize=(10, 6))
    
    # Get parameter means for each layer
    layer_names = []
    param_means = []
    
    for name, param in model.named_parameters():
        if len(param.shape) > 1:  # Only consider weight matrices, not biases
            layer_names.append(name)
            param_means.append(param.data.mean().item())
    
    # Create a simple line plot
    plt.plot(range(len(layer_names)), param_means, 'o-', linewidth=2, markersize=8)
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.title('Parameter Mean Values Across Layers')
    plt.ylabel('Mean Parameter Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('simplistic_plot.png')
    print("Simplistic plot saved as 'simplistic_plot.png'")
    
    return plt.gcf()

def create_simplistic_model_graph():
    """Create a simplistic diagram explaining the Vision Transformer architecture"""
    plt.figure(figsize=(12, 8))
    
    # Set up the canvas
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    plt.text(5, 9.5, 'Vision Transformer Architecture', fontsize=16, ha='center', fontweight='bold')
    
    # Input image
    input_rect = patches.Rectangle((1, 7), 2, 1.5, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.5)
    ax.add_patch(input_rect)
    plt.text(2, 7.75, 'Input Image\n(32x32x3)', ha='center', va='center')
    
    # Patch embedding   
    patch_rect = patches.Rectangle((4, 7), 2, 1.5, linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(patch_rect)
    plt.text(5, 7.75, 'Patch Embedding\n(14x14x768)', ha='center', va='center')
    
    # Position embedding
    pos_rect = patches.Rectangle((7, 7), 2, 1.5, linewidth=2, edgecolor='black', facecolor='orange', alpha=0.5)
    ax.add_patch(pos_rect)
    plt.text(8, 7.75, 'Position Embedding', ha='center', va='center')
    
    # Arrows
    plt.arrow(3, 7.75, 0.8, 0, head_width=0.1, head_length=0.2, fc='black', ec='black')
    plt.arrow(6, 7.75, 0.8, 0, head_width=0.1, head_length=0.2, fc='black', ec='black')
    
    # Transformer blocks
    transformer_rect = patches.Rectangle((3, 4), 4, 2, linewidth=2, edgecolor='black', facecolor='lightcoral', alpha=0.5)
    ax.add_patch(transformer_rect)
    plt.text(5, 5, 'Transformer Blocks\n(Self-Attention + MLP)', ha='center', va='center')
    
    # Arrow from patch embedding to transformer
    plt.arrow(5, 7, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')
    
    # MLP head
    mlp_rect = patches.Rectangle((3, 1), 4, 2, linewidth=2, edgecolor='black', facecolor='plum', alpha=0.5)
    ax.add_patch(mlp_rect)
    plt.text(5, 2, 'MLP Head\n(Classification)', ha='center', va='center')
    
    # Arrow from transformer to MLP head
    plt.arrow(5, 4, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='lightblue', edgecolor='black', label='Input Processing'),
        patches.Patch(facecolor='lightgreen', edgecolor='black', label='Feature Extraction'),
        patches.Patch(facecolor='orange', edgecolor='black', label='Position Information'),
        patches.Patch(facecolor='lightcoral', edgecolor='black', label='Transformer Processing'),
        patches.Patch(facecolor='plum', edgecolor='black', label='Classification')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('simplistic_model_graph.png', dpi=300, bbox_inches='tight')
    print("Simplistic model graph saved as 'simplistic_model_graph.png'")
    
    return plt.gcf()

def main():
    # Create model and get summary
    model, model_stats = create_model_summary()
    
    # Try to visualize architecture with Graphviz
    if not visualize_model_architecture(model):
        # If Graphviz visualization fails, use alternative method
        visualize_model_alternative(model)
    
    # Create simplistic model graph
    create_simplistic_model_graph()
    
    # Analyze parameters
    param_stats = analyze_parameters(model)
    
    # Create simplistic plot
    create_simplistic_plot(model)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("\nModel Architecture:")
    print(model_stats)

if __name__ == "__main__":
    main() 