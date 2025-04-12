import torch
import argparse
import os
from model import VisionTransformer
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to a standalone model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the model")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Determine if this is a full checkpoint with model_state_dict or just a state_dict
    if 'model_state_dict' in checkpoint:
        print("Found model_state_dict in checkpoint")
        state_dict = checkpoint['model_state_dict']
        
        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'train_acc' in checkpoint and 'val_acc' in checkpoint:
            print(f"Training accuracy: {checkpoint['train_acc']:.4f}, Validation accuracy: {checkpoint['val_acc']:.4f}")
    else:
        print("Assuming checkpoint contains only model state_dict")
        state_dict = checkpoint
    
    # Infer model configuration from state_dict
    model_config = infer_model_config(state_dict)
    print(f"Inferred model configuration: {model_config}")
    
    # Create a new model with the inferred configuration
    try:
        model = VisionTransformer(**model_config)
        model.load_state_dict(state_dict)
        print("Model created and loaded successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Save the model as a standalone model file
    try:
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to: {args.output}")
    except Exception as e:
        print(f"Error saving model: {e}")

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

if __name__ == "__main__":
    main() 