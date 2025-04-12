import torch
import argparse
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model import VisionTransformer
from tqdm import tqdm

def load_model(model_path, model_config=None):
    """Load a model from a file."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # If no model config is provided, try to infer it from the state dict
        if model_config is None:
            model_config = infer_model_config(state_dict)
            print(f"Inferred model configuration: {model_config}")
        
        # Create the model and load state dict
        model = VisionTransformer(**model_config)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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

def preprocess_image(image_path, img_size=224):
    """Preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor, class_names=None, device='cpu'):
    """Run prediction on an image tensor."""
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    results = {
        'prediction': predicted.item(),
        'confidence': probabilities[0, predicted].item(),
        'top5_indices': top5_indices[0].tolist(),
        'top5_probabilities': top5_prob[0].tolist()
    }
    
    # Add class names if available
    if class_names is not None:
        results['predicted_class'] = class_names[predicted.item()]
        results['top5_classes'] = [class_names[idx] for idx in top5_indices[0].tolist()]
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with a Vision Transformer model")
    parser.add_argument('--model', type=str, required=True, help="Path to the model file")
    parser.add_argument('--image', type=str, help="Path to an image for inference")
    parser.add_argument('--img_size', type=int, default=224, help="Image size")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to use for inference")
    parser.add_argument('--class_names', type=str, help="JSON file with class names")
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Load class names if provided
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        try:
            import json
            with open(args.class_names, 'r') as f:
                class_names = json.load(f)
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    # Run inference on a single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        
        print(f"Running inference on: {args.image}")
        image_tensor = preprocess_image(args.image, args.img_size)
        results = predict(model, image_tensor, class_names, device)
        
        print("\nInference Results:")
        if 'predicted_class' in results:
            print(f"Predicted class: {results['predicted_class']} (index: {results['prediction']})")
        else:
            print(f"Predicted class index: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.4f}")
        
        print("\nTop 5 Predictions:")
        for i in range(5):
            if 'top5_classes' in results:
                print(f"{i+1}. {results['top5_classes'][i]} (index: {results['top5_indices'][i]}) - {results['top5_probabilities'][i]:.4f}")
            else:
                print(f"{i+1}. Class index: {results['top5_indices'][i]} - {results['top5_probabilities'][i]:.4f}")

if __name__ == "__main__":
    main() 