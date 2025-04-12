# BV-ViT: Vision Transformer with Post-Training Quantization

This repository implements a Vision Transformer (ViT) model with support for post-training quantization to reduce model size and improve inference speed while maintaining accuracy.

## Overview

BV-ViT provides tools for:
- Training a Vision Transformer model on image classification tasks
- Post-training quantization at various bit depths (4-16 bits)
- Detailed analysis of quantization effects on model performance
- Comprehensive visualization of quantization results

## Model Architecture

The Vision Transformer (ViT) implementation follows the architecture described in the ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) paper with customizable parameters:

- Patchify images into fixed-size patches
- Linear projection of flattened patches
- Add positional embeddings
- Prepend a learnable classification token
- Process through Transformer encoder blocks
- Classification via MLP head on the classification token

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/username/BV-ViT.git
cd BV-ViT
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a Vision Transformer model:

```bash
python run.py --data-path /path/to/dataset \
              --output-dir ./outputs \
              --epochs 90 \
              --batch-size 128 \
              --img-size 224 \
              --patch-size 16 \
              --embed-dim 768 \
              --depth 12 \
              --num-heads 12 \
              --num-classes 1000
```

Key parameters:
- `--data-path`: Path to the dataset (supports ImageNet format)
- `--img-size`: Input image size
- `--patch-size`: Size of image patches
- `--embed-dim`: Embedding dimension
- `--depth`: Number of transformer blocks
- `--num-heads`: Number of attention heads

### Post-Training Quantization

After training, you can quantize the model to reduce its size and improve inference speed:

```bash
python post_training.py --model-path ./outputs/vit_model.pth \
                       --data-path /path/to/dataset \
                       --output-dir ./outputs/quantization
```

The post-training script automatically:
1. Quantizes the model to multiple bit depths (4, 6, 8, 10, 12, 16 bits)
2. Evaluates performance metrics for each bit depth
3. Generates visualizations comparing original vs. quantized models
4. Creates detailed reports on size, speed, and accuracy tradeoffs

## Key Components

### run.py

The main training script with the following features:
- Command-line interface for model configuration
- Training and validation loops
- Model checkpointing and backup
- Comprehensive metrics logging and visualization
- Support for resuming training from checkpoints

### post_training.py

Handles post-training quantization with:
- Support for multiple bit-width quantization (4-16 bits)
- Weight and activation quantization
- Comprehensive performance evaluation
- Detailed visualizations and comparisons
- Analysis of quantization effects on different layers

### model.py

Implements the Vision Transformer architecture:
- PatchEmbed: Converts images to patch embeddings
- MLP: Multi-layer perceptron for transformer blocks
- TransformerEncoderBlock: Self-attention and feed-forward layers
- VisionTransformer: Complete model implementation

## Visualization Outputs

The post-training quantization process generates:
- Accuracy comparisons between original and quantized models
- Inference time measurements
- Model size reductions
- Per-class accuracy changes
- Activation distribution histograms
- Confusion matrices
- Comparative analysis across different bit depths

## Performance Insights

Quantizing the model typically results in:
- 2-4x reduction in model size (8-bit quantization)
- 1.2-1.5x speedup in inference time
- Minimal accuracy degradation at 8+ bits
- Noticeable accuracy drops at 4-6 bits

## License

[MIT License](LICENSE)

## Acknowledgments

- The Vision Transformer implementation is based on the paper "An Image is Worth 16x16 Words" by Dosovitskiy et al.
- Quantization techniques draw inspiration from PyTorch's Quantization API and various research papers on efficient deep learning. 