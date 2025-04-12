# BV-ViT Model Conversion

This repository contains tools to work with Vision Transformer (ViT) model checkpoints.

## Scripts

### `convert_checkpoint.py`

This script converts training checkpoints (which contain optimizer state, epoch information, etc.) into standalone model files that can be easily used for inference.

#### Usage:

```
python convert_checkpoint.py --checkpoint /path/to/checkpoint_epoch_xx.pth --output /path/to/save/model.pth
```

Arguments:
- `--checkpoint`: Path to the checkpoint file to convert
- `--output`: Path where the converted model will be saved

### `inference.py`

This script loads a converted model and runs inference on an image.

#### Usage:

```
python inference.py --model /path/to/model.pth --image /path/to/image.jpg [--img_size 224] [--device cuda] [--class_names /path/to/class_names.json]
```

Arguments:
- `--model`: Path to the model file
- `--image`: Path to an image for inference
- `--img_size` (optional): Image size for inference (default: 224)
- `--device` (optional): Device to use for inference ('cpu' or 'cuda', default: 'cpu')
- `--class_names` (optional): JSON file with class names

## Examples

### Convert a checkpoint to a model:

```
python convert_checkpoint.py --checkpoint checkpoints/checkpoint_epoch_90.pth --output models/vit_model.pth
```

### Run inference with the converted model:

```
python inference.py --model models/vit_model.pth --image samples/cat.jpg --device cuda
```

### Run inference with class names:

```
python inference.py --model models/vit_model.pth --image samples/cat.jpg --class_names imagenet_classes.json
``` 