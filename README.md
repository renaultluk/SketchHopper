# SketchHopper Pipeline

This repository contains the SketchHopper pipeline, which converts hand-drawn sketches to parametric CadQuery Python code.

## Overview

The SketchHopper pipeline consists of three main stages:

1. **Sketch Cleaning** (`src/pipeline/sketch_cleaner.py`): Uses `img2vector` to clean and preprocess sketches
2. **Point Cloud Generation** (`src/pipeline/pointcloud_generator.py`): Uses `SketchSampler` to generate 3D point clouds
3. **Code Generation** (`src/pipeline/code_generator.py`): Uses fine-tuned `3D-LLaVA` to generate CadQuery code

## Quick Start

### Installation

```bash
# Install img2vector
pip install img2vector

# Install other dependencies (from project root)
pip install -r sketchsampler/requirements.txt
pip install -r 3d-llava/requirements.txt  # if available
```

### Inference (After Training)

```python
from src.pipeline import SketchHopperPipeline

# Initialize pipeline
pipeline = SketchHopperPipeline(
    sketchsampler_checkpoint="path/to/sketchsampler/checkpoint.ckpt",
    llava_checkpoint="path/to/3d-llava/checkpoint",
    device="cuda"
)

# Generate code from sketch
code = pipeline.generate("path/to/sketch.png", output_path="output.py")
print(code)
```

### CLI Usage

```bash
python src/pipeline/inference.py \
    --sketch path/to/sketch.png \
    --output output.py \
    --sketchsampler-checkpoint path/to/checkpoint.ckpt \
    --llava-checkpoint path/to/checkpoint
```

## Training Workflow

See `src/PIPELINE_PLAN.md` for detailed training instructions. Quick summary:

### 1. Prepare Training Data

```bash
# Generate point clouds and sketches from CadQuery code
python dataset_utils/generate_training_data_fixed.py

# Prepare SketchSampler training data
python src/training/prepare_sketchsampler_data.py \
    --processed-dir dataset/processed \
    --output-dir dataset/sketchsampler_data

# Prepare 3D-LLaVA training data
python src/training/prepare_3dllava_data.py \
    --processed-dir dataset/processed \
    --output-path dataset/3dllava_data/cadquery_train_3d_llava.json
```

### 2. Train SketchSampler

```bash
cd sketchsampler
# Configure .env file with data paths
python src/trainval.py
```

### 3. Fine-tune 3D-LLaVA

```bash
cd 3d-llava
# Modify training script for your data
bash scripts/train/finetune-3d-llava-cadquery.sh
```

## Repository Structure

```
.
├── src/
│   ├── pipeline/                  # Main pipeline components
│   ├── training/                  # Training utilities
│   ├── utils/                     # Shared utilities
│   └── PIPELINE_PLAN.md           # Detailed training plan
├── sketchsampler/                 # SketchSampler submodule
├── 3d-llava/                      # 3D-LLaVA submodule
├── dataset/                       # Data assets
└── README.md                      # This file
```

## Components

### SketchCleaner

Cleans and preprocesses hand-drawn sketches using `img2vector`.

```python
from src.pipeline.sketch_cleaner import SketchCleaner

cleaner = SketchCleaner(
    preprocessing_level="medium",
    colormode="binary"
)
cleaned = cleaner.clean("sketch.png", "cleaned.png")
```

### PointCloudGenerator

Generates 3D point clouds from sketches using SketchSampler.

```python
from src.pipeline.pointcloud_generator import PointCloudGenerator

generator = PointCloudGenerator(
    checkpoint_path="path/to/checkpoint.ckpt",
    device="cuda"
)
pointcloud = generator.generate("cleaned.png")  # (N, 3) numpy array
```

### CodeGenerator

Generates CadQuery code from point clouds using 3D-LLaVA.

```python
from src.pipeline.code_generator import CodeGenerator

generator = CodeGenerator(
    checkpoint_path="path/to/checkpoint",
    base_model="liuhaotian/llava-v1.5-7b",
    device="cuda"
)
code = generator.generate(pointcloud)
```

## Data Format

### SketchSampler Training Data

- **Sketches**: PNG images (256×256 or 512×512)
- **Point clouds**: NumPy arrays (N×3) or PCD files
- **Density maps**: NumPy arrays (H×W) for sampling guidance

### 3D-LLaVA Training Data

JSON format:
```json
{
  "id": "example_000",
  "conversations": [
    {
      "from": "human",
      "value": "Generate CadQuery code for this 3D model."
    },
    {
      "from": "gpt",
      "value": "```python\nimport cadquery as cq\n...\n```"
    }
  ],
  "pointcloud": "path/to/pointcloud.npy"
}
```

## Notes

- With only 50 training examples, consider:
  - Data augmentation
  - Transfer learning from pre-trained models
  - Few-shot learning techniques
  - Synthetic data generation

- Point cloud format conversion is handled automatically by `pointcloud_utils.py`

- Memory requirements:
  - SketchSampler: Moderate (single GPU)
  - 3D-LLaVA: High (multiple GPUs, DeepSpeed)

## Troubleshooting

### Import Errors

If you get import errors for `sketchsampler` or `3d-llava`:
- Make sure submodules are initialized: `git submodule update --init --recursive`
- Check that paths are correctly set in the scripts

### Model Loading Errors

- Verify checkpoint paths are correct
- Check that model architectures match expected formats
- Ensure all dependencies are installed

### Point Cloud Format Issues

- Use `pointcloud_utils.py` for format conversion
- Check coordinate systems and normalization
- Verify point cloud dimensions match model expectations

## References

- [SketchSampler](https://github.com/cjeen/sketchsampler)
- [3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA)
- [img2vector](https://pypi.org/project/img2vector/)


