# SketchHopper Pipeline: Training and Inference Plan

## Overview

This pipeline converts hand-drawn sketches to parametric CadQuery Python code through three stages:
1. **Sketch Cleaning**: `img2vector` library cleans and vectorizes input sketches
2. **Point Cloud Generation**: `SketchSampler` model generates 3D point clouds from cleaned sketches
3. **Code Generation**: LoRA fine-tuned `3D-LLaVA` model generates CadQuery code from point clouds

## Pipeline Architecture

```
Input Sketch (PNG/JPG)
    ↓
[img2vector] → Cleaned SVG/Image
    ↓
[SketchSampler] → Point Cloud (N×3)
    ↓
[3D-LLaVA] → CadQuery Python Code
```

## Directory Structure

```
sketchhopper/
├── src/                          # Main pipeline code
│   ├── pipeline/                 # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── sketch_cleaner.py    # img2vector wrapper
│   │   ├── pointcloud_generator.py  # SketchSampler wrapper
│   │   ├── code_generator.py    # 3D-LLaVA wrapper
│   │   └── inference.py         # End-to-end inference
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── prepare_sketchsampler_data.py
│   │   ├── prepare_3dllava_data.py
│   │   └── train_pipeline.py
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── data_loader.py
│       └── pointcloud_utils.py
├── sketchsampler/                # SketchSampler submodule
├── 3d-llava/                     # 3D-LLaVA submodule
└── dataset/                      # Training data
    ├── raw_code/                 # Original CadQuery code
    └── processed/                # Processed training data
```

## Phase 1: Data Preparation

### 1.1 Prepare Training Data for SketchSampler

**Goal**: Create sketch → point cloud pairs for training SketchSampler

**Steps**:
1. Use existing CadQuery code files in `dataset/raw_code/`
2. Execute each code file to generate 3D models
3. Convert 3D models to point clouds (already done in `generate_training_data_fixed.py`)
4. Generate multiple sketch views from each point cloud (7 views per model)
5. Create training pairs: (sketch image, point cloud, density map)

**Data Format**:
- Sketches: PNG images (256×256 or 512×512)
- Point clouds: NumPy arrays (N×3) or PCD files
- Density maps: NumPy arrays (H×W) for sampling guidance

**Script**: `src/training/prepare_sketchsampler_data.py`

### 1.2 Prepare Training Data for 3D-LLaVA

**Goal**: Create point cloud → CadQuery code pairs for fine-tuning

**Steps**:
1. Use point clouds generated from CadQuery models
2. Pair each point cloud with its corresponding CadQuery code
3. Format data in 3D-LLaVA's expected JSON format:
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
         "value": "<code>\nimport cadquery as cq\n..."
       }
     ],
     "pointcloud": "path/to/pointcloud.npy"
   }
   ```

**Script**: `src/training/prepare_3dllava_data.py`

## Phase 2: Component Training

### 2.1 Train SketchSampler

**Location**: Training happens in `sketchsampler/` directory

**Prerequisites**:
- Point cloud data from Phase 1.1
- Sketch images (cleaned with img2vector)
- Density maps (can be computed from point clouds)

**Training Steps**:
1. Set up environment variables (`.env` file):
   ```bash
   export PROJECT_ROOT="/path/to/sketchsampler"
   export SKETCH_PATH="/path/to/dataset/processed/sketches"
   export SHAPENET_PT_PATH="/path/to/dataset/processed/pointclouds"
   export SHAPENET_DENSITY_PATH="/path/to/dataset/processed/density_maps"
   export TRAIN_LIST="/path/to/dataset/processed/splits/train_list.txt"
   export TEST_LIST="/path/to/dataset/processed/splits/test_list.txt"
   ```

2. Configure Hydra configs in `sketchsampler/conf/`:
   - Adjust batch size, learning rate, epochs
   - Set number of points (default: 4096)

3. Run training:
   ```bash
   cd sketchsampler
   python src/trainval.py
   ```

4. Checkpoint will be saved in `sketchsampler/outputs/`

**Note**: If you have limited data (50 examples), consider:
- Data augmentation (rotation, scaling, noise)
- Transfer learning from pre-trained SketchSampler weights
- Few-shot learning techniques

### 2.2 Fine-tune 3D-LLaVA with LoRA

**Location**: Training happens in `3d-llava/` directory

**Prerequisites**:
- Pre-trained 3D-LLaVA checkpoint
- Point cloud → code pairs from Phase 1.2
- Data formatted in 3D-LLaVA JSON format

**Training Steps**:
1. Prepare training data JSON file:
   - Place in `3d-llava/playground/data/train_info/cadquery_train_3d_llava.json`
   - Format matches existing 3D-LLaVA training data structure

2. Prepare point cloud data:
   - Place point clouds in `3d-llava/playground/data/scannet/super_points/train/`
   - Or modify data loading to use your custom path

3. Create training script (modify `scripts/train/finetune-3d-llava-lora.sh`):
   ```bash
   export cadquery_data=./playground/data/train_info/cadquery_train_3d_llava.json
   
   EXP_NAME=finetune-3d-llava-cadquery
   
   PYTHONPATH=$(pwd) \
   deepspeed llava/train/train_mem.py \
       --lora_enable True --lora_r 32 --lora_alpha 64 \
       --deepspeed ./scripts/zero1_3d_llava.json \
       --model_name_or_path liuhaotian/llava-v1.5-7b \
       --version v1 \
       --data_path $cadquery_data \
       --scan_folder ./playground/data/scannet \
       --mm_projector_type mlp2x_gelu \
       --pointcloud_tower ./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth \
       --pc_modules_to_finetune alignment_proj hidden_seg_fc \
       --num_pc_tokens 100 \
       --freeze_pointcloud_tower True \
       --bf16 True \
       --output_dir ./checkpoints/${EXP_NAME} \
       --num_train_epochs 3 \
       --per_device_train_batch_size 2 \
       --gradient_accumulation_steps 8 \
       --learning_rate 2e-4 \
       --model_max_length 4096 \
       --gradient_checkpointing True
   ```

4. Run training:
   ```bash
   cd 3d-llava
   bash scripts/train/finetune-3d-llava-cadquery.sh
   ```

**Note**: With only 50 examples, you may need:
- Strong data augmentation
- Lower learning rate (1e-4 or 5e-5)
- More epochs (5-10)
- Consider using the pre-trained model for few-shot inference instead

## Phase 3: Pipeline Integration

### 3.1 Create Pipeline Components

**Files to create in `src/pipeline/`**:

1. **`sketch_cleaner.py`**: Wrapper for img2vector
   - Input: Raw sketch image (PNG/JPG)
   - Output: Cleaned image (PNG) ready for SketchSampler
   - Uses img2vector with appropriate preprocessing

2. **`pointcloud_generator.py`**: Wrapper for SketchSampler
   - Input: Cleaned sketch image
   - Output: Point cloud (N×3 numpy array)
   - Loads trained SketchSampler checkpoint
   - Handles model inference

3. **`code_generator.py`**: Wrapper for 3D-LLaVA
   - Input: Point cloud (N×3)
   - Output: CadQuery Python code string
   - Loads fine-tuned 3D-LLaVA checkpoint
   - Formats point cloud for 3D-LLaVA input
   - Handles text generation

4. **`inference.py`**: End-to-end pipeline
   - Orchestrates all three components
   - Handles errors and intermediate outputs
   - Provides clean API

### 3.2 Point Cloud Format Conversion

**Challenge**: SketchSampler outputs point clouds in one format, 3D-LLaVA expects another.

**Solution**: Create conversion utilities in `src/utils/pointcloud_utils.py`:
- Convert SketchSampler output to 3D-LLaVA format
- Handle coordinate systems, normalization
- Add required metadata (superpoints, features, etc.)

## Phase 4: Inference

### 4.1 Single Image Inference

**Usage**:
```python
from src.pipeline.inference import SketchHopperPipeline

# Initialize pipeline
pipeline = SketchHopperPipeline(
    sketchsampler_checkpoint="path/to/sketchsampler/checkpoint.ckpt",
    llava_checkpoint="path/to/3d-llava/checkpoint",
    device="cuda"
)

# Run inference
sketch_path = "path/to/sketch.png"
cadquery_code = pipeline.generate(sketch_path)

print(cadquery_code)
```

### 4.2 Batch Inference

```python
# Process multiple sketches
sketch_paths = ["sketch1.png", "sketch2.png", ...]
results = pipeline.generate_batch(sketch_paths)
```

### 4.3 CLI Interface

```bash
python src/pipeline/inference.py \
    --sketch path/to/sketch.png \
    --output output.py \
    --sketchsampler-checkpoint path/to/checkpoint.ckpt \
    --llava-checkpoint path/to/checkpoint
```

## Training Workflow Summary

### Step-by-Step Training Process

1. **Data Preparation** (Run once):
   ```bash
   # Generate point clouds and sketches from CadQuery code
   python dataset_utils/generate_training_data_fixed.py
   
   # Prepare SketchSampler training data
   python src/training/prepare_sketchsampler_data.py
   
   # Prepare 3D-LLaVA training data
   python src/training/prepare_3dllava_data.py
   ```

2. **Train SketchSampler**:
   ```bash
   cd sketchsampler
   # Configure .env file with data paths
   python src/trainval.py
   ```

3. **Fine-tune 3D-LLaVA**:
   ```bash
   cd 3d-llava
   # Prepare data in expected format
   bash scripts/train/finetune-3d-llava-cadquery.sh
   ```

4. **Test Pipeline**:
   ```bash
   python src/pipeline/inference.py --sketch test_sketch.png
   ```

## Key Considerations

### Data Limitations (50 examples)

**Strategies**:
1. **Data Augmentation**:
   - Rotate, scale, add noise to sketches
   - Vary point cloud sampling
   - Paraphrase code descriptions

2. **Transfer Learning**:
   - Use pre-trained SketchSampler weights
   - Use pre-trained 3D-LLaVA as base
   - Fine-tune only on your data

3. **Few-Shot Learning**:
   - Use 3D-LLaVA in few-shot mode
   - Provide examples in prompt
   - Consider retrieval-augmented generation

4. **Synthetic Data**:
   - Generate more CadQuery examples programmatically
   - Create variations of existing models
   - Use code templates

### Point Cloud Format Compatibility

**Issue**: SketchSampler and 3D-LLaVA may use different point cloud formats.

**Solution**:
- SketchSampler outputs: `(N, 3)` numpy array
- 3D-LLaVA expects: Voxelized format with features, superpoints, etc.
- Create conversion layer in `pointcloud_utils.py`

### Memory and Compute Requirements

- **SketchSampler**: Moderate (can run on single GPU)
- **3D-LLaVA**: High (requires multiple GPUs, DeepSpeed)
- **Inference**: Can optimize by loading models on-demand

## Next Steps

1. ✅ Create this plan document
2. ⬜ Implement data preparation scripts
3. ⬜ Create pipeline components
4. ⬜ Set up training configurations
5. ⬜ Test with small subset
6. ⬜ Full training run
7. ⬜ Inference pipeline testing
8. ⬜ Optimization and refinement

