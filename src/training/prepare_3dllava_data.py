"""
Prepare training data for 3D-LLaVA fine-tuning.

This script creates point cloud → CadQuery code pairs in the format
expected by 3D-LLaVA training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None


def load_cadquery_code(code_path: Path) -> str:
    """Load and clean CadQuery code."""
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Remove jupyter-specific code if present
    import re
    code = re.sub(r'from jupyter_cadquery import.*?\n', '', code)
    code = re.sub(r'import jupyter_cadquery.*?\n', '', code)
    code = re.sub(r'show\([^)]*\)', '', code)
    code = re.sub(r'display\([^)]*\)', '', code)
    
    return code.strip()


def create_3dllava_example(
    example_id: str,
    pointcloud_path: Path,
    code_path: Path,
    prompt: Optional[str] = None
) -> Dict:
    """
    Create a single 3D-LLaVA training example.
    
    Args:
        example_id: Unique identifier
        pointcloud_path: Path to point cloud file
        code_path: Path to CadQuery code file
        prompt: Optional custom prompt
    
    Returns:
        Dictionary in 3D-LLaVA format
    """
    if prompt is None:
        prompt = (
            "Generate complete, executable CadQuery Python code for this 3D model. "
            "The code should be well-structured and include all necessary imports. "
            "Return only the Python code, no explanations."
        )
    
    code = load_cadquery_code(code_path)
    
    # Format as conversation
    example = {
        "id": example_id,
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": f"```python\n{code}\n```"
            }
        ],
        "pointcloud": str(pointcloud_path)
    }
    
    return example


def prepare_3dllava_dataset(
    processed_dir: Path,
    output_path: Path,
    train_ratio: float = 0.8,
    prompt_template: Optional[str] = None
) -> Dict:
    """
    Prepare complete 3D-LLaVA training dataset.
    
    Args:
        processed_dir: Directory containing processed CadQuery data
        output_path: Path to save JSON dataset file
        train_ratio: Ratio for train/val split
        prompt_template: Optional custom prompt template
    
    Returns:
        Dictionary with train/val split info
    """
    # Find all metadata files
    metadata_files = sorted(processed_dir.glob("*/metadata.json"))
    
    if not metadata_files:
        raise ValueError(f"No metadata files found in {processed_dir}")
    
    print(f"Found {len(metadata_files)} examples")
    
    all_examples = []
    
    for metadata_path in metadata_files:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata['model_name']
        pointcloud_path = Path(metadata['pointcloud_numpy'])
        code_path = Path(metadata['code_path'])
        
        # Verify files exist
        if not pointcloud_path.exists():
            print(f"Warning: Point cloud not found: {pointcloud_path}")
            continue
        
        if not code_path.exists():
            print(f"Warning: Code file not found: {code_path}")
            continue
        
        # Create example
        example = create_3dllava_example(
            example_id=model_name,
            pointcloud_path=pointcloud_path,
            code_path=code_path,
            prompt=prompt_template
        )
        
        all_examples.append(example)
    
    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save full dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Save train split
    train_path = output_path.parent / f"{output_path.stem}_train.json"
    with open(train_path, 'w') as f:
        json.dump(train_examples, f, indent=2)
    
    # Save val split
    val_path = output_path.parent / f"{output_path.stem}_val.json"
    with open(val_path, 'w') as f:
        json.dump(val_examples, f, indent=2)
    
    print(f"\n✓ Dataset prepared:")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Saved to: {output_path}")
    print(f"  Train split: {train_path}")
    print(f"  Val split: {val_path}")
    
    return {
        'total': len(all_examples),
        'train': len(train_examples),
        'val': len(val_examples),
        'output_path': str(output_path),
        'train_path': str(train_path),
        'val_path': str(val_path)
    }


def copy_pointclouds_to_3dllava_format(
    processed_dir: Path,
    output_dir: Path,
    format: str = "numpy"
):
    """
    Copy point clouds to 3D-LLaVA expected format.
    
    3D-LLaVA expects point clouds in a specific directory structure.
    This function organizes them appropriately.
    
    Args:
        processed_dir: Directory with processed data
        output_dir: Output directory for 3D-LLaVA
        format: "numpy" or "pcd"
    """
    output_dir = Path(output_dir)
    pointclouds_dir = output_dir / "pointclouds"
    pointclouds_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_files = sorted(processed_dir.glob("*/metadata.json"))
    
    for metadata_path in metadata_files:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata['model_name']
        source_path = Path(metadata['pointcloud_numpy'])
        
        if not source_path.exists():
            continue
        
        # Copy to output directory
        if format == "numpy":
            dest_path = pointclouds_dir / f"{model_name}.npy"
            import shutil
            shutil.copy2(source_path, dest_path)
        else:
            # Convert to PCD if needed
            try:
                import open3d as o3d
                points = np.load(source_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                dest_path = pointclouds_dir / f"{model_name}.pcd"
                o3d.io.write_point_cloud(str(dest_path), pcd)
            except ImportError:
                raise ImportError("open3d is required for PCD format conversion")
        
        print(f"Copied: {model_name}")
    
    print(f"\n✓ Point clouds copied to {pointclouds_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare training data for 3D-LLaVA"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="../dataset/processed",
        help="Directory containing processed CadQuery data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../dataset/3dllava_data/cadquery_train_3d_llava.json",
        help="Output path for JSON dataset file"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training"
    )
    parser.add_argument(
        "--copy-pointclouds",
        action="store_true",
        help="Copy point clouds to 3D-LLaVA format directory"
    )
    parser.add_argument(
        "--pointcloud-output-dir",
        type=str,
        default="../dataset/3dllava_data/pointclouds",
        help="Output directory for point clouds (if --copy-pointclouds)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Custom prompt template"
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output_path)
    
    # Prepare dataset
    prepare_3dllava_dataset(
        processed_dir=processed_dir,
        output_path=output_path,
        train_ratio=args.train_ratio,
        prompt_template=args.prompt_template
    )
    
    # Copy point clouds if requested
    if args.copy_pointclouds:
        copy_pointclouds_to_3dllava_format(
            processed_dir=processed_dir,
            output_dir=args.pointcloud_output_dir
        )
    
    print("\n✓ 3D-LLaVA training data preparation complete!")
    print("\nNext steps:")
    print("1. Place the JSON file in 3d-llava/playground/data/train_info/")
    print("2. Place point clouds in 3d-llava/playground/data/scannet/super_points/train/")
    print("3. Modify training script to use your data")


if __name__ == "__main__":
    main()

