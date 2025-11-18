"""
Prepare training data for SketchSampler.

This script processes CadQuery code files to create:
- Sketch images (from point cloud renderings)
- Point clouds (from 3D models)
- Density maps (for training guidance)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle as pkl

# Add sketchsampler to path
SKETCHSAMPLER_PATH = Path(__file__).parent.parent.parent / "sketchsampler"
import sys
if str(SKETCHSAMPLER_PATH) not in sys.path:
    sys.path.insert(0, str(SKETCHSAMPLER_PATH))


def load_pointcloud(pcd_path: Path) -> np.ndarray:
    """Load point cloud from file."""
    if pcd_path.suffix == '.npy':
        return np.load(pcd_path)
    elif pcd_path.suffix == '.pcd':
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported point cloud format: {pcd_path.suffix}")


def compute_density_map(
    pointcloud: np.ndarray,
    image_size: Tuple[int, int] = (256, 256),
    camera_matrix: Optional[np.ndarray] = None,
    camera_position: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute density map from point cloud for a given view.
    
    This creates a 2D density map that guides point sampling.
    """
    H, W = image_size
    
    # If no camera info, use default orthographic projection
    if camera_matrix is None:
        # Simple orthographic projection: project to XY plane
        # Normalize point cloud
        pc_centered = pointcloud - pointcloud.mean(axis=0)
        scale = np.abs(pc_centered).max()
        if scale > 0:
            pc_normalized = pc_centered / scale
        else:
            pc_normalized = pc_centered
        
        # Project to 2D (use X, Y coordinates)
        x_coords = pc_normalized[:, 0]
        y_coords = pc_normalized[:, 1]
        
        # Map to image coordinates [0, 1]
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-8)
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-8)
        
        # Convert to pixel coordinates
        x_pix = (x_norm * (W - 1)).astype(int)
        y_pix = (y_norm * (H - 1)).astype(int)
        
        # Create density map
        density_map = np.zeros((H, W), dtype=np.float32)
        for x, y in zip(x_pix, y_pix):
            if 0 <= x < W and 0 <= y < H:
                density_map[y, x] += 1.0
        
        # Normalize
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum()
        
        return density_map
    
    else:
        # Use provided camera matrix for perspective projection
        # Project 3D points to 2D
        points_homogeneous = np.hstack([pointcloud, np.ones((len(pointcloud), 1))])
        projected = (camera_matrix @ points_homogeneous.T).T
        
        # Normalize by depth
        z = projected[:, 2]
        x_2d = projected[:, 0] / (z + 1e-8)
        y_2d = projected[:, 1] / (z + 1e-8)
        
        # Map to image coordinates
        x_norm = (x_2d - x_2d.min()) / (x_2d.max() - x_2d.min() + 1e-8)
        y_norm = (y_2d - y_2d.min()) / (y_2d.max() - y_2d.min() + 1e-8)
        
        x_pix = (x_norm * (W - 1)).astype(int)
        y_pix = (y_norm * (H - 1)).astype(int)
        
        density_map = np.zeros((H, W), dtype=np.float32)
        for x, y in zip(x_pix, y_pix):
            if 0 <= x < W and 0 <= y < H:
                density_map[y, x] += 1.0
        
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum()
        
        return density_map


def render_sketch_from_pointcloud(
    pointcloud: np.ndarray,
    output_path: Path,
    image_size: Tuple[int, int] = (256, 256),
    view_angle: Tuple[float, float] = (0, 0)  # (azimuth, elevation)
) -> np.ndarray:
    """
    Render a sketch-like image from point cloud.
    
    Returns the rendered image as numpy array.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    H, W = image_size
    azim, elev = view_angle
    
    # Normalize point cloud
    pc_centered = pointcloud - pointcloud.mean(axis=0)
    scale = np.abs(pc_centered).max()
    if scale > 0:
        pc_normalized = pc_centered / scale
    else:
        pc_normalized = pc_centered
    
    # Create figure
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(
        pc_normalized[:, 0],
        pc_normalized[:, 1],
        pc_normalized[:, 2],
        c='black',
        s=0.5,
        alpha=0.8
    )
    
    # Set view
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    
    # Set equal aspect
    max_range = np.array([
        pc_normalized[:, 0].max() - pc_normalized[:, 0].min(),
        pc_normalized[:, 1].max() - pc_normalized[:, 1].min(),
        pc_normalized[:, 2].max() - pc_normalized[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (pc_normalized[:, 0].max() + pc_normalized[:, 0].min()) * 0.5
    mid_y = (pc_normalized[:, 1].max() + pc_normalized[:, 1].min()) * 0.5
    mid_z = (pc_normalized[:, 2].max() + pc_normalized[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save
    plt.savefig(
        output_path,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        edgecolor='none',
        dpi=100
    )
    plt.close()
    
    # Load and return as array
    from PIL import Image
    img = Image.open(output_path).convert('L')
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    return np.array(img)


def prepare_single_example(
    metadata_path: Path,
    output_dir: Path,
    view_angles: List[Tuple[float, float]] = None
) -> Dict:
    """
    Prepare training data for a single CadQuery example.
    
    Args:
        metadata_path: Path to metadata.json from processed data
        output_dir: Output directory for SketchSampler training data
        view_angles: List of (azimuth, elevation) tuples for different views
    
    Returns:
        Dictionary with paths to generated files
    """
    if view_angles is None:
        # Default views
        view_angles = [
            (0, 0),      # Front
            (90, 0),     # Side
            (0, 90),     # Top
            (45, 30),    # Isometric 1
        ]
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    pointcloud_path = Path(metadata['pointcloud_numpy'])
    
    # Load point cloud
    pointcloud = load_pointcloud(pointcloud_path)
    
    # Create output structure
    # SketchSampler expects: sketches/, pointclouds/, density_maps/, cameras/
    sketches_dir = output_dir / "sketches"
    pointclouds_dir = output_dir / "pointclouds"
    density_maps_dir = output_dir / "density_maps"
    cameras_dir = output_dir / "cameras"
    
    for d in [sketches_dir, pointclouds_dir, density_maps_dir, cameras_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    examples = []
    
    # Generate data for each view
    for view_idx, (azim, elev) in enumerate(view_angles):
        view_name = f"{model_name}_view{view_idx}"
        
        # Render sketch
        sketch_path = sketches_dir / f"{view_name}.png"
        sketch_img = render_sketch_from_pointcloud(
            pointcloud,
            sketch_path,
            view_angle=(azim, elev)
        )
        
        # Compute density map
        density_map = compute_density_map(pointcloud, image_size=(256, 256))
        density_path = density_maps_dir / f"{view_name}.dat"
        with open(density_path, 'wb') as f:
            pkl.dump(density_map, f)
        
        # Save point cloud (normalized for this view)
        # Transform point cloud based on view
        pc_centered = pointcloud - pointcloud.mean(axis=0)
        scale = np.abs(pc_centered).max()
        if scale > 0:
            pc_normalized = pc_centered / scale
        else:
            pc_normalized = pc_centered
        
        # Rotate based on view angle
        # Simple rotation (can be improved)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('zyx', [azim, elev, 0], degrees=True)
        pc_rotated = rot.apply(pc_normalized)
        
        # Save point cloud
        pc_path = pointclouds_dir / f"{view_name}/pt.dat"
        pc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pc_path, 'wb') as f:
            pkl.dump(pc_rotated, f)
        
        # Save camera info (simplified)
        camera_matrix = np.eye(3)  # Identity for orthographic
        camera_position = np.zeros(3)
        camera_path = cameras_dir / f"{view_name}.dat"
        with open(camera_path, 'wb') as f:
            pkl.dump((camera_matrix, camera_position), f)
        
        examples.append({
            'view_name': view_name,
            'sketch': str(sketch_path),
            'pointcloud': str(pc_path),
            'density_map': str(density_path),
            'camera': str(camera_path)
        })
    
    return {
        'model_name': model_name,
        'examples': examples
    }


def create_train_test_split(
    all_examples: List[Dict],
    train_ratio: float = 0.8,
    output_dir: Path = None
):
    """Create train/test split files."""
    import random
    random.seed(42)
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    # Create split directory
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train list
    train_list_path = split_dir / "train_list.txt"
    with open(train_list_path, 'w') as f:
        for ex in train_examples:
            for view_ex in ex['examples']:
                f.write(f"{view_ex['view_name']}.dat\n")
    
    # Write test list
    test_list_path = split_dir / "test_list.txt"
    with open(test_list_path, 'w') as f:
        for ex in test_examples:
            for view_ex in ex['examples']:
                f.write(f"{view_ex['view_name']}.dat\n")
    
    print(f"Created train/test split:")
    print(f"  Train: {len(train_examples)} models ({sum(len(e['examples']) for e in train_examples)} views)")
    print(f"  Test: {len(test_examples)} models ({sum(len(e['examples']) for e in test_examples)} views)")
    
    return train_list_path, test_list_path


def main():
    """Main function to prepare all training data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare training data for SketchSampler"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="../dataset/processed",
        help="Directory containing processed CadQuery data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../dataset/sketchsampler_data",
        help="Output directory for SketchSampler training data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training"
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all metadata files
    metadata_files = sorted(processed_dir.glob("*/metadata.json"))
    
    if not metadata_files:
        print(f"No metadata files found in {processed_dir}")
        return
    
    print(f"Found {len(metadata_files)} examples to process")
    
    all_examples = []
    for i, metadata_path in enumerate(metadata_files):
        print(f"Processing {i+1}/{len(metadata_files)}: {metadata_path.parent.name}")
        try:
            example = prepare_single_example(metadata_path, output_dir)
            all_examples.append(example)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create train/test split
    create_train_test_split(all_examples, args.train_ratio, output_dir)
    
    print(f"\n✓ Training data prepared in {output_dir}")
    print(f"  Next: Configure SketchSampler .env file with these paths")


if __name__ == "__main__":
    main()

