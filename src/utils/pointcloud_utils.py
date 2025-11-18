"""
Point Cloud Utilities

Conversion and formatting utilities for point clouds between
SketchSampler and 3D-LLaVA formats.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud to unit sphere.
    
    Args:
        points: Point cloud (N, 3)
    
    Returns:
        Normalized point cloud (N, 3)
    """
    # Center
    centroid = points.mean(axis=0)
    points = points - centroid
    
    # Scale to unit sphere
    max_dist = np.abs(points).max()
    if max_dist > 0:
        points = points / max_dist
    
    return points


def convert_to_3dllava_format(
    pointcloud: np.ndarray,
    num_points: Optional[int] = None,
    add_features: bool = True
) -> Dict:
    """
    Convert point cloud from SketchSampler format to 3D-LLaVA format.
    
    3D-LLaVA expects:
    - coord: (N, 3) point coordinates
    - feat: (N, F) point features (optional, can be zeros)
    - offset: batch offsets for batching
    - grid_coord: voxel coordinates
    - superpoint_mask: superpoint segmentation
    
    Args:
        pointcloud: Point cloud from SketchSampler (N, 3)
        num_points: Target number of points (downsample/upsample if needed)
        add_features: Whether to add feature vectors
    
    Returns:
        Dictionary with 3D-LLaVA format data
    """
    # Normalize point cloud
    pointcloud = normalize_pointcloud(pointcloud)
    
    # Resample if needed
    if num_points is not None and len(pointcloud) != num_points:
        pointcloud = resample_pointcloud(pointcloud, num_points)
    
    N = len(pointcloud)
    
    # Convert to torch tensors
    coord = torch.from_numpy(pointcloud).float()  # (N, 3)
    
    # Create features (can be zeros or use point coordinates)
    if add_features:
        # Use normalized coordinates as features
        feat = coord.clone()  # (N, 3)
    else:
        feat = torch.zeros(N, 3)  # (N, 3)
    
    # Create offset (for single point cloud, offset is [0, N])
    offset = torch.tensor([0, N], dtype=torch.long)
    
    # Create grid coordinates (voxelize)
    # 3D-LLaVA uses voxelization, we'll create a simple grid
    voxel_size = 0.05  # Adjust based on your point cloud scale
    grid_coord = (coord / voxel_size).long()
    
    # Create superpoint mask (simple: all points in one superpoint)
    # In practice, you might want to use a superpoint segmentation algorithm
    superpoint_mask = torch.zeros(N, dtype=torch.long)
    
    return {
        'coord': coord,
        'feat': feat,
        'offset': offset,
        'grid_coord': grid_coord,
        'superpoint_mask': superpoint_mask,
        'spatial_shape': grid_coord.max(dim=0)[0] + 1,
        'condition': None  # Can add condition if needed
    }


def resample_pointcloud(
    points: np.ndarray,
    num_points: int,
    method: str = "fps"
) -> np.ndarray:
    """
    Resample point cloud to target number of points.
    
    Args:
        points: Input point cloud (N, 3)
        num_points: Target number of points
        method: "fps" (farthest point sampling) or "random"
    
    Returns:
        Resampled point cloud (num_points, 3)
    """
    if len(points) == num_points:
        return points
    
    if len(points) < num_points:
        # Upsample by repeating points
        indices = np.random.choice(len(points), num_points, replace=True)
        return points[indices]
    
    # Downsample
    if method == "fps":
        return farthest_point_sampling(points, num_points)
    else:
        # Random sampling
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for point cloud downsampling.
    
    Args:
        points: Point cloud (N, 3)
        num_samples: Number of samples to select
    
    Returns:
        Sampled points (num_samples, 3)
    """
    N = len(points)
    if N <= num_samples:
        return points
    
    # Initialize
    sampled_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.ones(N) * np.inf
    
    # Start with random point
    current = np.random.randint(0, N)
    
    for i in range(num_samples):
        sampled_indices[i] = current
        point = points[current]
        
        # Update distances
        dists = np.sum((points - point) ** 2, axis=1)
        distances = np.minimum(distances, dists)
        
        # Select farthest point
        current = np.argmax(distances)
    
    return points[sampled_indices]


def prepare_batch_for_3dllava(
    pointclouds: List[np.ndarray],
    num_points: Optional[int] = None
) -> Dict:
    """
    Prepare a batch of point clouds for 3D-LLaVA.
    
    Args:
        pointclouds: List of point clouds
        num_points: Target number of points per cloud
    
    Returns:
        Batched dictionary for 3D-LLaVA
    """
    # Convert each point cloud
    converted = [convert_to_3dllava_format(pc, num_points) for pc in pointclouds]
    
    # Batch coordinates and features
    coords = [c['coord'] for c in converted]
    feats = [c['feat'] for c in converted]
    offsets = [c['offset'] for c in converted]
    grid_coords = [c['grid_coord'] for c in converted]
    superpoint_masks = [c['superpoint_mask'] for c in converted]
    
    # Concatenate with offsets
    batch_coord = torch.cat(coords, dim=0)
    batch_feat = torch.cat(feats, dim=0)
    batch_grid_coord = torch.cat(grid_coords, dim=0)
    batch_superpoint_mask = torch.cat(superpoint_masks, dim=0)
    
    # Update offsets
    batch_offset = torch.zeros(len(pointclouds) + 1, dtype=torch.long)
    for i, offset in enumerate(offsets):
        batch_offset[i + 1] = batch_offset[i] + offset[1]
    
    # Spatial shape (max across batch)
    spatial_shape = torch.stack([c['spatial_shape'] for c in converted]).max(dim=0)[0]
    
    return {
        'coord': batch_coord,
        'feat': batch_feat,
        'offset': batch_offset,
        'grid_coord': batch_grid_coord,
        'superpoint_mask': batch_superpoint_mask,
        'spatial_shape': spatial_shape,
        'conditions': [c['condition'] for c in converted]
    }

