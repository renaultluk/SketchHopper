"""
Point Cloud Generation Module

Uses SketchSampler model to generate 3D point clouds from cleaned sketches.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional
import sys
import os

# Add sketchsampler to path
SKETCHSAMPLER_PATH = Path(__file__).parent.parent.parent / "sketchsampler"
if str(SKETCHSAMPLER_PATH) not in sys.path:
    sys.path.insert(0, str(SKETCHSAMPLER_PATH))


class PointCloudGenerator:
    """
    Generates 3D point clouds from sketches using SketchSampler model.
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        num_points: int = 4096
    ):
        """
        Initialize the point cloud generator.
        
        Args:
            checkpoint_path: Path to trained SketchSampler checkpoint
            device: "cuda" or "cpu"
            num_points: Number of points to generate
        """
        self.device = device
        self.num_points = num_points
        self.model = None
        self.checkpoint_path = Path(checkpoint_path)
        
        self._load_model()
    
    def _load_model(self):
        """Load the SketchSampler model from checkpoint."""
        import pytorch_lightning as pl
        from omegaconf import DictConfig, OmegaConf
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Please train SketchSampler first or provide a valid checkpoint path."
            )
        
        # Try to load config from checkpoint or use default
        try:
            # Load checkpoint to get config
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
            
            # Get model config
            if 'hyper_parameters' in checkpoint:
                cfg = checkpoint['hyper_parameters'].get('cfg')
                if cfg is None:
                    # Use default config
                    config_path = SKETCHSAMPLER_PATH / "conf" / "sketchsampler.yaml"
                    cfg = OmegaConf.load(config_path)
            else:
                config_path = SKETCHSAMPLER_PATH / "conf" / "sketchsampler.yaml"
                cfg = OmegaConf.load(config_path)
            
            # Update num_points
            if 'train' not in cfg:
                cfg['train'] = {}
            cfg['train']['n_points'] = self.num_points
            
            # Import model
            from src.model.sketchsampler import SketchSampler
            
            # Instantiate model
            self.model = SketchSampler.load_from_checkpoint(
                str(self.checkpoint_path),
                cfg=cfg,
                strict=False
            )
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SketchSampler model: {e}\n"
                "Make sure the checkpoint is valid and dependencies are installed."
            )
    
    def generate(
        self,
        sketch: Union[str, Path, np.ndarray],
        return_density_map: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Generate point cloud from sketch.
        
        Args:
            sketch: Path to sketch image or numpy array (H, W) or (H, W, 3)
            return_density_map: If True, also return density map
        
        Returns:
            Point cloud as numpy array (N, 3) or tuple (pointcloud, density_map)
        """
        # Load and preprocess sketch
        if isinstance(sketch, (str, Path)):
            from PIL import Image
            img = Image.open(sketch)
            sketch_array = np.array(img)
        else:
            sketch_array = sketch
        
        # Convert to format expected by SketchSampler
        # SketchSampler expects: (1, 1, H, W) normalized to [-1, 1]
        if len(sketch_array.shape) == 3:
            # Convert RGB to grayscale
            sketch_array = np.mean(sketch_array, axis=2)
        
        # Normalize to [0, 1] then to [-1, 1]
        sketch_array = sketch_array.astype(np.float32) / 255.0
        sketch_array = (sketch_array - 0.5) * 2.0
        
        # Add batch and channel dimensions: (1, 1, H, W)
        sketch_tensor = torch.from_numpy(sketch_array).unsqueeze(0).unsqueeze(0)
        sketch_tensor = sketch_tensor.to(self.device)
        
        # Generate point cloud
        with torch.no_grad():
            # Forward pass
            # SketchSampler.forward expects: (sketch, density_map, use_predicted_map)
            predicted_map, predicted_points = self.model(
                sketch_tensor,
                density_map=None,
                use_predicted_map=True
            )
            
            # Convert to numpy
            pointcloud = predicted_points[0].cpu().numpy()  # (N, 3)
            density_map = predicted_map[0, 0].cpu().numpy() if return_density_map else None
        
        if return_density_map:
            return pointcloud, density_map
        return pointcloud
    
    def generate_batch(
        self,
        sketches: list,
        return_density_maps: bool = False
    ) -> Union[list, tuple]:
        """
        Generate point clouds for multiple sketches.
        
        Args:
            sketches: List of sketch paths or arrays
            return_density_maps: If True, also return density maps
        
        Returns:
            List of point clouds or tuple (pointclouds, density_maps)
        """
        results = []
        density_maps = []
        
        for sketch in sketches:
            if return_density_maps:
                pc, dm = self.generate(sketch, return_density_map=True)
                results.append(pc)
                density_maps.append(dm)
            else:
                results.append(self.generate(sketch))
        
        if return_density_maps:
            return results, density_maps
        return results

