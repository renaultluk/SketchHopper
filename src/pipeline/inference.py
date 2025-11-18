"""
End-to-End Inference Pipeline

Orchestrates sketch cleaning, point cloud generation, and code generation.
"""

from pathlib import Path
from typing import Union, Optional, List
import numpy as np

from .sketch_cleaner import SketchCleaner
from .pointcloud_generator import PointCloudGenerator
from .code_generator import CodeGenerator


class SketchToCadQueryPipeline:
    """
    Complete pipeline from sketch to CadQuery code.
    """
    
    def __init__(
        self,
        sketchsampler_checkpoint: Union[str, Path],
        llava_checkpoint: Union[str, Path],
        llava_base_model: str = "liuhaotian/llava-v1.5-7b",
        device: str = "cuda",
        sketch_cleaner_kwargs: Optional[dict] = None,
        pointcloud_generator_kwargs: Optional[dict] = None,
        code_generator_kwargs: Optional[dict] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            sketchsampler_checkpoint: Path to SketchSampler checkpoint
            llava_checkpoint: Path to fine-tuned 3D-LLaVA checkpoint
            llava_base_model: Base LLaVA model name
            device: "cuda" or "cpu"
            sketch_cleaner_kwargs: Optional kwargs for SketchCleaner
            pointcloud_generator_kwargs: Optional kwargs for PointCloudGenerator
            code_generator_kwargs: Optional kwargs for CodeGenerator
        """
        self.device = device
        
        # Initialize components
        sketch_cleaner_kwargs = sketch_cleaner_kwargs or {}
        self.sketch_cleaner = SketchCleaner(**sketch_cleaner_kwargs)
        
        pointcloud_generator_kwargs = pointcloud_generator_kwargs or {}
        self.pointcloud_generator = PointCloudGenerator(
            checkpoint_path=sketchsampler_checkpoint,
            device=device,
            **pointcloud_generator_kwargs
        )
        
        code_generator_kwargs = code_generator_kwargs or {}
        self.code_generator = CodeGenerator(
            checkpoint_path=llava_checkpoint,
            base_model=llava_base_model,
            device=device,
            **code_generator_kwargs
        )
    
    def generate(
        self,
        sketch_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_intermediates: bool = False,
        intermediate_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate CadQuery code from a sketch.
        
        Args:
            sketch_path: Path to input sketch image
            output_path: Optional path to save generated code
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory to save intermediate results
        
        Returns:
            Generated CadQuery code as string
        """
        sketch_path = Path(sketch_path)
        
        if not sketch_path.exists():
            raise FileNotFoundError(f"Sketch not found: {sketch_path}")
        
        # Step 1: Clean sketch
        print("Step 1/3: Cleaning sketch...")
        if save_intermediates and intermediate_dir:
            intermediate_dir = Path(intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cleaned_path = intermediate_dir / f"{sketch_path.stem}_cleaned.png"
        else:
            cleaned_path = None
        
        cleaned_sketch = self.sketch_cleaner.clean(sketch_path, cleaned_path)
        print("  ✓ Sketch cleaned")
        
        # Step 2: Generate point cloud
        print("Step 2/3: Generating point cloud...")
        pointcloud = self.pointcloud_generator.generate(cleaned_sketch)
        print(f"  ✓ Generated point cloud with {len(pointcloud)} points")
        
        if save_intermediates and intermediate_dir:
            import numpy as np
            np.save(intermediate_dir / f"{sketch_path.stem}_pointcloud.npy", pointcloud)
        
        # Step 3: Generate code
        print("Step 3/3: Generating CadQuery code...")
        code = self.code_generator.generate(pointcloud)
        print("  ✓ Code generated")
        
        # Save code if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(code)
            print(f"  ✓ Code saved to {output_path}")
        
        return code
    
    def generate_batch(
        self,
        sketch_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Generate code for multiple sketches.
        
        Args:
            sketch_paths: List of sketch image paths
            output_dir: Optional directory to save generated code files
        
        Returns:
            List of generated code strings
        """
        results = []
        
        for i, sketch_path in enumerate(sketch_paths):
            print(f"\nProcessing sketch {i+1}/{len(sketch_paths)}: {sketch_path}")
            
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{Path(sketch_path).stem}.py"
            
            try:
                code = self.generate(sketch_path, output_path)
                results.append(code)
            except Exception as e:
                print(f"  ✗ Error processing {sketch_path}: {e}")
                results.append(None)
        
        return results


def main():
    """CLI interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate CadQuery code from hand-drawn sketches"
    )
    parser.add_argument(
        "--sketch",
        type=str,
        required=True,
        help="Path to input sketch image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save generated code (default: print to stdout)"
    )
    parser.add_argument(
        "--sketchsampler-checkpoint",
        type=str,
        required=True,
        help="Path to SketchSampler checkpoint"
    )
    parser.add_argument(
        "--llava-checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned 3D-LLaVA checkpoint"
    )
    parser.add_argument(
        "--llava-base-model",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        help="Base LLaVA model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate results (cleaned sketch, point cloud)"
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default=None,
        help="Directory to save intermediate results"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SketchToCadQueryPipeline(
        sketchsampler_checkpoint=args.sketchsampler_checkpoint,
        llava_checkpoint=args.llava_checkpoint,
        llava_base_model=args.llava_base_model,
        device=args.device
    )
    
    # Generate code
    code = pipeline.generate(
        sketch_path=args.sketch,
        output_path=args.output,
        save_intermediates=args.save_intermediates,
        intermediate_dir=args.intermediate_dir
    )
    
    # Print if no output path
    if not args.output:
        print("\n" + "="*60)
        print("Generated CadQuery Code:")
        print("="*60)
        print(code)


if __name__ == "__main__":
    main()

