"""
Sketch Cleaning Module

Uses img2vector library to clean and preprocess hand-drawn sketches
before feeding them to SketchSampler.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import tempfile
import os


class SketchCleaner:
    """
    Cleans and preprocesses hand-drawn sketches using img2vector.
    
    The cleaned sketches are optimized for SketchSampler input:
    - Binary (black/white) format
    - Appropriate resolution
    - Noise reduction
    """
    
    def __init__(
        self,
        preprocessing_level: str = "medium",
        colormode: str = "binary",
        output_size: tuple = (512, 512)
    ):
        """
        Initialize the sketch cleaner.
        
        Args:
            preprocessing_level: "none", "light", "medium", or "heavy"
            colormode: "color" or "binary" (binary recommended for sketches)
            output_size: (width, height) for output image
        """
        try:
            from img2vector import Img2Vector
            self.converter = Img2Vector()
        except ImportError:
            raise ImportError(
                "img2vector is not installed. Install it with: pip install img2vector"
            )
        
        self.preprocessing_level = preprocessing_level
        self.colormode = colormode
        self.output_size = output_size
    
    def clean(
        self,
        input_path: Union[str, Path, Image.Image],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Clean a sketch image.
        
        Args:
            input_path: Path to input image or PIL Image object
            output_path: Optional path to save cleaned image
        
        Returns:
            Cleaned image as numpy array (H, W) for binary or (H, W, 3) for color
        """
        # Handle PIL Image input
        if isinstance(input_path, Image.Image):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                input_path.save(tmp.name)
                input_path = tmp.name
                temp_file = tmp.name
        else:
            temp_file = None
        
        try:
            # Convert to SVG first (vectorization helps clean the image)
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_tmp:
                svg_path = svg_tmp.name
            
            # Use img2vector to convert and clean
            self.converter.convert(
                str(input_path),
                output_path=str(svg_path),
                auto_optimize=True,
                preprocessing_level=self.preprocessing_level,
                colormode=self.colormode,
                mode="spline" if self.colormode == "binary" else "polygon"
            )
            
            # Convert SVG back to image for SketchSampler
            # For now, we'll use the original image with preprocessing
            # In practice, you might want to render the SVG to image
            from PIL import Image
            img = Image.open(input_path)
            
            # Apply preprocessing directly
            img = self._preprocess_image(img)
            
            # Resize to target size
            img = img.resize(self.output_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            if self.colormode == "binary":
                # Convert to grayscale and threshold
                img = img.convert('L')
                img_array = np.array(img)
                # Threshold to binary
                threshold = 128
                img_array = (img_array > threshold).astype(np.float32) * 255
            else:
                img_array = np.array(img.convert('RGB'))
            
            # Save if output path provided
            if output_path:
                output_img = Image.fromarray(img_array.astype(np.uint8))
                output_img.save(output_path)
            
            return img_array
            
        finally:
            # Clean up temp files
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(svg_path):
                os.unlink(svg_path)
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Apply preprocessing to image based on preprocessing_level.
        
        Args:
            img: PIL Image
        
        Returns:
            Preprocessed PIL Image
        """
        if self.preprocessing_level == "none":
            return img
        
        import cv2
        img_array = np.array(img)
        
        if self.preprocessing_level == "light":
            # Basic noise reduction
            img_array = cv2.bilateralFilter(img_array, 5, 50, 50)
            # Slight contrast enhancement
            img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)
        
        elif self.preprocessing_level == "medium":
            # More aggressive denoising
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            # Edge enhancement
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
            # Contrast enhancement
            img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=15)
        
        elif self.preprocessing_level == "heavy":
            # Convert to grayscale for heavy processing
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            img_array = binary
        
        return Image.fromarray(img_array)

