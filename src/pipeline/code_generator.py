"""
Code Generation Module

Uses fine-tuned 3D-LLaVA model to generate CadQuery code from point clouds.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict
import sys

# Add 3d-llava to path
LLAVA_PATH = Path(__file__).parent.parent.parent / "3d-llava"
if str(LLAVA_PATH) not in sys.path:
    sys.path.insert(0, str(LLAVA_PATH))

from src.utils.pointcloud_utils import convert_to_3dllava_format, prepare_batch_for_3dllava


class CodeGenerator:
    """
    Generates CadQuery code from point clouds using fine-tuned 3D-LLaVA.
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        base_model: str = "liuhaotian/llava-v1.5-7b",
        device: str = "cuda",
        max_new_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """
        Initialize the code generator.
        
        Args:
            checkpoint_path: Path to fine-tuned 3D-LLaVA checkpoint
            base_model: Base LLaVA model name
            device: "cuda" or "cpu"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the 3D-LLaVA model and tokenizer."""
        try:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.conversation import conv_templates, SeparatorStyle
            
            disable_torch_init()
            
            # Load model
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=str(self.checkpoint_path) if self.checkpoint_path.is_dir() else self.base_model,
                model_base=self.base_model if self.checkpoint_path.is_dir() else None,
                model_name="llava",
                device_map=self.device
            )
            
            # If checkpoint is a file (LoRA weights), load them
            if self.checkpoint_path.is_file():
                # Load LoRA weights
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(self.checkpoint_path))
            
            self.model = model
            self.tokenizer = tokenizer
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load 3D-LLaVA model: {e}\n"
                "Make sure the checkpoint is valid and dependencies are installed."
            )
    
    def generate(
        self,
        pointcloud: Union[np.ndarray, Dict],
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate CadQuery code from point cloud.
        
        Args:
            pointcloud: Point cloud (N, 3) or 3D-LLaVA format dict
            prompt: Optional custom prompt (default: asks for CadQuery code)
        
        Returns:
            Generated CadQuery code as string
        """
        # Convert point cloud to 3D-LLaVA format if needed
        if isinstance(pointcloud, np.ndarray):
            pc_data = convert_to_3dllava_format(pointcloud)
        else:
            pc_data = pointcloud
        
        # Prepare prompt
        if prompt is None:
            prompt = (
                "Generate complete, executable CadQuery Python code for this 3D model. "
                "The code should be well-structured and include all necessary imports. "
                "Return only the Python code, no explanations."
            )
        
        # Format conversation
        from llava.conversation import conv_templates
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = self.tokenizer(
            prompt_text,
            return_tensors='pt'
        ).input_ids.to(self.device)
        
        # Prepare point cloud data
        coord = pc_data['coord'].unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        feat = pc_data['feat'].unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        offset = pc_data['offset'].unsqueeze(0).to(self.device)
        grid_coord = pc_data['grid_coord'].unsqueeze(0).to(self.device)
        superpoint_mask = [pc_data['superpoint_mask'].to(self.device)]
        spatial_shape = pc_data['spatial_shape'].unsqueeze(0).to(self.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                coord=coord,
                grid_coord=grid_coord,
                offset=offset,
                feat=feat,
                p2v_map=None,  # Will be computed internally
                v2p_map=None,  # Will be computed internally
                spatial_shape=spatial_shape,
                superpoint_mask=superpoint_mask,
                conditions=[pc_data.get('condition')],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=0.9,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                tokenizer=self.tokenizer,
                use_cache=True
            )
        
        # Decode
        output_text = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]
        
        # Extract code from response
        code = self._extract_code(output_text)
        
        return code
    
    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from model response.
        
        Args:
            text: Model output text
        
        Returns:
            Extracted code
        """
        # Try to find code blocks
        import re
        
        # Look for ```python ... ```
        code_block = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Look for ``` ... ```
        code_block = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Look for code after "import" or "from"
        if 'import' in text or 'from' in text:
            # Try to extract from first import to end
            lines = text.split('\n')
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    start_idx = i
                    break
            
            if start_idx is not None:
                return '\n'.join(lines[start_idx:]).strip()
        
        # Return as-is if no pattern matches
        return text.strip()
    
    def generate_batch(
        self,
        pointclouds: list,
        prompts: Optional[list] = None
    ) -> list:
        """
        Generate code for multiple point clouds.
        
        Args:
            pointclouds: List of point clouds
            prompts: Optional list of custom prompts
        
        Returns:
            List of generated code strings
        """
        if prompts is None:
            prompts = [None] * len(pointclouds)
        
        results = []
        for pc, prompt in zip(pointclouds, prompts):
            results.append(self.generate(pc, prompt))
        
        return results

