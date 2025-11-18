"""
CadQuery Dataset Preparation Script
Extracts CadQuery code from Jupyter notebook and organizes for training
"""

import json
import os
import re
from pathlib import Path

def extract_cadquery_examples(notebook_path, output_dir="../dataset/raw_code"):
    """Extract CadQuery code cells from notebook"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    examples = []
    
    # Extract code cells
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # Skip if empty or just imports
            if len(source.strip()) < 20:
                continue
            
            # Check if it contains CadQuery code
            if 'cadquery' in source.lower() or 'cq.' in source or 'Workplane' in source:
                examples.append({
                    'id': i,
                    'code': source,
                    'cell_index': i
                })
                
                # Save individual file
                output_path = Path(output_dir) / f"example_{i:03d}.py"
                with open(output_path, 'w') as f:
                    f.write(source)
                
                print(f"✓ Extracted example {i:03d} ({len(source)} chars)")
    
    # Save metadata
    metadata = {
        'total_examples': len(examples),
        'source_notebook': notebook_path,
        'examples': [{'id': ex['id'], 'length': len(ex['code'])} for ex in examples]
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Extracted {len(examples)} CadQuery examples to {output_dir}/")
    return examples

def analyze_parameters(code):
    """Extract parameter definitions from CadQuery code"""
    
    params = []
    
    # Look for variable assignments with numeric values
    # Pattern: variable_name = numeric_value  # optional comment
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)(?:\s*#\s*(.*))?'
    
    for match in re.finditer(pattern, code):
        var_name = match.group(1)
        value = float(match.group(2))
        comment = match.group(3) if match.group(3) else ""
        
        # Skip common non-parameter variables
        if var_name in ['i', 'j', 'k', 'x', 'y', 'z', 'idx']:
            continue
        
        params.append({
            'name': var_name,
            'value': value,
            'description': comment.strip()
        })
    
    return params

def create_augmented_versions(code, params, num_variations=5):
    """Create augmented versions by varying parameters"""
    
    if not params:
        return []
    
    variations = []
    
    for i in range(num_variations):
        new_code = code
        variation_info = {}
        
        for param in params:
            # Vary by ±30%
            import random
            factor = random.uniform(0.7, 1.3)
            new_value = param['value'] * factor
            
            # Replace in code
            pattern = f"{param['name']}\\s*=\\s*{param['value']}"
            replacement = f"{param['name']} = {new_value:.2f}"
            new_code = re.sub(pattern, replacement, new_code)
            
            variation_info[param['name']] = new_value
        
        variations.append({
            'code': new_code,
            'variations': variation_info
        })
    
    return variations

if __name__ == "__main__":
    # Extract examples
    notebook_path = "CQ_examples.ipynb"
    examples = extract_cadquery_examples(notebook_path)
    
    # Analyze parameters in first few examples
    print("\n📊 Parameter Analysis (first 5 examples):")
    for i, example in enumerate(examples[:5]):
        params = analyze_parameters(example['code'])
        if params:
            print(f"\nExample {example['id']}:")
            for p in params:
                print(f"  - {p['name']} = {p['value']} ({p['description']})")
    
    print("\n✅ Extraction complete!")
    print("\nNext steps:")
    print("1. Run each .py file to generate 3D models")
    print("2. Export models to STL/STEP format")
    print("3. Generate point clouds from meshes")
    print("4. Create sketch-style renderings")
