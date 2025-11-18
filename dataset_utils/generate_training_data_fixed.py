"""
Fixed: Generate 3D Models and Point Clouds from CadQuery Code
This version handles Jupyter notebook-specific code and cleans it up before execution
"""

import os
import re
import sys
import numpy as np
from pathlib import Path
import json

def clean_jupyter_code(code_str):
    """
    Remove Jupyter-specific code and fix common issues
    """
    # Remove jupyter_cadquery imports and show() calls
    code_str = re.sub(r'from jupyter_cadquery import.*?\n', '', code_str)
    code_str = re.sub(r'import jupyter_cadquery.*?\n', '', code_str)
    code_str = re.sub(r'show\([^)]*\)', '', code_str)
    
    # Remove display calls
    code_str = re.sub(r'display\([^)]*\)', '', code_str)
    
    # Remove IPython display calls
    code_str = re.sub(r'IPython\.display\..*?\n', '', code_str)
    
    # Add common missing imports at the top
    imports_to_add = []
    
    # Check if math is used but not imported
    if re.search(r'\bmath\.(pi|sin|cos|tan|sqrt)', code_str) and 'import math' not in code_str:
        imports_to_add.append('import math')
    
    # Check if numpy is used but not imported
    if re.search(r'\bnp\.', code_str) and 'import numpy' not in code_str:
        imports_to_add.append('import numpy as np')
    
    # Add missing imports at the beginning
    if imports_to_add:
        code_str = '\n'.join(imports_to_add) + '\n' + code_str
    
    # Ensure cadquery is imported
    if 'import cadquery' not in code_str:
        code_str = 'import cadquery as cq\n' + code_str
    
    return code_str

def execute_cadquery_code(code_str, output_name="model"):
    """
    Execute CadQuery code and return the result object
    """
    try:
        # Clean the code first
        code_str = clean_jupyter_code(code_str)
        
        # Create a namespace for execution
        namespace = {}
        
        # Execute the code
        exec(code_str, namespace)
        
        # Try to find the result object
        # Common variable names: result, c, model, part, p, obj
        for var_name in ['result', 'c', 'model', 'part', 'p', 'obj', 'assembly', 'body']:
            if var_name in namespace:
                obj = namespace[var_name]
                # Check if it's a CadQuery object
                if hasattr(obj, 'val') or hasattr(obj, 'objects'):
                    return obj
        
        # If no explicit result variable, try to get any CadQuery object
        import cadquery as cq
        for key, value in namespace.items():
            if isinstance(value, (cq.Workplane, cq.Assembly)):
                return value
        
        print(f"⚠️  Could not find CadQuery object in: {output_name}")
        print(f"   Available variables: {[k for k in namespace.keys() if not k.startswith('_')]}")
        return None
        
    except Exception as e:
        print(f"❌ Error executing {output_name}: {str(e)}")
        import traceback
        print(f"   {traceback.format_exc().split('File')[0]}")
        return None

def export_to_stl(cq_object, output_path):
    """Export CadQuery object to STL file"""
    try:
        from cadquery import exporters
        exporters.export(cq_object, str(output_path))
        return True
    except Exception as e:
        print(f"❌ Error exporting to STL: {str(e)}")
        return False

def mesh_to_pointcloud(stl_path, num_points=4096):
    """Convert STL mesh to point cloud"""
    try:
        import open3d as o3d
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        
        if not mesh.has_triangles():
            print(f"⚠️  Mesh has no triangles: {stl_path}")
            return None
        
        # Compute normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Sample point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        return pcd
        
    except Exception as e:
        print(f"❌ Error creating point cloud: {str(e)}")
        return None

def render_sketch_views(pcd, output_dir, model_name):
    """
    Render multiple sketch-like views of the point cloud
    Returns list of generated image paths
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        points = np.asarray(pcd.points)
        
        # Define viewpoints (azimuth, elevation)
        viewpoints = [
            (0, 0, "front"),      # Front view
            (90, 0, "side"),      # Side view
            (0, 90, "top"),       # Top view
            (45, 30, "iso1"),     # Isometric 1
            (135, 30, "iso2"),    # Isometric 2
            (-45, 30, "iso3"),    # Isometric 3
            (225, 30, "iso4"),    # Isometric 4
        ]
        
        image_paths = []
        
        for azim, elev, view_name in viewpoints:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points as lines (sketch style)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='black', marker='.', s=1, linewidths=0.5)
            
            # Set viewpoint
            ax.view_init(elev=elev, azim=azim)
            
            # Remove axes for sketch-like appearance
            ax.set_axis_off()
            
            # Set equal aspect ratio
            max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                 points[:, 1].max()-points[:, 1].min(),
                                 points[:, 2].max()-points[:, 2].min()]).max() / 2.0
            
            mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save
            output_path = Path(output_dir) / f"{model_name}_view_{view_name}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            image_paths.append(str(output_path))
        
        return image_paths
        
    except Exception as e:
        print(f"❌ Error rendering views: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_single_model(code_path, output_base_dir):
    """Process a single CadQuery code file"""
    
    model_name = Path(code_path).stem
    model_dir = Path(output_base_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Read original code
    with open(code_path, 'r') as f:
        original_code = f.read()
    
    # Clean code
    cleaned_code = clean_jupyter_code(original_code)
    
    # Save cleaned code
    code_output_path = model_dir / "code.py"
    with open(code_output_path, 'w') as f:
        f.write(cleaned_code)
    
    # Execute code
    print("  1. Executing CadQuery code...")
    cq_obj = execute_cadquery_code(original_code, model_name)
    if cq_obj is None:
        return None
    
    # Export to STL
    print("  2. Exporting to STL...")
    stl_path = model_dir / f"{model_name}.stl"
    if not export_to_stl(cq_obj, stl_path):
        return None
    
    # Generate point cloud
    print("  3. Generating point cloud...")
    pcd = mesh_to_pointcloud(stl_path, num_points=4096)
    if pcd is None:
        return None
    
    # Save point cloud
    import open3d as o3d
    pcd_path = model_dir / f"{model_name}.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    
    # Also save as numpy array
    points = np.asarray(pcd.points)
    np_path = model_dir / f"{model_name}_pointcloud.npy"
    np.save(np_path, points)
    
    # Render views
    print("  4. Rendering sketch views...")
    image_paths = render_sketch_views(pcd, model_dir, model_name)
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'code_path': str(code_path),
        'stl_path': str(stl_path),
        'pointcloud_path': str(pcd_path),
        'pointcloud_numpy': str(np_path),
        'num_points': len(points),
        'sketch_views': image_paths,
        'num_views': len(image_paths)
    }
    
    # Save metadata
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Complete! Generated {len(image_paths)} views")
    return metadata

def process_all_models(input_dir="../dataset/raw_code", output_dir="../dataset/processed"):
    """Process all CadQuery code files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Python files
    code_files = sorted(Path(input_dir).glob("*.py"))
    
    print(f"Found {len(code_files)} CadQuery code files")
    print("="*60)
    
    results = []
    successful = 0
    failed = 0
    errors = {}
    
    for code_file in code_files:
        try:
            metadata = process_single_model(code_file, output_dir)
            if metadata:
                results.append(metadata)
                successful += 1
            else:
                failed += 1
                errors[code_file.name] = "Could not find CadQuery object"
        except Exception as e:
            print(f"❌ Failed to process {code_file.name}: {str(e)}")
            failed += 1
            errors[code_file.name] = str(e)
    
    # Save overall summary
    summary = {
        'total_models': len(code_files),
        'successful': successful,
        'failed': failed,
        'success_rate': f"{successful/len(code_files)*100:.1f}%",
        'models': results,
        'errors': errors
    }
    
    with open(Path(output_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {successful}/{len(code_files)} ({successful/len(code_files)*100:.1f}%)")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total views generated: {successful * 7}")
    
    if successful > 0:
        print("\n" + "="*60)
        print("✅ SUCCESS! You now have training data ready!")
        print("="*60)
        print(f"📦 {successful} 3D models")
        print(f"☁️  {successful} point clouds (4,096 points each)")
        print(f"🖼️  {successful * 7} sketch views (7 per model)")
        print(f"\n💡 Next: Run data augmentation to expand to 500+ examples")

if __name__ == "__main__":
    process_all_models()
