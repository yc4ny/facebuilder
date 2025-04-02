"""
Copyright (c) SECERN AI, Inc. and its affiliates.
File input/output module.

This module handles loading images, saving 3D models, and exporting mesh data
to various file formats.
"""
import pickle
import os
import cv2

def load_images_from_directory(dir_path="images"):
    """
    Load all images from a directory.
    
    Loads and optionally resizes all image files from the specified directory.
    If the directory doesn't exist or contains no valid images, a default image
    is loaded instead.
    
    Args:
        dir_path: Path to the directory containing images (default: "images")
        
    Returns:
        list: List of loaded images as numpy arrays in BGR format
    """
    images = []
    
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return [cv2.imread("images/side.jpg")]  # Return default image
    
    # Process each file in the directory
    for filename in sorted(os.listdir(dir_path)):
        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Get original dimensions
                img_h, img_w = img.shape[:2]
                
                # Resize large images to a more manageable size
                max_dimension = 1600  # Maximum dimension for display
                if img_h > max_dimension or img_w > max_dimension:
                    # Calculate scale factor to maintain aspect ratio
                    scale = max_dimension / max(img_h, img_w)
                    new_width = int(img_w * scale)
                    new_height = int(img_h * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"Resized image {filename} from {img_w}x{img_h} to {new_width}x{new_height}")
                
                images.append(img)
                print(f"Loaded image: {filename} - Size: {img.shape[1]}x{img.shape[0]}")
    
    # If no valid images found, load default image
    if not images:
        print("No valid images found in directory")
        return [cv2.imread("images/side.jpg")]
    
    return images

def save_model(verts2d, verts3d, faces, pins_per_image, camera_matrices, rotations, translations):
    """
    Save the current 3D model to file.
    
    Saves the complete model state including vertices, faces, pins, and camera
    parameters. Also creates an OBJ file for import into 3D modeling software.
    
    Args:
        verts2d: 2D vertex positions as numpy array
        verts3d: 3D vertex positions as numpy array
        faces: Face indices as numpy array
        pins_per_image: List of pins for each image
        camera_matrices: Camera intrinsic matrices for each view
        rotations: Rotation vectors for each view
        translations: Translation vectors for each view
    """
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Compile model data
    model_data = {
        "verts2d": verts2d,
        "verts3d": verts3d,  # Contains the modified 3D vertices
        "faces": faces,
        "pins": pins_per_image,
        "camera_matrices": camera_matrices,
        "rotations": rotations,
        "translations": translations
    }
    
    # Save as pickle file (binary format)
    with open("output/face_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model saved to output/face_model.pkl")
    
    # Save as OBJ file for 3D software import
    save_to_obj(verts3d, faces, "output/face_model.obj")
    
    # Create default model reference file
    try:
        with open("output/face_model_default.obj", "w") as f:
            f.write("# OBJ file of the original unmodified model\n")
            
        print("Also saved original model to output/face_model_default.obj for comparison")
    except Exception as e:
        print(f"Error saving default model: {e}")

def save_to_obj(vertices, faces, filename="output/face_model.obj"):
    """
    Save the 3D model as an OBJ file.
    
    Converts the model data to the standard OBJ file format, which can be
    imported by most 3D modeling software. Also attempts to create preview
    renders using trimesh if available.
    
    Args:
        vertices: Array of 3D vertex positions
        faces: Array of face indices (0-indexed)
        filename: Output filename (default: "output/face_model.obj")
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write OBJ file
    with open(filename, 'w') as f:
        # Write header
        f.write("# OBJ file created by Face Builder\n")
        
        # Write vertices (v x y z)
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (f v1 v2 v3) - convert from 0-indexed to 1-indexed
        for face in faces:
            # Convert from 0-indexed to 1-indexed (OBJ standard)
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # Try to create preview visualizations using trimesh
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save as PLY with vertex colors for better visualization
        colored_mesh = mesh.copy()
        colored_mesh.visual.vertex_colors = trimesh.visual.random_colors(len(vertices))
        colored_mesh.export(filename.replace('.obj', '.ply'))
        
        # Save a PNG render for quick preview
        scene = trimesh.Scene(colored_mesh)
        png = scene.save_image(resolution=[1024, 768])
        with open(filename.replace('.obj', '.png'), 'wb') as f:
            f.write(png)
            
        print(f"Also saved visualization to {filename.replace('.obj', '.png')}")
    except ImportError:
        print("Trimesh not installed. Install with: pip install trimesh")
    except Exception as e:
        print(f"Error creating visualization: {e}")
            
    print(f"Model saved as OBJ to {filename}")