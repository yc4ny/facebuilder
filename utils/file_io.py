# Copyright (c) CUBOX, Inc. and its affiliates.
import pickle
import os
import cv2
import numpy as np

def load_images_from_directory(dir_path="images"):
    """Load all images from a directory"""
    images = []
    
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return [cv2.imread("images/side.jpg")]
    
    for filename in sorted(os.listdir(dir_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Ensure image has a reasonable size for display
                img_h, img_w = img.shape[:2]
                
                # If image is very large, resize it to a more manageable size
                max_dimension = 1600  # Maximum dimension for display
                if img_h > max_dimension or img_w > max_dimension:
                    scale = max_dimension / max(img_h, img_w)
                    new_width = int(img_w * scale)
                    new_height = int(img_h * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"Resized image {filename} from {img_w}x{img_h} to {new_width}x{new_height}")
                
                images.append(img)
                print(f"Loaded image: {filename} - Size: {img.shape[1]}x{img.shape[0]}")
    
    if not images:
        print("No valid images found in directory")
        return [cv2.imread("images/side.jpg")]
    
    return images

def save_model(verts2d, verts3d, faces, pins_per_image, camera_matrices, rotations, translations):
    """
    Save the current 3D model to file, using the modified 3D vertices
    
    Args:
        verts2d: Current 2D vertices (for reference)
        verts3d: Current 3D vertices (modified by user)
        faces: Face indices
        pins_per_image: Custom pins for each image
        camera_matrices: Camera matrices for each view
        rotations: Rotation vectors for each view
        translations: Translation vectors for each view
    """
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Save model data as pickle
    model_data = {
        "verts2d": verts2d,
        "verts3d": verts3d,  # Now contains the modified 3D vertices
        "faces": faces,
        "pins": pins_per_image,
        "camera_matrices": camera_matrices,
        "rotations": rotations,
        "translations": translations
    }
    
    with open("output/face_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model saved to output/face_model.pkl")
    
    # Save as OBJ file for direct import into Meshlab
    # Now using the modified 3D vertices
    save_to_obj(verts3d, faces, "output/face_model.obj")
    
    # Also save the default model for comparison
    try:
        with open("output/face_model_default.obj", "w") as f:
            f.write("# OBJ file of the original unmodified model\n")
            
        print("Also saved original model to output/face_model_default.obj for comparison")
    except Exception as e:
        print(f"Error saving default model: {e}")

def save_to_obj(vertices, faces, filename="output/face_model.obj"):
    """
    Save the 3D model as an OBJ file format
    
    Args:
        vertices: Array of 3D vertex positions
        faces: Array of face indices (0-indexed)
        filename: Output filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# OBJ file created by Face Builder\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (OBJ is 1-indexed)
        for face in faces:
            # Convert from 0-indexed to 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # Also save a visualization of the final mesh
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