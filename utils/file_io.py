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
    """Save the current 3D model to file"""
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Save model data as pickle
    model_data = {
        "verts2d": verts2d,
        "verts3d": verts3d,
        "faces": faces,
        "pins": pins_per_image,
        "camera_matrices": camera_matrices,
        "rotations": rotations,
        "translations": translations
    }
    
    with open("output/face_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model saved to output/face_model.pkl")
    
    # Also save as OBJ file for direct import into Meshlab
    save_to_obj(verts3d, faces, "output/face_model.obj")

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
            
    print(f"Model saved as OBJ to {filename}")
