# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np

def ortho(v3d, c3d, c2d, sc):
    """
    Orthographic projection of 3D points to 2D
    """
    xy = v3d[:, :2] - c3d
    xy *= sc
    xy[:, 1] = -xy[:, 1]
    xy += c2d
    return xy

def remove_eyeballs(verts, faces, start_idx=3931, end_idx=5022):
    """
    Remove eyeball vertices from the mesh and update faces accordingly
    
    Args:
        verts: Array of vertex positions
        faces: Array of face indices
        start_idx: Start index of eyeball vertices (inclusive)
        end_idx: End index of eyeball vertices (inclusive)
        
    Returns:
        new_verts: Updated vertices array without eyeball vertices
        new_faces: Updated faces array with adjusted indices
    """
    # Create a mask for vertices to keep
    keep_mask = np.ones(verts.shape[0], dtype=bool)
    keep_mask[start_idx:end_idx+1] = False
    
    # Create index mapping from old to new vertex indices
    old_to_new = np.cumsum(keep_mask) - 1
    
    # Get new vertices
    new_verts = verts[keep_mask]
    
    # Keep only faces that don't use eyeball vertices
    valid_faces = []
    for face in faces:
        if np.all((face < start_idx) | (face > end_idx)):
            # Remap vertex indices
            new_face = [old_to_new[idx] for idx in face]
            valid_faces.append(new_face)
    
    new_faces = np.array(valid_faces)
    
    print(f"Removed {end_idx - start_idx + 1} eyeball vertices")
    print(f"Faces reduced from {len(faces)} to {len(new_faces)}")
    
    return new_verts, new_faces