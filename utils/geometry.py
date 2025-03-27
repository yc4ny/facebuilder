# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2

def project_3d_to_2d(verts3d, camera_matrix, rvec, tvec):
    """
    Project 3D vertices to 2D using camera parameters
    
    Args:
        verts3d: 3D vertices
        camera_matrix: Camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        
    Returns:
        2D projected vertices
    """
    if rvec is None or tvec is None:
        return None
        
    try:
        projected_verts, _ = cv2.projectPoints(
            np.array(verts3d, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        return projected_verts.reshape(-1, 2)
    except cv2.error:
        return None

def back_project_2d_to_3d(verts2d, verts3d_ref, camera_matrix, rvec, tvec):
    """
    Back-project 2D points to 3D using reference 3D points
    For spherical rotation, preserves the distance from center
    
    Args:
        verts2d: Current 2D vertices
        verts3d_ref: Reference 3D vertices (original shape)
        camera_matrix: Camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        
    Returns:
        Updated 3D vertices
    """
    if rvec is None or tvec is None:
        return None
    
    # Convert to numpy arrays
    verts2d = np.array(verts2d, dtype=np.float32)
    verts3d_ref = np.array(verts3d_ref, dtype=np.float32)
    
    # Detect if we're in spherical rotation mode by checking landmark movements
    # In spherical rotation, vertices maintain distance from center
    # First, compute center of mass for original and current 3D vertices
    com_orig = np.mean(verts3d_ref, axis=0)
    
    # Get the rotation matrix from rotation vector
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    
    # Check if we're doing a spherical rotation
    # In that case, we'll preserve distances from the center
    is_spherical_rotation = False
    
    # Count how many vertices have significantly changed distance from center in 2D
    # If fewer than 5% have changed, we're likely doing a spherical rotation
    center_2d = np.mean(verts2d, axis=0)
    dist_2d = np.linalg.norm(verts2d - center_2d, axis=1)
    
    # Create a new 3D vertices array based on the back-projection
    verts3d_new = verts3d_ref.copy()
    
    # Determine transformation type based on movement patterns
    # Get camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    
    # Camera center in world coordinates
    camera_center = -R_inv.dot(tvec).ravel()
    
    # For each vertex, find the best 3D position
    for i in range(len(verts2d)):
        # Get 2D target point
        x, y = verts2d[i]
        
        # Create ray direction in camera coordinates
        ray_camera = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
        
        # Transform ray to world coordinates
        ray_world = R_inv.dot(ray_camera)
        ray_world = ray_world / np.linalg.norm(ray_world)  # Normalize
        
        # Original 3D point position
        orig_pos = verts3d_ref[i]
        
        # Vector from camera center to reference point
        w = orig_pos - camera_center
        
        # Distance from center to original vertex
        dist_from_center = np.linalg.norm(orig_pos - com_orig)
        
        # Project w onto ray_world to find closest point on ray
        proj_length = np.dot(w, ray_world)
        
        # Calculate the closest point on the ray
        closest_point = camera_center + proj_length * ray_world
        
        # For spherical rotation: preserve distance from center
        # Find the point on the ray that preserves the distance from the center
        if is_spherical_rotation:
            # Direction from center to the point on the ray
            dir_to_ray = closest_point - com_orig
            dir_to_ray = dir_to_ray / np.linalg.norm(dir_to_ray)
            
            # Place point at the correct distance from center
            spherical_point = com_orig + dir_to_ray * dist_from_center
            
            # Update the vertex
            verts3d_new[i] = spherical_point
        else:
            # Standard back-projection - use closest point on ray
            verts3d_new[i] = closest_point
    
    return verts3d_new

def calculate_front_facing(verts3d, faces, camera_matrix=None, rvec=None, tvec=None):
    """
    Calculate which faces are front-facing (facing the camera)
    
    Args:
        verts3d: 3D vertices
        faces: Face indices
        camera_matrix: Camera intrinsic matrix (optional)
        rvec: Rotation vector (optional)
        tvec: Translation vector (optional)
        
    Returns:
        Boolean array indicating which faces are front-facing
    """
    # If we have camera parameters, use them to determine camera position in world space
    if rvec is not None and tvec is not None:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate camera position in world coordinates
        camera_center = -R.T.dot(tvec).ravel()
    else:
        # Default camera position along the z-axis
        camera_center = np.array([0, 0, -1])
    
    # Determine front-facing for each face
    front_facing = np.zeros(len(faces), dtype=bool)
    
    for i, (i0, i1, i2) in enumerate(faces):
        # Get the vertices of this face
        v0 = verts3d[i0]
        v1 = verts3d[i1]
        v2 = verts3d[i2]
        
        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize the normal
        normal_length = np.linalg.norm(normal)
        if normal_length > 1e-10:  # Avoid division by zero
            normal = normal / normal_length
        
        # Vector from any point on the face to the camera
        face_center = (v0 + v1 + v2) / 3
        view_vector = camera_center - face_center
        
        # Normalize view vector
        view_length = np.linalg.norm(view_vector)
        if view_length > 1e-10:  # Avoid division by zero
            view_vector = view_vector / view_length
        
        # Dot product between normal and view vector
        # If positive, the face is front-facing (normal points towards camera)
        dot_product = np.dot(normal, view_vector)
        front_facing[i] = (dot_product > 0)
    
    return front_facing

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