"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Mesh projection module.

This module handles the projection of 3D mesh vertices to 2D screen coordinates
using a perspective camera model. It provides functionality for configuring
camera parameters, applying projections, and handling projection failures.
"""
import numpy as np
import cv2
from utils.geometry import project_3d_to_2d, calculate_front_facing

def project_current_3d_to_2d(state):
    """
    Project the current 3D vertices to 2D using camera parameters.
    
    Args:
        state: Application state containing 3D vertices and camera parameters
        
    Returns:
        bool: True if projection successful, False otherwise
    """
    # Skip projection if we've directly manipulated 2D vertices in the two-pin case
    if hasattr(state, 'skip_projection') and state.skip_projection:
        state.skip_projection = False
        return True
        
    # Get current camera parameters
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Ensure we have valid camera parameters
    if camera_matrix is None or rvec is None or tvec is None:
        # Initialize default camera parameters if not present
        focal_length = max(state.img_w, state.img_h)
        camera_matrix = np.array([
            [focal_length, 0, state.img_w / 2],
            [0, focal_length, state.img_h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Calculate center of the model
        center = np.mean(state.verts3d, axis=0)
        
        # Initialize rotation (identity rotation)
        R = np.eye(3, dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)
        
        # Position the model in front of the camera
        # Translation vector: t = -R*C where C is the camera center
        distance = focal_length * 1.5
        tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
        
        # Save camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Use perspective projection with equation x' = K[R|t]X
    # where K is camera matrix, [R|t] is extrinsic matrix, X is 3D point
    projected_verts = project_3d_to_2d(
        state.verts3d,  # Use current 3D vertices, not defaults
        camera_matrix,
        rvec,
        tvec
    )
    
    if projected_verts is not None:
        # Remember dragged landmark position if any
        dragged_landmark_pos = None
        if hasattr(state, 'drag_index') and state.drag_index != -1 and state.drag_index < len(state.landmark_positions):
            dragged_landmark_pos = state.landmark_positions[state.drag_index]
        
        # Update 2D vertices
        state.verts2d = projected_verts
        
        # Update backface culling
        state.front_facing = calculate_front_facing(
            state.verts3d, state.faces,
            camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
        )
        
        # Ensure landmarks are updated but preserve the dragged one
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        
        # Restore dragged landmark position if needed
        if dragged_landmark_pos is not None:
            state.landmark_positions[state.drag_index] = dragged_landmark_pos
        
        # Make sure landmark pins remain visible
        if hasattr(state, 'landmark_pins_hidden'):
            was_hidden = state.landmark_pins_hidden
            if not was_hidden:
                state.landmark_pins_hidden = False
        
        return True
    
    # If projection fails, try with adjusted parameters
    return _retry_projection_with_adjusted_params(state)


def _retry_projection_with_adjusted_params(state):
    """
    Retry projection with adjusted camera parameters if initial projection fails.
    
    Args:
        state: Application state
    
    Returns:
        bool: True if projection successful, False otherwise
    """
    print("Warning: Initial projection failed. Trying with adjusted parameters.")
    
    # Recalculate camera parameters with a larger distance
    focal_length = max(state.img_w, state.img_h)
    center = np.mean(state.verts3d, axis=0)
    R = np.eye(3, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    distance = focal_length * 2.0  # Increased distance from camera
    tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
    
    # Recalculate camera matrix
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Save adjusted parameters
    state.camera_matrices[state.current_image_idx] = camera_matrix
    state.rotations[state.current_image_idx] = rvec
    state.translations[state.current_image_idx] = tvec
    
    # Try projection again
    projected_verts = project_3d_to_2d(
        state.verts3d,
        camera_matrix,
        rvec,
        tvec
    )
    
    if projected_verts is not None:
        state.verts2d = projected_verts
        state.front_facing = np.ones(len(state.faces), dtype=bool)
        
        # Update landmarks but preserve dragged one
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        
        # Make sure landmark pins remain visible
        if hasattr(state, 'landmark_pins_hidden'):
            was_hidden = state.landmark_pins_hidden
            if not was_hidden:
                state.landmark_pins_hidden = False
        
        return True
    
    # If all else fails, log error but don't crash
    print("Error: Failed to project 3D to 2D with perspective projection.")
    return False