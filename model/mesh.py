# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
from utils.geometry import back_project_2d_to_3d, inverse_ortho

def move_mesh_2d(state, old_lx, old_ly, dx, dy):
    """Move the mesh vertices based on the dragged point and update 3D mesh"""
    # Calculate distance from each vertex to the dragged point
    diffs = state.verts2d - np.array([old_lx, old_ly])
    dists_sq = np.sum(diffs**2, axis=1)
    
    # Calculate radial weights - inverse distance weighting
    radial_w = 1.0 / (1.0 + state.alpha * dists_sq)
    
    # If dragging a landmark, use skinning weights for better deformation
    if state.drag_index != -1 and state.drag_index < len(state.landmark_positions):
        f_idx = state.face_for_lmk[state.drag_index]
        v_indices = state.faces[f_idx]
        avg_weight = np.mean(state.weights[v_indices, :], axis=0)
        norm_avg = avg_weight / np.sum(avg_weight)
        joint_w = np.dot(state.weights, norm_avg)
        combined_w = radial_w * joint_w
    elif state.drag_index != -1:  # Dragging a custom pin
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(state.pins_per_image[state.current_image_idx]):
            # Handle both 4-tuple and 5-tuple pin formats
            pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
            # Extract face_idx which is always at index 2
            face_idx = pin_data[2]
            
            v_indices = state.faces[face_idx]
            avg_weight = np.mean(state.weights[v_indices, :], axis=0)
            norm_avg = avg_weight / np.sum(avg_weight)
            joint_w = np.dot(state.weights, norm_avg)
            combined_w = radial_w * joint_w
    else:
        combined_w = radial_w
    
    # Store original 2D positions before modification
    original_verts2d = state.verts2d.copy()
    
    # Apply the calculated weights to move vertices in 2D
    shift = np.column_stack((combined_w * dx, combined_w * dy))
    state.verts2d += shift
    
    # Update the 3D vertices based on the 2D movement
    update_3d_vertices(state, original_verts2d)

def update_3d_vertices(state, original_verts2d=None):
    """
    Update the 3D vertices based on the current 2D projection
    
    Args:
        state: Application state
        original_verts2d: Original 2D vertices before modification (for differential update)
    """
    # Get current camera parameters for this view
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Check if we have camera parameters for this view
    if camera_matrix is not None and rvec is not None and tvec is not None:
        # Use perspective back-projection when we have camera parameters
        updated_verts3d = back_project_2d_to_3d(
            state.verts2d,
            state.verts3d,  # Use current 3D vertices, not defaults
            camera_matrix,
            rvec,
            tvec
        )
        
        if updated_verts3d is not None:
            # Blend the updated vertices with current vertices to maintain stability
            blend_factor = 0.8  # 80% new, 20% old - can be adjusted for stability
            state.verts3d = blend_factor * updated_verts3d + (1 - blend_factor) * state.verts3d
    else:
        # Use inverse orthographic projection when no camera parameters
        # Calculate the model parameters for inverse orthographic projection
        mn = state.verts3d[:, :2].min(axis=0)
        mx = state.verts3d[:, :2].max(axis=0)
        c3d = 0.5 * (mn + mx)
        s3d = (mx - mn).max()
        sc = 0.8 * min(state.img_w, state.img_h) / s3d
        c2d = np.array([state.img_w/2.0, state.img_h/2.0])
        
        # Use inverse orthographic projection
        updated_verts3d = inverse_ortho(
            state.verts2d,
            state.verts3d,  # Use current 3D vertices, not defaults
            c3d,
            c2d,
            sc
        )
        
        state.verts3d = updated_verts3d

def project_current_3d_to_2d(state):
    """
    Project the current 3D vertices to 2D for the current view
    
    Args:
        state: Application state
    """
    # Get current camera parameters
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Check if we have camera parameters for this view
    if camera_matrix is not None and rvec is not None and tvec is not None:
        # Use perspective projection
        from utils.geometry import project_3d_to_2d
        
        projected_verts = project_3d_to_2d(
            state.verts3d,  # Use current 3D vertices, not defaults
            camera_matrix,
            rvec,
            tvec
        )
        
        if projected_verts is not None:
            state.verts2d = projected_verts
            return True
    
    # Fall back to orthographic projection if no camera parameters or projection failed
    mn = state.verts3d[:, :2].min(axis=0)
    mx = state.verts3d[:, :2].max(axis=0)
    c3d = 0.5 * (mn + mx)
    s3d = (mx - mn).max()
    sc = 0.8 * min(state.img_w, state.img_h) / s3d
    c2d = np.array([state.img_w/2.0, state.img_h/2.0])
    
    from utils.geometry import ortho
    state.verts2d = ortho(state.verts3d, c3d, c2d, sc)
    return True