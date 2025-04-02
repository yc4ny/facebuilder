"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Mesh deformation module.

This module provides functionality for deforming the 3D mesh based on pin movements,
including direct manipulation of vertices and back-projection to 3D space.
"""
import numpy as np
import cv2
from utils.geometry import back_project_2d_to_3d, calculate_front_facing
from model.landmarks import update_all_landmarks
from model.pins import update_custom_pins

def move_mesh_2d(state, old_lx, old_ly, dx, dy):
    """
    Move the mesh based on a single point of interaction.
    
    This function handles mesh movement in several modes:
    1. Rotation + translation based on a single pin movement
    2. Weighted deformation based on inverse distance from the dragged point
    
    Args:
        state (FaceBuilderState): Application state
        old_lx (float): Original x-coordinate of the dragged point
        old_ly (float): Original y-coordinate of the dragged point
        dx (float): X-displacement of the dragged point
        dy (float): Y-displacement of the dragged point
        
    Returns:
        None
    """
    # Check if we're dragging a pin (not a landmark)
    is_dragging_pin = (state.drag_index != -1)
    is_dragging_landmark = (state.drag_index != -1 and 
                           state.drag_index < len(state.landmark_positions))
    is_dragging_custom_pin = (state.drag_index != -1 and 
                             state.drag_index >= len(state.landmark_positions))

    # Get current camera parameters
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Count both landmarks and custom pins to determine pin mode
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    landmark_count = 0 if landmark_pins_hidden else len(state.landmark_positions)
    custom_pin_count = len(state.pins_per_image[state.current_image_idx])
    total_pin_count = landmark_count + custom_pin_count
    
    # Determine if we have multiple pins (for mode selection)
    has_multiple_pins = total_pin_count >= 2
    
    # Special handling for dragging landmarks or custom pins with camera parameters
    if is_dragging_pin and camera_matrix is not None and rvec is not None and tvec is not None:
        pin_3d_pos = None
        
        # Get 3D position based on pin type
        if is_dragging_landmark:
            # For landmarks, calculate 3D position from landmark3d
            landmark_idx = state.drag_index
            if landmark_idx < len(state.landmark3d):
                pin_3d_pos = state.landmark3d[landmark_idx]
        elif is_dragging_custom_pin:
            # For custom pins, extract 3D position from pin data
            pin_idx = state.drag_index - len(state.landmark_positions)
            if pin_idx < len(state.pins_per_image[state.current_image_idx]):
                dragged_pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
                pin_3d_pos = dragged_pin_data[4] if len(dragged_pin_data) >= 5 else None
        
        if pin_3d_pos is not None:
            # Save original values to revert if something goes wrong
            original_rvec = rvec.copy()
            original_tvec = tvec.copy()
            original_verts3d = state.verts3d.copy()
            original_verts2d = state.verts2d.copy()
            
            # Get target position
            new_x = old_lx + dx
            new_y = old_ly + dy
            
            # MULTIPLE PINS CASE (2 or more pins)
            if has_multiple_pins:
                # For landmarks, convert to a pin index relative to all pins
                if is_dragging_landmark:
                    # If we're dragging a landmark, use its index directly
                    pin_idx = state.drag_index
                else:
                    # Otherwise, use the custom pin index (already adjusted)
                    pin_idx = state.drag_index - len(state.landmark_positions)
                
                # Handle with transform_mesh_rigid which supports 2-pin, 3-pin, and 4+ pin cases
                try:
                    from model.mesh.transform import transform_mesh_rigid
                    transform_mesh_rigid(state, state.drag_index, old_lx, old_ly, new_x, new_y)
                    return
                except Exception as e:
                    # Revert to original values and fall back to single pin case if there's an error
                    print(f"Error during multi-pin handling: {e}")
                    state.verts2d = original_verts2d
            
            # SINGLE PIN CASE
            if not has_multiple_pins or state.verts2d is original_verts2d:
                if _handle_single_pin_3d_rotation(state, pin_3d_pos, old_lx, old_ly, dx, dy, 
                                               rvec, tvec, camera_matrix, original_rvec,
                                               original_tvec, new_x, new_y):
                    return
    
    # If not applying rotation + translation, fall back to weighted movement method
    _apply_weighted_deformation(state, old_lx, old_ly, dx, dy)


def _handle_single_pin_3d_rotation(state, pin_3d_pos, old_lx, old_ly, dx, dy, rvec, tvec, 
                                 camera_matrix, original_rvec, original_tvec, new_x, new_y):
    """
    Handle single pin case with 3D rotation and translation.
    
    This function applies a 3D rotation based on the movement direction, then
    adjusts the translation to ensure the pin follows the cursor position.
    
    Args:
        state (FaceBuilderState): Application state
        pin_3d_pos (ndarray): 3D position of the pin
        old_lx, old_ly (float): Original pin position
        dx, dy (float): Movement delta
        rvec, tvec (ndarray): Current rotation and translation vectors
        camera_matrix (ndarray): Camera intrinsic matrix
        original_rvec, original_tvec (ndarray): Original rotation and translation to revert if needed
        new_x, new_y (float): New position of the pin
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert movement to rotation
    # Calculate movement vector in 2D screen space
    movement_2d = np.array([dx, dy])
    
    # Normalize movement to scale rotation appropriately
    movement_norm = np.linalg.norm(movement_2d)
    if movement_norm > 1e-5:  # Only apply rotation if there's significant movement
        # Apply rotation proportional to movement
        rotation_scale = 0.001
        
        # Get current rotation matrix
        current_R, _ = cv2.Rodrigues(rvec)
        
        # Calculate rotation angles based on movement
        # Negative angle for X to match natural rotation direction
        angle_y = -dx * rotation_scale
        angle_x = dy * rotation_scale
        
        # Create rotation matrices for each axis using Rodrigues' formula:
        # Rx = [1      0       0    ]
        #      [0  cos(θx) -sin(θx) ]
        #      [0  sin(θx)  cos(θx) ]
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        
        # Ry = [ cos(θy)  0  sin(θy) ]
        #      [    0     1     0    ]
        #      [-sin(θy)  0  cos(θy) ]
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        # Combine rotations: R_delta = Rx * Ry
        R_delta = Rx @ Ry
        
        # Apply to current rotation: R_new = R_delta * R_current
        new_R = R_delta @ current_R
        
        # Convert back to rotation vector
        new_rvec, _ = cv2.Rodrigues(new_R)
        
        # Update rotation for current view
        state.rotations[state.current_image_idx] = new_rvec
        
        try:
            # Now calculate translation to make the pin follow the mouse
            # Get current mouse position
            mouse_pos = np.array([new_x, new_y])
            
            # Current rotation matrix
            R, _ = cv2.Rodrigues(new_rvec)
            
            # Extract camera intrinsics
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Original pin position in camera space: X_camera = R*X_world + t
            pin_camera_orig = R @ pin_3d_pos + tvec.reshape(3)
            z_camera = pin_camera_orig[2]
            
            # Ensure z is positive and reasonable
            if z_camera <= 0:
                z_camera = 1.0
            
            # Calculate camera space coordinates that project to mouse position
            # Using perspective projection equations:
            # x = (X * fx / Z) + cx
            # y = (Y * fy / Z) + cy
            # Rearranging to solve for X and Y:
            # X = (x - cx) * Z / fx
            # Y = (y - cy) * Z / fy
            x_camera = z_camera * (mouse_pos[0] - cx) / fx
            y_camera = z_camera * (mouse_pos[1] - cy) / fy
            
            # Form desired camera-space position
            desired_camera_pos = np.array([x_camera, y_camera, z_camera])
            
            # Solve for translation: t = X_camera - R*X_world
            new_tvec = (desired_camera_pos - R @ pin_3d_pos).reshape(3, 1)
            
            # Apply new translation
            state.translations[state.current_image_idx] = new_tvec
            
            # Project all vertices with new rotation and translation
            projected_verts, _ = cv2.projectPoints(
                state.verts3d, new_rvec, new_tvec, camera_matrix, np.zeros((4, 1))
            )
            
            # Check if projection is valid (not too far outside viewport)
            projected_2d = projected_verts.reshape(-1, 2)
            img_w, img_h = state.img_w, state.img_h
            margin = max(img_w, img_h) * 0.8
            
            vertices_in_bounds = np.logical_and(
                np.logical_and(projected_2d[:, 0] > -margin, projected_2d[:, 0] < img_w + margin),
                np.logical_and(projected_2d[:, 1] > -margin, projected_2d[:, 1] < img_h + margin)
            )
            
            if np.mean(vertices_in_bounds) >= 0.6:
                # Update 2D vertex positions
                state.verts2d = projected_2d
                
                # CRITICAL: Update the dragged pin position to follow mouse
                _update_dragged_pin_position(state, new_x, new_y)
                
                # Update other pins
                update_all_landmarks(state)
                update_custom_pins(state)
                return True
            else:
                # Revert to original values if projection is invalid
                state.rotations[state.current_image_idx] = original_rvec
                state.translations[state.current_image_idx] = original_tvec
                print("Pin movement would cause too many vertices to go out of bounds.")
        
        except Exception as e:
            # Revert to original values if there's an error
            state.rotations[state.current_image_idx] = original_rvec
            state.translations[state.current_image_idx] = original_tvec
            print(f"Error during pin following: {e}")
    
    return False


def _apply_weighted_deformation(state, old_lx, old_ly, dx, dy):
    """
    Apply weighted deformation to the mesh based on distance from the dragged point.
    
    This approach uses a combination of:
    1. Inverse distance weighting for all vertices
    2. Skinning weights for better deformation when dragging landmarks/pins
    
    Args:
        state (FaceBuilderState): Application state
        old_lx, old_ly (float): Original position of the dragged point
        dx, dy (float): Movement delta in x and y directions
        
    Returns:
        None
    """
    # Calculate distance from each vertex to the dragged point
    diffs = state.verts2d - np.array([old_lx, old_ly])
    dists_sq = np.sum(diffs**2, axis=1)
    
    # Calculate radial weights - inverse distance weighting
    # w_i = 1 / (1 + α * d_i²)
    # where d_i is the distance from vertex i to the dragged point
    # and α is a parameter controlling the falloff rate
    radial_w = 1.0 / (1.0 + state.alpha * dists_sq)
    
    # If dragging a landmark, use skinning weights for better deformation
    if state.drag_index != -1 and state.drag_index < len(state.landmark_positions):
        # For landmarks, use skinning weights from the landmark's face
        combined_w = _compute_joint_weights(state, state.face_for_lmk[state.drag_index], radial_w)
    elif state.drag_index != -1:  # Dragging a custom pin
        # For custom pins, use skinning weights from the pin's face
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(state.pins_per_image[state.current_image_idx]):
            pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
            face_idx = pin_data[2]  # face_idx is always at index 2
            combined_w = _compute_joint_weights(state, face_idx, radial_w)
            
            # Get new position for direct pin update
            new_x = old_lx + dx
            new_y = old_ly + dy
    else:
        # If not dragging anything, just use radial weights
        combined_w = radial_w
    
    # Store original 2D positions before modification
    original_verts2d = state.verts2d.copy()
    
    # Apply the calculated weights to move vertices in 2D
    # δv_i = w_i * [dx, dy]  for each vertex i
    shift = np.column_stack((combined_w * dx, combined_w * dy))
    state.verts2d += shift
    
    # If we're dragging a pin, update its position directly
    if state.drag_index != -1:
        _update_dragged_pin_position(state, old_lx + dx, old_ly + dy)
    
    # Update the 3D vertices based on the 2D movement
    update_3d_vertices(state, original_verts2d)


def _compute_joint_weights(state, face_idx, radial_w):
    """
    Compute combined joint weights for deformation.
    
    This combines skinning weights with radial weights for more natural deformation.
    
    Args:
        state (FaceBuilderState): Application state
        face_idx (int): Face index to get skinning weights from
        radial_w (ndarray): Radial weights for all vertices
        
    Returns:
        ndarray: Combined weights for all vertices
    """
    # Get vertices of the face
    v_indices = state.faces[face_idx]
    
    # Compute average skinning weights for this face
    avg_weight = np.mean(state.weights[v_indices, :], axis=0)
    
    # Normalize to sum to 1
    norm_avg = avg_weight / np.sum(avg_weight)
    
    # Apply to all vertices using dot product
    # This distributes the influence according to skinning weights
    joint_w = np.dot(state.weights, norm_avg)
    
    # Combine with radial weights
    return radial_w * joint_w


def _update_dragged_pin_position(state, new_x, new_y):
    """
    Update the position of the dragged pin.
    
    Args:
        state (FaceBuilderState): Application state 
        new_x, new_y (float): New position for the dragged pin
        
    Returns:
        None
    """
    if state.drag_index < len(state.landmark_positions):
        # Update landmark position directly
        state.landmark_positions[state.drag_index] = (new_x, new_y)
    else:
        # Update custom pin
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(state.pins_per_image[state.current_image_idx]):
            pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
            
            if len(pin_data) >= 5:  # 5-tuple format
                state.pins_per_image[state.current_image_idx][pin_idx] = (
                    new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
                )
            else:  # 4-tuple format (legacy)
                state.pins_per_image[state.current_image_idx][pin_idx] = (
                    new_x, new_y, pin_data[2], pin_data[3]
                )


def update_3d_vertices(state, original_verts2d=None):
    """
    Update the 3D vertices based on 2D vertex positions.
    
    This performs back-projection from 2D to 3D using the current camera parameters.
    
    Args:
        state (FaceBuilderState): Application state
        original_verts2d (ndarray, optional): Original 2D vertex positions for reference
        
    Returns:
        None
    """
    # Skip 3D update if we've directly manipulated 2D vertices in the two-pin case
    if hasattr(state, 'skip_projection') and state.skip_projection:
        state.skip_projection = False
        return
        
    # Get current camera parameters for this view
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
        
        # Initialize default rotation and translation
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.array([[0, 0, focal_length]], dtype=np.float32).T
        
        # Save camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Back-project 2D points to 3D using ray-casting approach
    updated_verts3d = back_project_2d_to_3d(
        state.verts2d,
        state.verts3d,  # Use current 3D vertices, not defaults
        camera_matrix,
        rvec,
        tvec
    )
    
    if updated_verts3d is not None:
        # Blend the updated vertices with current vertices to maintain stability
        # X_new = α * X_updated + (1-α) * X_current
        blend_factor = 0.8  # 80% new, 20% old - can be adjusted for stability
        state.verts3d = blend_factor * updated_verts3d + (1 - blend_factor) * state.verts3d
    
    # After updating 3D vertices, update front_facing property
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )