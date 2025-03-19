# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2
from utils.geometry import back_project_2d_to_3d, inverse_ortho

def move_mesh_2d(state, old_lx, old_ly, dx, dy):
    """Move the mesh vertices based on the dragged point and update 3D mesh"""
    # Check if we're dragging a pin (not a landmark)
    is_dragging_pin = (state.drag_index != -1 and 
                       state.drag_index >= len(state.landmark_positions))

    # Get current camera parameters
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Check if we have exactly two pins
    pins = state.pins_per_image[state.current_image_idx]
    has_two_pins = len(pins) == 2
    
    # If we're dragging a pin and have camera parameters
    if is_dragging_pin and camera_matrix is not None and rvec is not None and tvec is not None:
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(pins):
            # Get the dragged pin data
            dragged_pin_data = pins[pin_idx]
            dragged_pin_3d_pos = dragged_pin_data[4] if len(dragged_pin_data) >= 5 else None
            
            if dragged_pin_3d_pos is not None:
                # Save original values to revert if something goes wrong
                original_rvec = rvec.copy()
                original_tvec = tvec.copy()
                original_verts3d = state.verts3d.copy()
                original_verts2d = state.verts2d.copy()
                
                # TWO PINS CASE
                if has_two_pins:
                    # Find the other (fixed) pin
                    other_pin_idx = 1 if pin_idx == 0 else 0
                    other_pin_data = pins[other_pin_idx]
                    other_pin_3d_pos = other_pin_data[4] if len(other_pin_data) >= 5 else None
                    
                    if other_pin_3d_pos is None:
                        # If the other pin doesn't have a 3D position, fall back to single pin case
                        has_two_pins = False
                    else:
                        try:
                            # Project both pins and mesh to 2D using current transform
                            # Get current 2D positions
                            dragged_pin_2d = np.array([old_lx, old_ly])
                            other_pin_2d = np.array([other_pin_data[0], other_pin_data[1]])
                            
                            # Calculate center between the two pins
                            center_2d = (dragged_pin_2d + other_pin_2d) / 2
                            
                            # Get new desired position for dragged pin
                            new_dragged_pin_2d = dragged_pin_2d + np.array([dx, dy])
                            
                            # Calculate vectors from center
                            vec_old = dragged_pin_2d - center_2d
                            vec_new = new_dragged_pin_2d - center_2d
                            
                            # Calculate scaling factor (enlargement/reduction)
                            old_dist = np.linalg.norm(vec_old)
                            new_dist = np.linalg.norm(vec_new)
                            
                            if old_dist > 1e-5:  # Avoid division by zero
                                scale_factor = new_dist / old_dist
                            else:
                                scale_factor = 1.0
                            
                            # Limit scaling to prevent extreme changes
                            scale_factor = np.clip(scale_factor, 0.95, 1.05)
                            
                            # Calculate rotation angle for circular movement
                            angle = 0.0
                            if old_dist > 1e-5 and new_dist > 1e-5:
                                # Normalize vectors
                                vec_old_norm = vec_old / old_dist
                                vec_new_norm = vec_new / new_dist
                                
                                # Calculate angle between vectors
                                dot_product = np.clip(np.dot(vec_old_norm, vec_new_norm), -1.0, 1.0)
                                angle = np.arccos(dot_product)
                                
                                # Determine rotation direction using cross product (2D version)
                                cross_z = vec_old_norm[0] * vec_new_norm[1] - vec_old_norm[1] * vec_new_norm[0]
                                if cross_z < 0:
                                    angle = -angle
                            
                            # Create 2D transformation matrix (rotation + scaling)
                            cos_angle = np.cos(angle)
                            sin_angle = np.sin(angle)
                            rotation_matrix = np.array([
                                [cos_angle, -sin_angle],
                                [sin_angle, cos_angle]
                            ])
                            
                            # Apply transformation to vertices in screen space:
                            # 1. Center the vertices on the pin center
                            centered_verts = state.verts2d - center_2d
                            
                            # 2. Apply rotation
                            rotated_verts = np.zeros_like(centered_verts)
                            for i in range(len(centered_verts)):
                                rotated_verts[i] = rotation_matrix @ centered_verts[i]
                            
                            # 3. Apply scaling
                            scaled_verts = rotated_verts * scale_factor
                            
                            # 4. Move back to original center
                            transformed_verts = scaled_verts + center_2d
                            
                            # Transform the pins too (to keep them attached to mesh)
                            # For the dragged pin
                            centered_dragged = dragged_pin_2d - center_2d
                            rotated_dragged = rotation_matrix @ centered_dragged
                            scaled_dragged = rotated_dragged * scale_factor
                            transformed_dragged = scaled_dragged + center_2d
                            
                            # For the other pin
                            centered_other = other_pin_2d - center_2d
                            rotated_other = rotation_matrix @ centered_other
                            scaled_other = rotated_other * scale_factor
                            transformed_other = scaled_other + center_2d
                            
                            # Check that the transformed positions are reasonable
                            # (not going too far out of bounds)
                            img_w, img_h = state.img_w, state.img_h
                            margin = max(img_w, img_h) * 0.8
                            
                            vertices_in_bounds = np.logical_and(
                                np.logical_and(transformed_verts[:, 0] > -margin, transformed_verts[:, 0] < img_w + margin),
                                np.logical_and(transformed_verts[:, 1] > -margin, transformed_verts[:, 1] < img_h + margin)
                            )
                            
                            pins_in_bounds = (
                                -margin < transformed_dragged[0] < img_w + margin and
                                -margin < transformed_dragged[1] < img_h + margin and
                                -margin < transformed_other[0] < img_w + margin and
                                -margin < transformed_other[1] < img_h + margin
                            )
                            
                            if np.mean(vertices_in_bounds) >= 0.6 and pins_in_bounds:
                                # Directly update 2D vertex positions
                                state.verts2d = transformed_verts
                                
                                # Directly update pin positions in the state
                                state.pins_per_image[state.current_image_idx][pin_idx] = (
                                    transformed_dragged[0], transformed_dragged[1],
                                    dragged_pin_data[2], dragged_pin_data[3], dragged_pin_3d_pos
                                )
                                
                                state.pins_per_image[state.current_image_idx][other_pin_idx] = (
                                    transformed_other[0], transformed_other[1],
                                    other_pin_data[2], other_pin_data[3], other_pin_3d_pos
                                )
                                
                                # Set the flag to skip 3D-2D projection and back-projection
                                # This ensures we use our 2D coordinates directly
                                state.skip_projection = True
                                
                                # Redraw with our 2D changes
                                state.callbacks['redraw'](state)
                                return
                            else:
                                # Revert changes if the transformation would go out of bounds
                                print("Transformation would cause vertices to go out of bounds")
                        
                        except Exception as e:
                            # Revert to original values and fall back to single-pin case if there's an error
                            print(f"Error during dual pin handling: {e}")
                            state.verts2d = original_verts2d
                
                # SINGLE PIN CASE
                if not has_two_pins:
                    # Convert movement to rotation
                    # Calculate movement vector in 2D screen space
                    movement_2d = np.array([dx, dy])
                    
                    # Normalize movement to scale rotation appropriately
                    movement_norm = np.linalg.norm(movement_2d)
                    if movement_norm > 1e-5:  # Only apply rotation if there's significant movement
                        # First, apply rotation as before
                        rotation_scale = 0.001
                        
                        # Get current rotation matrix
                        current_R, _ = cv2.Rodrigues(rvec)
                        
                        # Calculate rotation angles based on movement
                        angle_y = -dx * rotation_scale  # Negative to match natural rotation direction
                        angle_x = dy * rotation_scale
                        
                        # Create rotation matrices for each axis
                        Rx = np.array([
                            [1, 0, 0],
                            [0, np.cos(angle_x), -np.sin(angle_x)],
                            [0, np.sin(angle_x), np.cos(angle_x)]
                        ])
                        
                        Ry = np.array([
                            [np.cos(angle_y), 0, np.sin(angle_y)],
                            [0, 1, 0],
                            [-np.sin(angle_y), 0, np.cos(angle_y)]
                        ])
                        
                        # Combine rotations
                        R_delta = Rx @ Ry
                        
                        # Apply rotation to current rotation
                        new_R = R_delta @ current_R
                        
                        # Convert back to rotation vector
                        new_rvec, _ = cv2.Rodrigues(new_R)
                        
                        # Update rotation for current view
                        state.rotations[state.current_image_idx] = new_rvec
                        
                        try:
                            # Now calculate translation to make the pin follow the mouse
                            # Get current mouse position
                            mouse_pos = np.array([old_lx + dx, old_ly + dy])
                            
                            # Current rotation matrix
                            R, _ = cv2.Rodrigues(new_rvec)
                            
                            # Extract camera intrinsics
                            fx = camera_matrix[0, 0]
                            fy = camera_matrix[1, 1]
                            cx = camera_matrix[0, 2]
                            cy = camera_matrix[1, 2]
                            
                            # Original pin position in camera space
                            pin_camera_orig = R @ dragged_pin_3d_pos + tvec.reshape(3)
                            z_camera = pin_camera_orig[2]
                            
                            # Ensure z is positive and reasonable
                            if z_camera <= 0:
                                z_camera = 1.0
                            
                            # Calculate camera space coordinates that project to mouse position
                            x_camera = z_camera * (mouse_pos[0] - cx) / fx
                            y_camera = z_camera * (mouse_pos[1] - cy) / fy
                            
                            # Form desired camera-space position
                            desired_camera_pos = np.array([x_camera, y_camera, z_camera])
                            
                            # Solve for translation
                            new_tvec = (desired_camera_pos - R @ dragged_pin_3d_pos).reshape(3, 1)
                            
                            # Apply new translation
                            state.translations[state.current_image_idx] = new_tvec
                            
                            # Project all vertices with new rotation and translation
                            projected_verts, _ = cv2.projectPoints(
                                state.verts3d, new_rvec, new_tvec, camera_matrix, np.zeros((4, 1))
                            )
                            
                            # Check if projection is valid
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
                                
                                # Update pin positions
                                state.callbacks['update_custom_pins'](state)
                                return
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
    
    # If not applying rotation + translation, fall back to original weighted movement method
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
    # Skip 3D update if we've directly manipulated 2D vertices in the two-pin case
    if hasattr(state, 'skip_projection') and state.skip_projection:
        state.skip_projection = False
        return
        
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
    # Skip projection if we've directly manipulated 2D vertices in the two-pin case
    if hasattr(state, 'skip_projection') and state.skip_projection:
        state.skip_projection = False
        return True
        
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


def transform_mesh_rigid(state, dragged_pin_idx, old_x, old_y, new_x, new_y):
    """
    Apply a rigid transformation to the mesh, preserving the position of 
    all pins except the dragged one, and updating the scale based on dragging.
    
    Args:
        state: Application state
        dragged_pin_idx: Index of the pin being dragged (in state.pins_per_image)
        old_x, old_y: Original position of the dragged pin
        new_x, new_y: New position of the dragged pin
    """
    # Get pins for current image
    pins = state.pins_per_image[state.current_image_idx]
    
    # We need at least 2 pins (dragged + anchor)
    if len(pins) < 2:
        return
    
    # Find anchor pin (first non-dragged pin)
    anchor_idx = None
    for i, pin in enumerate(pins):
        if i != dragged_pin_idx:
            anchor_idx = i
            break
    
    if anchor_idx is None:
        return  # Should not happen
    
    # Get anchor pin coordinates
    anchor_x, anchor_y = pins[anchor_idx][0], pins[anchor_idx][1]
    
    # Calculate vectors before and after dragging
    vec_before = np.array([old_x - anchor_x, old_y - anchor_y])
    vec_after = np.array([new_x - anchor_x, new_y - anchor_y])
    
    # Calculate scale change
    len_before = np.linalg.norm(vec_before)
    len_after = np.linalg.norm(vec_after)
    scale_factor = len_after / len_before if len_before > 0 else 1.0
    
    # Calculate rotation angle (in radians)
    dot_product = np.dot(vec_before, vec_after)
    det = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]
    angle = np.arctan2(det, dot_product)
    
    # Create transformation matrix (scale and rotation around anchor)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Translation to anchor (for transformation center)
    translate_to_anchor = np.array([
        [1, 0, -anchor_x],
        [0, 1, -anchor_y],
        [0, 0, 1]
    ])
    
    # Scale and rotation
    scale_and_rotate = np.array([
        [scale_factor * cos_angle, -scale_factor * sin_angle, 0],
        [scale_factor * sin_angle, scale_factor * cos_angle, 0],
        [0, 0, 1]
    ])
    
    # Translation back from anchor
    translate_from_anchor = np.array([
        [1, 0, anchor_x],
        [0, 1, anchor_y],
        [0, 0, 1]
    ])
    
    # Combined transformation matrix
    transform = translate_from_anchor @ scale_and_rotate @ translate_to_anchor
    
    # Apply transformation to 2D vertices
    verts_homogeneous = np.column_stack((state.verts2d, np.ones(len(state.verts2d))))
    transformed_verts = (transform @ verts_homogeneous.T).T
    state.verts2d = transformed_verts[:, :2]
    
    # Update 3D vertices based on the 2D transformation
    update_3d_vertices(state)
    
    # Update landmarks and custom pins
    from model.landmarks import update_all_landmarks
    update_all_landmarks(state)
    from model.pins import update_custom_pins
    update_custom_pins(state)