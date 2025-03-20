# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2
from utils.geometry import back_project_2d_to_3d, inverse_ortho

def update_3d_vertices(state, original_verts2d=None):
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

def move_mesh_2d(state, old_lx, old_ly, dx, dy):
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
    has_multiple_pins = len(pins) >= 2
    
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
                
                # MULTIPLE PINS CASE (2 or more pins)
                if has_multiple_pins:
                    # Handle with transform_mesh_rigid which now supports both 2-pin and 3+ pin cases
                    try:
                        # Get target position
                        new_x = old_lx + dx
                        new_y = old_ly + dy
                        
                        transform_mesh_rigid(state, pin_idx, old_lx, old_ly, new_x, new_y)
                        
                        # Set the flag to skip 3D-2D projection and back-projection
                        # This ensures we use our 2D coordinates directly
                        state.skip_projection = True
                        
                        # Redraw with our 2D changes
                        state.callbacks['redraw'](state)
                        return
                    except Exception as e:
                        # Revert to original values and fall back to single pin case if there's an error
                        print(f"Error during multi-pin handling: {e}")
                        state.verts2d = original_verts2d
                
                # SINGLE PIN CASE
                if not has_multiple_pins:
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


def project_current_3d_to_2d(state):
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
    Transform the mesh based on pin movement.
    For 2+ pins: Uses rigid transformation that keeps all non-dragged pins fixed.
    """
    # Get pins for current image
    pins = state.pins_per_image[state.current_image_idx]
    
    # We need at least 2 pins (dragged + anchor)
    if len(pins) < 2:
        return
    
    # Collect all non-dragged pins
    fixed_pins = []
    for i, pin in enumerate(pins):
        if i != dragged_pin_idx:
            fixed_pins.append((i, pin[0], pin[1]))
    
    # For all multi-pin cases (2 or more pins)
    if len(fixed_pins) > 0:
        # We'll use the first fixed pin as our primary anchor
        anchor_idx = fixed_pins[0][0]
        anchor_x, anchor_y = fixed_pins[0][1], fixed_pins[0][2]
        
        # Calculate vectors before and after dragging relative to this anchor
        vec_before = np.array([old_x - anchor_x, old_y - anchor_y])
        vec_after = np.array([new_x - anchor_x, new_y - anchor_y])
        
        # Calculate scale change
        len_before = np.linalg.norm(vec_before)
        len_after = np.linalg.norm(vec_after)
        scale_factor = len_after / len_before if len_before > 0 else 1.0
        
        # Limit scaling to prevent extreme changes
        scale_factor = np.clip(scale_factor, 0.8, 1.2)
        
        # Calculate rotation angle (in radians)
        if len_before > 1e-5 and len_after > 1e-5:
            dot_product = np.dot(vec_before, vec_after)
            det = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]
            angle = np.arctan2(det, dot_product)
        else:
            angle = 0.0
        
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
        initial_transform = translate_from_anchor @ scale_and_rotate @ translate_to_anchor
        
        # Apply initial transformation to 2D vertices
        verts_homogeneous = np.column_stack((state.verts2d, np.ones(len(state.verts2d))))
        transformed_verts = (initial_transform @ verts_homogeneous.T).T
        transformed_2d = transformed_verts[:, :2]
        
        # Now we need to apply additional corrections to keep ALL other fixed pins in place
        # This is crucial for the 3+ pin case
        
        # If we have more than one fixed pin, we need additional corrections
        if len(fixed_pins) > 1:
            # We'll use a thin plate spline-like approach
            # First, identify the original and target positions for the control points
            
            # Control points include all fixed pins (which should stay in place)
            # and the dragged pin (which should move to the new position)
            control_points_src = []  # Original positions
            control_points_dst = []  # Target positions
            
            # Add the dragged pin
            control_points_src.append([old_x, old_y])
            control_points_dst.append([new_x, new_y])
            
            # Add all fixed pins (they should stay in place)
            for _, fx, fy in fixed_pins:
                control_points_src.append([fx, fy])
                control_points_dst.append([fx, fy])
            
            # Convert to numpy arrays
            control_points_src = np.array(control_points_src)
            control_points_dst = np.array(control_points_dst)
            
            # Compute displacements between where pins would go under the initial 
            # transformation vs. where they should be
            pin_displacements = []
            
            for i, (_, fx, fy) in enumerate(fixed_pins):
                # Original position
                original_pos = np.array([fx, fy])
                
                # Position after initial transformation
                pin_homogeneous = np.array([fx, fy, 1])
                transformed_pos = (initial_transform @ pin_homogeneous)[:2]
                
                # Displacement needed to keep this pin fixed
                displacement = original_pos - transformed_pos
                pin_displacements.append((fx, fy, displacement))
            
            # Apply displacements to all vertices using inverse distance weighting
            # This ensures pins stay fixed while smoothly transitioning between them
            correction = np.zeros((len(state.verts2d), 2))
            
            for v_idx, v_pos in enumerate(transformed_2d):
                # Compute weights based on distance to each pin
                weights = []
                
                for px, py, _ in pin_displacements:
                    dist = np.sqrt((v_pos[0] - px)**2 + (v_pos[1] - py)**2)
                    # Avoid division by zero and create smoother falloff
                    weight = 1.0 / (dist + 1e-6)**2
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                
                # Apply weighted displacements
                for i, (_, _, disp) in enumerate(pin_displacements):
                    correction[v_idx] += weights[i] * disp
            
            # Apply corrections to get final vertex positions
            final_verts = transformed_2d + correction
            
            # Check that the transformed positions are reasonable
            img_w, img_h = state.img_w, state.img_h
            margin = max(img_w, img_h) * 0.8
            
            vertices_in_bounds = np.logical_and(
                np.logical_and(final_verts[:, 0] > -margin, final_verts[:, 0] < img_w + margin),
                np.logical_and(final_verts[:, 1] > -margin, final_verts[:, 1] < img_h + margin)
            )
            
            if np.mean(vertices_in_bounds) >= 0.6:
                # Update vertices with corrected positions
                state.verts2d = final_verts
            else:
                print("Transformation would cause vertices to go out of bounds")
                return
        else:
            # For 2-pin case, just use the initial transformation
            state.verts2d = transformed_2d
        
        # Update dragged pin position
        pin_data = pins[dragged_pin_idx]
        if len(pin_data) >= 5:  # 5-tuple format
            state.pins_per_image[state.current_image_idx][dragged_pin_idx] = (
                new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
            )
        else:  # 4-tuple format
            state.pins_per_image[state.current_image_idx][dragged_pin_idx] = (
                new_x, new_y, pin_data[2], pin_data[3]
            )
        
        # Update 3D vertices based on the 2D transformation
        update_3d_vertices(state)
        
        # Update landmarks and custom pins
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        from model.pins import update_custom_pins
        update_custom_pins(state)