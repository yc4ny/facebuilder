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
    
    # Check pins in the current image
    pins = state.pins_per_image[state.current_image_idx]
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
                
                # Get target position
                new_x = old_lx + dx
                new_y = old_ly + dy
                
                # MULTIPLE PINS CASE (2 or more pins)
                if has_multiple_pins:
                    # Handle with transform_mesh_rigid which now supports 2-pin, 3-pin, and 4+ pin cases
                    try:
                        transform_mesh_rigid(state, pin_idx, old_lx, old_ly, new_x, new_y)
                        
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
                            mouse_pos = np.array([new_x, new_y])
                            
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
                                
                                # CRITICAL: Directly update the dragged pin position to follow mouse
                                pin_data = pins[pin_idx]
                                if len(pin_data) >= 5:  # 5-tuple format
                                    state.pins_per_image[state.current_image_idx][pin_idx] = (
                                        new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
                                    )
                                else:  # 4-tuple format
                                    state.pins_per_image[state.current_image_idx][pin_idx] = (
                                        new_x, new_y, pin_data[2], pin_data[3]
                                    )
                                
                                # Update other pins (the dragged pin will be preserved by update_custom_pins)
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
            
            # Get new position for direct pin update
            new_x = old_lx + dx
            new_y = old_ly + dy
    else:
        combined_w = radial_w
    
    # Store original 2D positions before modification
    original_verts2d = state.verts2d.copy()
    
    # Apply the calculated weights to move vertices in 2D
    shift = np.column_stack((combined_w * dx, combined_w * dy))
    state.verts2d += shift
    
    # If we're dragging a pin, update its position directly
    if is_dragging_pin:
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(state.pins_per_image[state.current_image_idx]):
            pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
            new_x = old_lx + dx
            new_y = old_ly + dy
            
            if len(pin_data) >= 5:  # 5-tuple format
                state.pins_per_image[state.current_image_idx][pin_idx] = (
                    new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
                )
            else:  # 4-tuple format
                state.pins_per_image[state.current_image_idx][pin_idx] = (
                    new_x, new_y, pin_data[2], pin_data[3]
                )
    
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
    - For 2 pins: Uses rigid 2D transformation that keeps the other pin fixed
    - For 3 pins: Uses 3D rotation that keeps two pins fixed
    - For 4+ pins: Uses deformation that keeps all non-dragged pins fixed
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
    
    # Special case for exactly 3 pins total (1 dragged + 2 fixed)
    if len(pins) == 3:
        # Get camera parameters
        camera_matrix = state.camera_matrices[state.current_image_idx]
        rvec = state.rotations[state.current_image_idx]
        tvec = state.translations[state.current_image_idx]
        
        # We can only do 3D rotation if we have camera parameters
        if camera_matrix is not None and rvec is not None and tvec is not None:
            # Save original values to revert if something goes wrong
            original_rvec = rvec.copy()
            original_tvec = tvec.copy()
            original_verts2d = state.verts2d.copy()
            
            try:
                # Get 3D positions of all pins
                dragged_pin_data = pins[dragged_pin_idx]
                dragged_pin_3d_pos = dragged_pin_data[4] if len(dragged_pin_data) >= 5 else None
                
                # Get both fixed pins' data
                fixed_pin1_data = pins[fixed_pins[0][0]]
                fixed_pin1_3d_pos = fixed_pin1_data[4] if len(fixed_pin1_data) >= 5 else None
                fixed_pin1_2d_pos = np.array([fixed_pin1_data[0], fixed_pin1_data[1]])
                
                fixed_pin2_data = pins[fixed_pins[1][0]]
                fixed_pin2_3d_pos = fixed_pin2_data[4] if len(fixed_pin2_data) >= 5 else None
                fixed_pin2_2d_pos = np.array([fixed_pin2_data[0], fixed_pin2_data[1]])
                
                # Only proceed if we have all 3D positions
                if dragged_pin_3d_pos is not None and fixed_pin1_3d_pos is not None and fixed_pin2_3d_pos is not None:
                    # We'll use a PnP (Perspective-n-Point) approach to solve for rotation and translation
                    # that best keeps both fixed pins in place and moves the dragged pin to the new position
                    
                    # First, define the original 3D positions for all three pins
                    obj_points = np.array([fixed_pin1_3d_pos, fixed_pin2_3d_pos, dragged_pin_3d_pos], dtype=np.float32)
                    
                    # Define the target 2D positions (fixed pins stay in place, dragged pin moves to new position)
                    img_points = np.array([fixed_pin1_2d_pos, fixed_pin2_2d_pos, [new_x, new_y]], dtype=np.float32)
                    
                    # Solve for the rotation and translation
                    success, new_rvec, new_tvec = cv2.solvePnP(
                        obj_points, img_points, camera_matrix, np.zeros((4, 1)),
                        flags=cv2.SOLVEPNP_ITERATIVE, 
                        useExtrinsicGuess=True, 
                        rvec=rvec, 
                        tvec=tvec
                    )
                    
                    if success:
                        # Refine the solution for better accuracy
                        new_rvec, new_tvec = cv2.solvePnPRefineLM(
                            obj_points, img_points, camera_matrix, np.zeros((4, 1)),
                            new_rvec, new_tvec,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 1e-6)
                        )
                        
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
                        
                        # Verify that the solution keeps fixed pins in place
                        fixed_pins_projected, _ = cv2.projectPoints(
                            np.array([fixed_pin1_3d_pos, fixed_pin2_3d_pos]), new_rvec, new_tvec, camera_matrix, np.zeros((4, 1))
                        )
                        fixed_pins_projected = fixed_pins_projected.reshape(-1, 2)
                        
                        deviation1 = np.linalg.norm(fixed_pins_projected[0] - fixed_pin1_2d_pos)
                        deviation2 = np.linalg.norm(fixed_pins_projected[1] - fixed_pin2_2d_pos)
                        max_allowed_deviation = img_w * 0.01  # 1% of image width
                        
                        if deviation1 <= max_allowed_deviation and deviation2 <= max_allowed_deviation and np.mean(vertices_in_bounds) >= 0.6:
                            # Update rotations and translations
                            state.rotations[state.current_image_idx] = new_rvec
                            state.translations[state.current_image_idx] = new_tvec
                            
                            # Update 2D vertex positions
                            state.verts2d = projected_2d
                            
                            # Update dragged pin position
                            pin_data = pins[dragged_pin_idx]
                            if len(pin_data) >= 5:
                                state.pins_per_image[state.current_image_idx][dragged_pin_idx] = (
                                    new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
                                )
                            else:
                                state.pins_per_image[state.current_image_idx][dragged_pin_idx] = (
                                    new_x, new_y, pin_data[2], pin_data[3]
                                )
                            
                            # Set the flag to skip 3D-2D projection cycle
                            state.skip_projection = True
                            
                            # Update landmarks and custom pins
                            from model.landmarks import update_all_landmarks
                            update_all_landmarks(state)
                            from model.pins import update_custom_pins
                            update_custom_pins(state)
                            return
                        else:
                            print(f"PnP solution deviations: {deviation1:.2f}, {deviation2:.2f} pixels, vertices in bounds: {np.mean(vertices_in_bounds):.2f}")
                            # Fall through to 2D transformation as backup
                    else:
                        print("PnP solver failed, falling back to 2D transformation")
                        # Fall through to 2D transformation
                else:
                    print("Missing 3D positions for pins, falling back to 2D transformation")
                    # Fall through to 2D transformation
            
            except Exception as e:
                # Revert to original values if there's an error
                state.rotations[state.current_image_idx] = original_rvec
                state.translations[state.current_image_idx] = original_tvec
                state.verts2d = original_verts2d
                print(f"Error during 3-pin 3D rotation: {e}")
                # Fall through to 2D transformation as backup
    
    # Regular 2D transformation for 2-pin case or fallback for 3-pin case
    if len(pins) == 2 or (len(pins) == 3 and state.verts2d is original_verts2d):
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
        state.verts2d = transformed_verts[:, :2]
        
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
        
        # Set the flag to skip 3D-2D projection cycle
        state.skip_projection = True
        
        # Update 3D vertices based on the 2D transformation
        update_3d_vertices(state)
        
        # Update landmarks and custom pins
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        from model.pins import update_custom_pins
        update_custom_pins(state)
        return
    
    # Case for 4+ pins: Use deformation that keeps all fixed pins in place
    if len(pins) >= 4:
        # We'll use a thin plate spline-like approach with inverse distance weighting
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
        
        # Compute displacements between control points
        displacements = control_points_dst - control_points_src
        
        # Apply displacements to all vertices using inverse distance weighting
        new_verts2d = state.verts2d.copy()
        
        for v_idx, v_pos in enumerate(state.verts2d):
            # Compute weights based on distance to each control point
            weights = []
            
            for cp_pos in control_points_src:
                dist = np.linalg.norm(v_pos - cp_pos)
                # Avoid division by zero and create smoother falloff
                weight = 1.0 / (dist + 1e-6)**2
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Apply weighted displacements
            total_displacement = np.zeros(2)
            for i, disp in enumerate(displacements):
                total_displacement += weights[i] * disp
            
            # Update vertex position
            new_verts2d[v_idx] = v_pos + total_displacement
        
        # Check that the transformed positions are reasonable
        img_w, img_h = state.img_w, state.img_h
        margin = max(img_w, img_h) * 0.8
        
        vertices_in_bounds = np.logical_and(
            np.logical_and(new_verts2d[:, 0] > -margin, new_verts2d[:, 0] < img_w + margin),
            np.logical_and(new_verts2d[:, 1] > -margin, new_verts2d[:, 1] < img_h + margin)
        )
        
        if np.mean(vertices_in_bounds) >= 0.6:
            # Update vertices with new positions
            state.verts2d = new_verts2d
            
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
        else:
            print("Transformation would cause vertices to go out of bounds")
            return