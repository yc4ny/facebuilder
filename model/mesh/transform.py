"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Mesh transformation module.

This module provides functionality for rigid transformations of the 3D mesh
based on pin movements, including rotations, scaling, and translations.
It supports different transformation modes based on the number of active pins.
"""
import numpy as np
import cv2
from utils.geometry import calculate_front_facing
from model.mesh.deformation import update_3d_vertices
from model.landmarks import update_all_landmarks
from model.pins import update_custom_pins

def transform_mesh_rigid(state, dragged_pin_idx, old_x, old_y, new_x, new_y):
    """
    Transform the mesh rigidly based on pin movement.
    
    This function applies different transformation strategies based on the number of pins:
    - For 2 pins: Rigid 2D transformation (scale + rotation) that keeps the other pin fixed
    - For 3 pins: 3D rotation using PnP that keeps two pins fixed
    - For 4+ pins: Non-rigid deformation that keeps all non-dragged pins fixed
    
    Args:
        state (FaceBuilderState): Application state
        dragged_pin_idx (int): Index of the pin being dragged
        old_x (float): Previous x-coordinate of the dragged pin
        old_y (float): Previous y-coordinate of the dragged pin
        new_x (float): New x-coordinate of the dragged pin
        new_y (float): New y-coordinate of the dragged pin
        
    Returns:
        None
    """
    # Determine if we're dragging a landmark or custom pin
    is_dragging_landmark = dragged_pin_idx < len(state.landmark_positions)
    is_dragging_custom_pin = not is_dragging_landmark
    
    # Calculate total number of active pins
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    landmark_count = 0 if landmark_pins_hidden else len(state.landmark_positions)
    custom_pins = state.pins_per_image[state.current_image_idx]
    total_pin_count = landmark_count + len(custom_pins)
    
    # We need at least 2 pins (dragged + anchor) for rigid transformation
    if total_pin_count < 2:
        return
    
    # Collect all non-dragged pins to use as anchors/constraints
    fixed_pins = []
    
    # Add landmark pins (except dragged one)
    if not landmark_pins_hidden:
        for i, (lx, ly) in enumerate(state.landmark_positions):
            if i != dragged_pin_idx or not is_dragging_landmark:
                fixed_pins.append(('landmark', i, lx, ly))
    
    # Add custom pins (except dragged one)
    for i, pin in enumerate(custom_pins):
        # Calculate the global pin index (offset by landmarks)
        global_idx = i + len(state.landmark_positions)
        if global_idx != dragged_pin_idx or not is_dragging_custom_pin:
            fixed_pins.append(('custom', i, pin[0], pin[1]))
    
    # CASE 1: Exactly 3 pins - use 3D rotation with PnP (Perspective-n-Point)
    if total_pin_count == 3:
        # Try to use 3D rotation via PnP if we have camera parameters
        if _handle_three_pin_case(state, dragged_pin_idx, is_dragging_landmark, 
                                  is_dragging_custom_pin, fixed_pins, 
                                  old_x, old_y, new_x, new_y, custom_pins):
            return
    
    # CASE 2: 2 pins (or fallback from 3-pin case)
    if total_pin_count == 2 or (total_pin_count == 3 and state.verts2d is getattr(state, 'original_verts2d', None)):
        # Handle with 2D rigid transformation
        if _handle_two_pin_case(state, dragged_pin_idx, is_dragging_landmark, 
                               is_dragging_custom_pin, fixed_pins, 
                               old_x, old_y, new_x, new_y, custom_pins):
            return
    
    # CASE 3: 4+ pins - use non-rigid deformation that preserves fixed pins
    if total_pin_count >= 4:
        _handle_multi_pin_case(state, dragged_pin_idx, is_dragging_landmark,
                              is_dragging_custom_pin, fixed_pins,
                              old_x, old_y, new_x, new_y, custom_pins)


def _handle_three_pin_case(state, dragged_pin_idx, is_dragging_landmark, is_dragging_custom_pin, 
                          fixed_pins, old_x, old_y, new_x, new_y, custom_pins):
    """
    Handle the case with exactly 3 pins (1 dragged + 2 fixed).
    
    Uses 3D rotation with PnP algorithm to find optimal camera parameters that
    maintain the fixed pins' positions while moving the dragged pin.
    
    Args:
        state (FaceBuilderState): Application state
        dragged_pin_idx (int): Index of the pin being dragged
        is_dragging_landmark (bool): Whether we're dragging a landmark
        is_dragging_custom_pin (bool): Whether we're dragging a custom pin
        fixed_pins (list): List of fixed pins [(type, idx, x, y), ...]
        old_x, old_y (float): Previous position of dragged pin
        new_x, new_y (float): New position of dragged pin
        custom_pins (list): List of custom pins data
        
    Returns:
        bool: True if successful, False if falling back to 2D transformation
    """
    # Get camera parameters
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # We can only do 3D rotation if we have camera parameters
    if camera_matrix is None or rvec is None or tvec is None:
        return False
        
    # Save original values to revert if something goes wrong
    original_rvec = rvec.copy()
    original_tvec = tvec.copy()
    state.original_verts2d = state.verts2d.copy()
    
    try:
        # Get 3D position of dragged pin
        dragged_pin_3d_pos = None
        if is_dragging_landmark:
            if dragged_pin_idx < len(state.landmark3d):
                dragged_pin_3d_pos = state.landmark3d[dragged_pin_idx]
        else:  # Custom pin
            custom_pin_idx = dragged_pin_idx - len(state.landmark_positions)
            dragged_pin_data = custom_pins[custom_pin_idx]
            dragged_pin_3d_pos = dragged_pin_data[4] if len(dragged_pin_data) >= 5 else None
        
        # Get 3D positions of fixed pins
        fixed_pin1_type, fixed_pin1_idx, fixed_pin1_x, fixed_pin1_y = fixed_pins[0]
        fixed_pin1_3d_pos = None
        fixed_pin1_2d_pos = np.array([fixed_pin1_x, fixed_pin1_y])
        
        if fixed_pin1_type == 'landmark':
            if fixed_pin1_idx < len(state.landmark3d):
                fixed_pin1_3d_pos = state.landmark3d[fixed_pin1_idx]
        else:  # Custom pin
            fixed_pin1_data = custom_pins[fixed_pin1_idx]
            fixed_pin1_3d_pos = fixed_pin1_data[4] if len(fixed_pin1_data) >= 5 else None
        
        fixed_pin2_type, fixed_pin2_idx, fixed_pin2_x, fixed_pin2_y = fixed_pins[1]
        fixed_pin2_3d_pos = None
        fixed_pin2_2d_pos = np.array([fixed_pin2_x, fixed_pin2_y])
        
        if fixed_pin2_type == 'landmark':
            if fixed_pin2_idx < len(state.landmark3d):
                fixed_pin2_3d_pos = state.landmark3d[fixed_pin2_idx]
        else:  # Custom pin
            fixed_pin2_data = custom_pins[fixed_pin2_idx]
            fixed_pin2_3d_pos = fixed_pin2_data[4] if len(fixed_pin2_data) >= 5 else None
        
        # Only proceed if we have all 3D positions
        if dragged_pin_3d_pos is not None and fixed_pin1_3d_pos is not None and fixed_pin2_3d_pos is not None:
            # PnP (Perspective-n-Point) to solve for rotation and translation that best keeps both fixed pins in place and moves the dragged pin
            
            # Define the original 3D positions for all three pins
            obj_points = np.array([fixed_pin1_3d_pos, fixed_pin2_3d_pos, dragged_pin_3d_pos], dtype=np.float32)
            
            # Define the target 2D positions (fixed pins stay in place, dragged pin moves)
            img_points = np.array([fixed_pin1_2d_pos, fixed_pin2_2d_pos, [new_x, new_y]], dtype=np.float32)
            
            # Solve for rotation and translation using PnP algorithm:
            # The PnP problem finds [R|t] such that:
            # λᵢ [uᵢ, vᵢ, 1]ᵀ = K[R|t][Xᵢ, Yᵢ, Zᵢ, 1]ᵀ
            # where K is the camera matrix, [R|t] is the extrinsic matrix,
            # [Xᵢ, Yᵢ, Zᵢ] are 3D points, and [uᵢ, vᵢ] are 2D projections
            success, new_rvec, new_tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE, 
                useExtrinsicGuess=True, 
                rvec=rvec, 
                tvec=tvec
            )
            
            if success:
                # Refine the solution for better accuracy using Levenberg-Marquardt
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
                    np.array([fixed_pin1_3d_pos, fixed_pin2_3d_pos]), 
                    new_rvec, new_tvec, camera_matrix, np.zeros((4, 1))
                )
                fixed_pins_projected = fixed_pins_projected.reshape(-1, 2)
                
                # Calculate how much the fixed pins deviated from their original positions
                deviation1 = np.linalg.norm(fixed_pins_projected[0] - fixed_pin1_2d_pos)
                deviation2 = np.linalg.norm(fixed_pins_projected[1] - fixed_pin2_2d_pos)
                max_allowed_deviation = img_w * 0.01  # 1% of image width
                
                # Apply transformation if deviations are acceptable and most vertices are in view
                if (deviation1 <= max_allowed_deviation and 
                    deviation2 <= max_allowed_deviation and 
                    np.mean(vertices_in_bounds) >= 0.6):
                    
                    # Update rotations and translations
                    state.rotations[state.current_image_idx] = new_rvec
                    state.translations[state.current_image_idx] = new_tvec
                    
                    # Update 2D vertex positions
                    state.verts2d = projected_2d
                    
                    # Update dragged pin position
                    _update_dragged_pin_position(state, dragged_pin_idx, is_dragging_landmark, 
                                               is_dragging_custom_pin, new_x, new_y, custom_pins)
                    
                    # Skip 3D-2D projection cycle since we've directly calculated 2D positions
                    state.skip_projection = True
                    
                    # Update backface culling
                    state.front_facing = calculate_front_facing(
                        state.verts3d, state.faces,
                        camera_matrix=camera_matrix, rvec=new_rvec, tvec=new_tvec
                    )
                    
                    update_all_landmarks(state)
                    update_custom_pins(state)
                    return True
                else:
                    # Log diagnostics and fall back to 2D transformation
                    print(f"PnP solution deviations: {deviation1:.2f}, {deviation2:.2f} pixels, "
                          f"vertices in bounds: {np.mean(vertices_in_bounds):.2f}")
        
    except Exception as e:
        # Revert to original values if there's an error
        state.rotations[state.current_image_idx] = original_rvec
        state.translations[state.current_image_idx] = original_tvec
        print(f"Error during 3-pin 3D rotation: {e}")
    
    return False  # Fall through to 2D transformation as backup


def _handle_two_pin_case(state, dragged_pin_idx, is_dragging_landmark, is_dragging_custom_pin, 
                        fixed_pins, old_x, old_y, new_x, new_y, custom_pins):
    """
    Handle the case with exactly 2 pins (1 dragged + 1 fixed).
    
    Applies a 2D rigid transformation (scale + rotation) that keeps the fixed pin in place
    and moves the dragged pin to the new position.
    
    Args:
        state (FaceBuilderState): Application state
        dragged_pin_idx (int): Index of the pin being dragged
        is_dragging_landmark (bool): Whether we're dragging a landmark
        is_dragging_custom_pin (bool): Whether we're dragging a custom pin
        fixed_pins (list): List of fixed pins [(type, idx, x, y), ...]
        old_x, old_y (float): Previous position of dragged pin
        new_x, new_y (float): New position of dragged pin
        custom_pins (list): List of custom pins data
        
    Returns:
        bool: True if successful
    """
    # We'll use the first fixed pin as our anchor
    anchor_type, anchor_idx, anchor_x, anchor_y = fixed_pins[0]
    
    # Calculate vectors before and after dragging relative to this anchor
    vec_before = np.array([old_x - anchor_x, old_y - anchor_y])
    vec_after = np.array([new_x - anchor_x, new_y - anchor_y])
    
    # Calculate scale change
    # Scale factor = |vec_after| / |vec_before|
    len_before = np.linalg.norm(vec_before)
    len_after = np.linalg.norm(vec_after)
    scale_factor = len_after / len_before if len_before > 0 else 1.0
    
    # Limit scaling to prevent extreme changes
    scale_factor = np.clip(scale_factor, 0.8, 1.2)
    
    # Calculate rotation angle (in radians)
    # cos(θ) = (vec_before · vec_after) / (|vec_before| * |vec_after|)
    # sin(θ) = (vec_before × vec_after) / (|vec_before| * |vec_after|)
    if len_before > 1e-5 and len_after > 1e-5:
        dot_product = np.dot(vec_before, vec_after)
        # Cross product in 2D gives a scalar (z-component of the 3D cross product)
        det = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]
        angle = np.arctan2(det, dot_product)
    else:
        angle = 0.0
    
    # Compute transformation matrix components
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Create a composite 2D transformation matrix that:
    # 1. Translates to the anchor point
    # 2. Applies scaling and rotation
    # 3. Translates back from the anchor point
    
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
    
    # Combined transformation matrix: T = T₂ × R × T₁
    initial_transform = translate_from_anchor @ scale_and_rotate @ translate_to_anchor
    
    # Apply transformation to 2D vertices using homogeneous coordinates
    # X' = T × X where X = [x, y, 1]ᵀ
    verts_homogeneous = np.column_stack((state.verts2d, np.ones(len(state.verts2d))))
    transformed_verts = (initial_transform @ verts_homogeneous.T).T
    state.verts2d = transformed_verts[:, :2]
    
    # Update dragged pin position
    _update_dragged_pin_position(state, dragged_pin_idx, is_dragging_landmark, 
                               is_dragging_custom_pin, new_x, new_y, custom_pins)
    
    # Set the flag to skip 3D-2D projection cycle
    state.skip_projection = True
    
    # Get current camera parameters for backface culling
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Update 3D vertices based on the 2D transformation
    update_3d_vertices(state)
    
    # After 3D update, calculate which faces are front-facing
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Update landmarks and custom pins
    update_all_landmarks(state)
    update_custom_pins(state)
    return True


def _handle_multi_pin_case(state, dragged_pin_idx, is_dragging_landmark, is_dragging_custom_pin, 
                          fixed_pins, old_x, old_y, new_x, new_y, custom_pins):
    """
    Handle the case with 4 or more pins using non-rigid deformation.
    
    Uses inverse distance weighting to deform the mesh, ensuring all fixed pins
    stay in place while the dragged pin moves to the new position.
    
    Args:
        state (FaceBuilderState): Application state
        dragged_pin_idx (int): Index of the pin being dragged
        is_dragging_landmark (bool): Whether we're dragging a landmark
        is_dragging_custom_pin (bool): Whether we're dragging a custom pin
        fixed_pins (list): List of fixed pins [(type, idx, x, y), ...]
        old_x, old_y (float): Previous position of dragged pin
        new_x, new_y (float): New position of dragged pin
        custom_pins (list): List of custom pins data
        
    Returns:
        None
    """
    # Thin plate spline-like approach with inverse distance weighting
    # First, identify the original and target positions for the control points
    
    # Control points include all fixed pins (which should stay in place)
    # and the dragged pin (which should move to the new position)
    control_points_src = []  # Original positions
    control_points_dst = []  # Target positions
    
    # Add the dragged pin
    control_points_src.append([old_x, old_y])
    control_points_dst.append([new_x, new_y])
    
    # Add all fixed pins (they should stay in place)
    for _, _, fx, fy in fixed_pins:
        control_points_src.append([fx, fy])
        control_points_dst.append([fx, fy])
    
    # Convert to numpy arrays
    control_points_src = np.array(control_points_src)
    control_points_dst = np.array(control_points_dst)
    
    # Compute displacements between control points
    # δᵢ = destination_pointᵢ - source_pointᵢ
    displacements = control_points_dst - control_points_src
    
    # Apply displacements to all vertices using inverse distance weighting
    new_verts2d = state.verts2d.copy()
    
    # For each vertex, compute a weighted average of the displacements
    for v_idx, v_pos in enumerate(state.verts2d):
        # Compute weights based on distance to each control point
        # wᵢ = 1 / (|v - cᵢ|² + ε)  where ε is a small constant to avoid division by zero
        weights = []
        
        for cp_pos in control_points_src:
            dist = np.linalg.norm(v_pos - cp_pos)
            # Inverse squared distance weighting with small epsilon to avoid division by zero
            weight = 1.0 / (dist + 1e-6)**2
            weights.append(weight)
        
        # Normalize weights so they sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Apply weighted displacements
        # δv = Σᵢ wᵢ * δᵢ
        total_displacement = np.zeros(2)
        for i, disp in enumerate(displacements):
            total_displacement += weights[i] * disp
        
        # Update vertex position: v' = v + δv
        new_verts2d[v_idx] = v_pos + total_displacement
    
    # Check that the transformed positions are reasonable (not too far outside viewport)
    img_w, img_h = state.img_w, state.img_h
    margin = max(img_w, img_h) * 0.8
    
    vertices_in_bounds = np.logical_and(
        np.logical_and(new_verts2d[:, 0] > -margin, new_verts2d[:, 0] < img_w + margin),
        np.logical_and(new_verts2d[:, 1] > -margin, new_verts2d[:, 1] < img_h + margin)
    )
    
    # Apply transformation if most vertices remain in view
    if np.mean(vertices_in_bounds) >= 0.6:
        # Update vertices with new positions
        state.verts2d = new_verts2d
        
        # Update dragged pin position
        _update_dragged_pin_position(state, dragged_pin_idx, is_dragging_landmark, 
                                   is_dragging_custom_pin, new_x, new_y, custom_pins)
        
        # Update 3D vertices based on the 2D transformation
        update_3d_vertices(state)
        
        # Update backface culling
        camera_matrix = state.camera_matrices[state.current_image_idx]
        rvec = state.rotations[state.current_image_idx]
        tvec = state.translations[state.current_image_idx]
        
        state.front_facing = calculate_front_facing(
            state.verts3d, state.faces,
            camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
        )

        update_all_landmarks(state)
        update_custom_pins(state)
    else:
        print("Transformation would cause vertices to go out of bounds")


def _update_dragged_pin_position(state, dragged_pin_idx, is_dragging_landmark, 
                               is_dragging_custom_pin, new_x, new_y, custom_pins):
    """
    Update the position of the dragged pin.
    
    Args:
        state (FaceBuilderState): Application state
        dragged_pin_idx (int): Index of the pin being dragged
        is_dragging_landmark (bool): Whether we're dragging a landmark
        is_dragging_custom_pin (bool): Whether we're dragging a custom pin
        new_x, new_y (float): New position of dragged pin
        custom_pins (list): List of custom pins data
        
    Returns:
        None
    """
    if is_dragging_landmark:
        # Update landmark position directly
        state.landmark_positions[dragged_pin_idx] = (new_x, new_y)
    else:  # Custom pin
        custom_pin_idx = dragged_pin_idx - len(state.landmark_positions)
        pin_data = custom_pins[custom_pin_idx]
        
        # Handle both 5-tuple (with 3D position) and 4-tuple formats
        if len(pin_data) >= 5:  # 5-tuple format
            state.pins_per_image[state.current_image_idx][custom_pin_idx] = (
                new_x, new_y, pin_data[2], pin_data[3], pin_data[4]
            )
        else:  # 4-tuple format (legacy)
            state.pins_per_image[state.current_image_idx][custom_pin_idx] = (
                new_x, new_y, pin_data[2], pin_data[3]
            )