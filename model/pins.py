"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Pin management module.

This module handles all pin-related operations, including adding pins,
updating their positions, removing pins, and synchronizing them across views.
Pins are points attached to the mesh that allow users to interact with it.
"""
import numpy as np
import cv2
from model.mesh import project_current_3d_to_2d
from utils.geometry import calculate_front_facing

def add_custom_pin(x, y, state):
    """
    Add a new custom pin at the specified 2D coordinates.
    
    This function finds the closest front-facing triangle to the pin
    position, calculates its barycentric coordinates and 3D position,
    and adds it to the current image's pin list.
    
    Args:
        x (float): X-coordinate in the 2D view
        y (float): Y-coordinate in the 2D view
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
    # Find the closest face to the pin position
    closest_face_idx = -1
    min_dist = float('inf')
    pin_pos = np.array([x, y])
    
    # Use front-facing information if available, otherwise assume all faces are front-facing
    front_facing = state.front_facing if state.front_facing is not None else np.ones(len(state.faces), dtype=bool)
    
    # Iterate through all faces to find the one containing the pin
    for face_idx, (i0, i1, i2) in enumerate(state.faces):
        # Skip back-facing faces
        if not front_facing[face_idx]:
            continue
            
        # Get the 2D positions of the triangle's vertices
        v0 = state.verts2d[i0]
        v1 = state.verts2d[i1]
        v2 = state.verts2d[i2]
        
        # Calculate triangle area using the cross product method:
        # A = 0.5 * ||(v1-v0) × (v2-v0)||
        area = 0.5 * abs((v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1])))
        
        if area > 0:
            # Calculate barycentric coordinates (a, b, c) for the point
            # These coordinates express the point as a weighted sum of the triangle vertices:
            # P = a*v0 + b*v1 + c*v2, where a + b + c = 1
            a = 0.5 * abs((v1[0]*(v2[1]-pin_pos[1]) + v2[0]*(pin_pos[1]-v1[1]) + pin_pos[0]*(v1[1]-v2[1]))) / area
            b = 0.5 * abs((v0[0]*(pin_pos[1]-v2[1]) + pin_pos[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-pin_pos[1]))) / area
            c = 1 - a - b
            
            # Check if point is inside triangle (with small tolerance for numerical stability)
            if -0.01 <= a <= 1.01 and -0.01 <= b <= 1.01 and -0.01 <= c <= 1.01:
                # Calculate distance to center of triangle to find the closest one
                center = (v0 + v1 + v2) / 3
                dist = np.sum((center - pin_pos)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_face_idx = face_idx
                    bc_coords = np.array([a, b, c])
    
    # If we found a valid face for the pin
    if closest_face_idx != -1:
        # Calculate the 3D position using barycentric coordinates
        i0, i1, i2 = state.faces[closest_face_idx]
        v0_3d = state.verts3d[i0]
        v1_3d = state.verts3d[i1]
        v2_3d = state.verts3d[i2]
        pin_pos_3d = bc_coords[0]*v0_3d + bc_coords[1]*v1_3d + bc_coords[2]*v2_3d
        
        # Add pin data as a 5-tuple: (x, y, face_idx, barycentric_coords, 3d_position)
        state.pins_per_image[state.current_image_idx].append(
            (x, y, closest_face_idx, bc_coords, pin_pos_3d)
        )
        
        print(f"Added pin at ({x:.2f}, {y:.2f}) to face {closest_face_idx}")
    else:
        print("Could not find a valid face for the pin")


def update_custom_pins(state):
    """
    Update positions of all custom pins and landmark pins after mesh movement.
    
    This function projects the 3D positions of pins to 2D using the current
    camera parameters, ensuring pins stay attached to their respective faces.
    It also preserves the positions of any currently dragged pins.
    
    Args:
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
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
        
        # Initialize default rotation and translation
        R = np.eye(3, dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)
        distance = focal_length * 1.5
        tvec = np.array([[0, 0, distance]], dtype=np.float32).T
        
        # Save camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Identify dragged pins to preserve their positions
    is_dragging = state.drag_index != -1
    is_dragging_landmark = is_dragging and state.drag_index < len(state.landmark_positions)
    is_dragging_custom_pin = is_dragging and state.drag_index >= len(state.landmark_positions)
    dragged_landmark_idx = state.drag_index if is_dragging_landmark else -1
    dragged_pin_idx = state.drag_index - len(state.landmark_positions) if is_dragging_custom_pin else -1
    
    # Update custom pins
    updated_pins = []
    for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
        # Preserve the position of any dragged pin
        if i == dragged_pin_idx:
            updated_pins.append(pin_data)
            continue
        
        # Extract face index and barycentric coordinates
        face_idx = pin_data[2]
        bc = pin_data[3]
        
        # Get 3D coordinates of the triangle vertices
        i0, i1, i2 = state.faces[face_idx]
        v0_3d, v1_3d, v2_3d = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        
        # Calculate updated 3D position using barycentric coordinates:
        # P = a*v0 + b*v1 + c*v2, where [a,b,c] are barycentric coordinates
        pin_pos_3d = bc[0]*v0_3d + bc[1]*v1_3d + bc[2]*v2_3d
        
        # Project 3D position to 2D using perspective projection
        try:
            # Use camera projection: x' = P*X, where P is the projection matrix
            # and X is the homogeneous 3D point
            projected_pin, _ = cv2.projectPoints(
                np.array([pin_pos_3d], dtype=np.float32),
                rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            new_pos_2d = projected_pin.reshape(-1, 2)[0]
        except cv2.error:
            # Fallback: use barycentric interpolation in 2D if projection fails
            v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
            new_pos_2d = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        
        # Store updated pin with 5-tuple format: (x, y, face_idx, barycentric_coords, 3d_position)
        updated_pins.append((new_pos_2d[0], new_pos_2d[1], face_idx, bc, pin_pos_3d))
    
    # Update the pins for the current image
    state.pins_per_image[state.current_image_idx] = updated_pins
    
    # Update landmark positions (if they're not being dragged)
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    if not landmark_pins_hidden:
        # Update 3D landmark positions
        for i, (f_idx, bc) in enumerate(zip(state.face_for_lmk, state.lmk_b_coords)):
            # Skip if this landmark is being dragged
            if i == dragged_landmark_idx:
                continue
                
            # Update 3D position using barycentric coordinates
            i0, i1, i2 = state.faces[f_idx]
            v0, v1, v2 = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
            state.landmark3d[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
            
            # Project to 2D
            try:
                projected_lmk, _ = cv2.projectPoints(
                    np.array([state.landmark3d[i]], dtype=np.float32),
                    rvec, tvec, camera_matrix, np.zeros((4, 1))
                )
                state.landmark_positions[i] = projected_lmk.reshape(-1, 2)[0]
            except cv2.error:
                # Fallback to 2D interpolation if projection fails
                v0_2d, v1_2d, v2_2d = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
                state.landmark_positions[i] = bc[0]*v0_2d + bc[1]*v1_2d + bc[2]*v2_2d


def synchronize_pins_across_views(state):
    """
    Ensure pins are consistent across all views by updating 2D positions
    based on the shared 3D model.
    
    This function temporarily switches to each view, projects the 3D model 
    to 2D for that view, updates the pins, and then restores the original view.
    
    Args:
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
    # Store current view information to restore later
    current_view = state.current_image_idx
    current_img = state.overlay.copy()
    current_img_h, current_img_w = state.img_h, state.img_w
    
    # Remember landmark visibility state
    was_landmarks_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    
    # Process each view
    for view_idx in range(len(state.images)):
        if view_idx == current_view:
            continue  # Skip current view as it's already up to date
        
        # Temporarily switch to this view
        state.current_image_idx = view_idx
        state.overlay = state.images[view_idx].copy()
        state.img_h, state.img_w = state.overlay.shape[:2]
        
        # Project the current 3D model to 2D for this view
        project_current_3d_to_2d(state)
        
        # Update pins for this view
        update_custom_pins(state)
    
    # Restore current view
    state.current_image_idx = current_view
    state.overlay = current_img
    state.img_h, state.img_w = current_img_h, current_img_w
    
    # Project again to make sure we're consistent
    project_current_3d_to_2d(state)
    
    # Restore landmark visibility state
    if not was_landmarks_hidden:
        # Make sure landmarks are visible
        if hasattr(state, 'landmark_pins_hidden'):
            state.landmark_pins_hidden = False
            
        # Update all landmarks to ensure they're positioned correctly
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)


def remove_pins(state):
    """
    Remove pins from the current image.
    
    If custom pins exist, they are removed first. If no custom pins exist,
    landmark pins are hidden instead.
    
    Args:
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
    # First check if there are any custom pins
    if len(state.pins_per_image[state.current_image_idx]) > 0:
        # Remove all custom pins
        state.pins_per_image[state.current_image_idx] = []
        
        # Reset the pins_moved flag to restore green color for new pins
        if hasattr(state, 'pins_moved'):
            state.pins_moved = False
            
        print("Removed all custom pins from current image")
    else:
        # If no custom pins exist, hide landmark pins
        state.landmark_pins_hidden = True
        
        # Clear alignment cache for this image to force re-alignment next time
        try:
            from model.landmarks import clear_image_alignment
            clear_image_alignment(state)
        except (ImportError, NameError) as e:
            print(f"Note: Could not clear alignment cache: {e}")
            print("You may need to restart the application to restore landmarks")
        
        print("Landmark pins hidden")


def center_geo(state):
    """
    Reset the mesh to its default position and set up perspective projection.
    
    This resets the 3D vertices to their default positions, calculates
    appropriate camera parameters, and updates all 2D projections.
    
    Args:
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
    # Reset 3D vertices to default (original model)
    state.verts3d = state.verts3d_default.copy()
    
    # Reset the pins_moved flag to restore green color for new pins
    if hasattr(state, 'pins_moved'):
        state.pins_moved = False
    
    # Set up perspective projection with proper camera parameters
    # Camera intrinsic matrix:
    # K = [fx  0  cx]
    #     [0  fy  cy]
    #     [0   0   1]
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Calculate center of the model
    center = np.mean(state.verts3d, axis=0)
    
    # Initialize rotation (identity matrix)
    R = np.eye(3, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    
    # Position the model in front of the camera
    distance = -0.5 
    # Translation vector: t = -R*C where C is the camera center
    tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
    
    # Store camera parameters for the current image
    state.camera_matrices[state.current_image_idx] = camera_matrix
    state.rotations[state.current_image_idx] = rvec
    state.translations[state.current_image_idx] = tvec
    
    # Project 3D to 2D using perspective projection
    if 'project_2d' in state.callbacks:
        state.callbacks['project_2d'](state)
    else:
        project_current_3d_to_2d(state)
    
    # Update landmarks
    from model.landmarks import update_all_landmarks
    update_all_landmarks(state)
    
    # Make landmarks visible again if they were hidden
    if hasattr(state, 'landmark_pins_hidden'):
        state.landmark_pins_hidden = False
        print("Landmarks made visible again")
    
    # Update custom pins
    update_custom_pins(state)
    
    # Calculate which faces are front-facing
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    print("Reset mesh to default position with perspective projection")


def reset_shape(state):
    """
    Reset only the shape parameters while preserving position and orientation.
    
    This resets the 3D vertices to their default positions, but then scales
    and translates them to maintain the current pose and size.
    
    Args:
        state (FaceBuilderState): Application state
        
    Returns:
        None
    """
    # Reset the pins_moved flag to restore green color for new pins
    if hasattr(state, 'pins_moved'):
        state.pins_moved = False
    
    # Store the current transformation information
    current_center_3d = np.mean(state.verts3d, axis=0)
    current_scale_3d = np.mean(np.linalg.norm(state.verts3d - current_center_3d, axis=1))
    
    # Reset to default shape in 3D space
    state.verts3d = state.verts3d_default.copy()
    
    # Calculate default shape parameters for scaling
    default_center_3d = np.mean(state.verts3d, axis=0)
    default_scale_3d = np.mean(np.linalg.norm(state.verts3d - default_center_3d, axis=1))
    
    # Scale factor to maintain current size
    scale_factor = current_scale_3d / default_scale_3d if default_scale_3d > 0 else 1.0
    
    # Apply scaling and translation:
    # X_new = (X_default - center_default) * scale + center_current
    state.verts3d = (state.verts3d - default_center_3d) * scale_factor + current_center_3d
    
    # Ensure we have camera parameters for this view
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
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
        distance = focal_length * 1.5
        tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
        
        # Store camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Project 3D mesh to 2D using perspective projection
    if 'project_2d' in state.callbacks:
        state.callbacks['project_2d'](state)
    else:
        project_current_3d_to_2d(state)
    
    # Update landmarks
    from model.landmarks import update_all_landmarks
    update_all_landmarks(state)
    
    # Make landmarks visible again if they were hidden
    if hasattr(state, 'landmark_pins_hidden'):
        state.landmark_pins_hidden = False
        print("Landmarks made visible again")
    
    # Update custom pins
    update_custom_pins(state)
    
    # Calculate which faces are front-facing
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Synchronize pins across all views to maintain consistency
    synchronize_pins_across_views(state)
    
    print("Reset mesh shape to default while preserving position and orientation")