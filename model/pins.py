# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import pygame
import cv2 
from model.mesh import project_current_3d_to_2d


def add_custom_pin(x, y, state):
    """Add a new custom pin at the given coordinates"""
    # Find the closest face to the pin position
    closest_face_idx = -1
    min_dist = float('inf')
    pin_pos = np.array([x, y])
    
    # Check if we have determined which faces are front-facing
    # If not, default to all faces being front-facing
    front_facing = state.front_facing if state.front_facing is not None else np.ones(len(state.faces), dtype=bool)
    
    for face_idx, (i0, i1, i2) in enumerate(state.faces):
        # Skip back-facing faces
        if not front_facing[face_idx]:
            continue
            
        v0 = state.verts2d[i0]
        v1 = state.verts2d[i1]
        v2 = state.verts2d[i2]
        
        # Calculate if pin is inside the triangle
        area = 0.5 * abs((v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1])))
        if area > 0:
            # Calculate barycentric coordinates
            a = 0.5 * abs((v1[0]*(v2[1]-pin_pos[1]) + v2[0]*(pin_pos[1]-v1[1]) + pin_pos[0]*(v1[1]-v2[1]))) / area
            b = 0.5 * abs((v0[0]*(pin_pos[1]-v2[1]) + pin_pos[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-pin_pos[1]))) / area
            c = 1 - a - b
            
            # Check if point is inside triangle (with some tolerance)
            if -0.01 <= a <= 1.01 and -0.01 <= b <= 1.01 and -0.01 <= c <= 1.01:
                # Calculate distance to center of triangle
                center = (v0 + v1 + v2) / 3
                dist = np.sum((center - pin_pos)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_face_idx = face_idx
                    bc_coords = np.array([a, b, c])
    
    if closest_face_idx != -1:
        # Also calculate the 3D position of the pin using barycentric coordinates
        i0, i1, i2 = state.faces[closest_face_idx]
        v0_3d = state.verts3d[i0]
        v1_3d = state.verts3d[i1]
        v2_3d = state.verts3d[i2]
        pin_pos_3d = bc_coords[0]*v0_3d + bc_coords[1]*v1_3d + bc_coords[2]*v2_3d
        
        # Add pin data to the current image's pin list
        state.pins_per_image[state.current_image_idx].append((x, y, closest_face_idx, bc_coords, pin_pos_3d))
        
        print(f"Added pin at ({x}, {y}) to face {closest_face_idx}")
    else:
        print("Could not find a face for the pin")

def update_custom_pins(state):
    """Update positions of custom pins after mesh movement"""
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
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.array([[0, 0, focal_length]], dtype=np.float32).T
        
        # Save camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Check if a pin is being dragged
    is_dragging_pin = (state.drag_index != -1 and 
                      state.drag_index >= len(state.landmark_positions))
    dragged_pin_idx = state.drag_index - len(state.landmark_positions) if is_dragging_pin else -1
    
    updated_pins = []
    for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
        # Check if this pin is being dragged
        if i == dragged_pin_idx:
            # Preserve the position of the dragged pin
            updated_pins.append(pin_data)
            continue
        
        # Support both 4-tuple and 5-tuple pin formats for backward compatibility
        face_idx = pin_data[2]
        bc = pin_data[3]
        
        # Get 3D coordinates of the triangle vertices
        i0, i1, i2 = state.faces[face_idx]
        v0_3d, v1_3d, v2_3d = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        
        # Calculate 3D position using barycentric coordinates
        pin_pos_3d = bc[0]*v0_3d + bc[1]*v1_3d + bc[2]*v2_3d
        
        # Project 3D position to 2D using perspective projection
        try:
            projected_pin, _ = cv2.projectPoints(
                np.array([pin_pos_3d], dtype=np.float32),
                rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            new_pos_2d = projected_pin.reshape(-1, 2)[0]
        except cv2.error:
            # If projection fails, use barycentric interpolation in 2D as fallback
            v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
            new_pos_2d = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        
        # Always use 5-tuple format for new pins
        updated_pins.append((new_pos_2d[0], new_pos_2d[1], face_idx, bc, pin_pos_3d))
    
    state.pins_per_image[state.current_image_idx] = updated_pins

def synchronize_pins_across_views(state):
    """
    Ensure pins are consistent across all views by updating 2D positions
    based on the shared 3D model
    """
    # Store current view
    current_view = state.current_image_idx
    current_img = state.overlay.copy()
    current_img_h, current_img_w = state.img_h, state.img_w
    
    # For each view
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

def remove_pins(state):
    """Remove custom pins or landmark pins from the current image"""
    # First check if there are any custom pins
    if len(state.pins_per_image[state.current_image_idx]) > 0:
        # Remove all custom pins (original behavior)
        state.pins_per_image[state.current_image_idx] = []
        print("Removed all custom pins from current image")
    else:
        # If no custom pins exist, permanently hide landmark pins
        state.landmark_pins_hidden = True
        
        # Clear alignment cache for this image to force re-alignment next time
        from model.landmarks import clear_image_alignment
        clear_image_alignment(state)
        
        print("Landmark pins removed")

def center_geo(state):
    """Reset the mesh to its default position and set up perspective projection"""
    # Reset 3D vertices to default (original model)
    state.verts3d = state.verts3d_default.copy()
    
    # Set up proper camera parameters with perspective projection
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
    distance = -0.5 
    tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
    
    # Store camera parameters for the current image
    state.camera_matrices[state.current_image_idx] = camera_matrix
    state.rotations[state.current_image_idx] = rvec
    state.translations[state.current_image_idx] = tvec
    
    # Project 3D to 2D using perspective projection
    if 'project_2d' in state.callbacks:
        state.callbacks['project_2d'](state)
    else:
        from model.mesh import project_current_3d_to_2d
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
    from utils.geometry import calculate_front_facing
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    print("Reset mesh to default position with perspective projection")
    

def reset_shape(state):
    """Reset only the shape parameters while preserving position and orientation"""
    
    # Store the current transformation
    current_center_3d = np.mean(state.verts3d, axis=0)
    current_scale_3d = np.mean(np.linalg.norm(state.verts3d - current_center_3d, axis=1))
    
    # Reset to default shape in 3D space
    state.verts3d = state.verts3d_default.copy()
    
    # Apply the current scale and position to maintain orientation
    default_center_3d = np.mean(state.verts3d, axis=0)
    default_scale_3d = np.mean(np.linalg.norm(state.verts3d - default_center_3d, axis=1))
    
    # Scale factor to maintain current size
    scale_factor = current_scale_3d / default_scale_3d if default_scale_3d > 0 else 1.0
    
    # Apply scaling and translation
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
        from model.mesh import project_current_3d_to_2d
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
    from utils.geometry import calculate_front_facing
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Synchronize pins across all views to maintain consistency
    synchronize_pins_across_views(state)
    
    print("Reset mesh shape to default while preserving position and orientation")