
# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2 

def add_custom_pin(x, y, state):
    """Add a new custom pin at the given coordinates"""
    # Find the closest face to the pin position
    closest_face_idx = -1
    min_dist = float('inf')
    pin_pos = np.array([x, y])
    
    for face_idx, (i0, i1, i2) in enumerate(state.faces):
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
        # Add pin data to the current image's pin list
        state.pins_per_image[state.current_image_idx].append((x, y, closest_face_idx, bc_coords))
        
        print(f"Added pin at ({x}, {y}) to face {closest_face_idx}")
        state.callbacks['redraw'](state)
    else:
        print("Could not find a face for the pin")

def update_custom_pins(state):
    """Update positions of custom pins after mesh movement"""
    updated_pins = []
    for _, _, face_idx, bc in state.pins_per_image[state.current_image_idx]:
        i0, i1, i2 = state.faces[face_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        new_pos = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        updated_pins.append((new_pos[0], new_pos[1], face_idx, bc))
    
    state.pins_per_image[state.current_image_idx] = updated_pins

def remove_pins(state):
    """Remove custom pins or landmark pins from the current image"""
    # First check if there are any custom pins
    if len(state.pins_per_image[state.current_image_idx]) > 0:
        # Remove all custom pins (original behavior)
        state.pins_per_image[state.current_image_idx] = []
        print("Removed all custom pins from current image")
    else:
        # If no custom pins exist, permanently hide landmark pins
        # We'll use a simple flag since we can't really delete the landmarks
        # (they're needed for the underlying mesh structure)
        state.landmark_pins_hidden = True
        print("Landmark pins removed")
    
    # Redraw to update the display
    state.callbacks['redraw'](state)

def center_geo(state):
    """Reset the mesh to its default position and clear any camera parameters"""
    # Reset to default positions
    state.verts2d = state.verts2d_default.copy()
    state.landmark_positions = state.landmark_positions_default.copy()
    
    # Clear camera parameters for the current image
    # ensure no alignment parameters remain
    state.camera_matrices[state.current_image_idx] = None
    state.rotations[state.current_image_idx] = None
    state.translations[state.current_image_idx] = None
    
    state.callbacks['update_custom_pins'](state)
    
    print("Reset mesh to default position")
    state.callbacks['redraw'](state)

def reset_shape(state):
    """Reset only the FLAME shape parameters while preserving position and orientation"""
    
    # Store the current transformation
    current_center = np.mean(state.verts2d, axis=0)
    current_scale = np.mean(np.linalg.norm(state.verts2d - current_center, axis=1))
    
    # Reset to default shape
    if state.camera_matrices[state.current_image_idx] is not None:
        # If we have camera parameters, project the default 3D vertices
        camera_matrix = state.camera_matrices[state.current_image_idx]
        rvec = state.rotations[state.current_image_idx]
        tvec = state.translations[state.current_image_idx]
        
        # Project default 3D vertices to 2D using current camera parameters
        projected_verts, _ = cv2.projectPoints(
            np.array(state.verts3d_default, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        new_verts = projected_verts.reshape(-1, 2)
        
        # Project default landmarks
        projected_landmarks, _ = cv2.projectPoints(
            np.array(state.landmark3d_default, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        new_landmarks = projected_landmarks.reshape(-1, 2)
    else:
        # If no camera parameters, use 2D default with current transformation
        new_verts = state.verts2d_default.copy()
        new_landmarks = state.landmark_positions_default.copy()
        
        # Apply the current transformation (center and scale)
        new_center = np.mean(new_verts, axis=0)
        new_scale = np.mean(np.linalg.norm(new_verts - new_center, axis=1))
        
        # Scale to match current size
        scale_factor = current_scale / new_scale
        new_verts = (new_verts - new_center) * scale_factor + current_center
        new_landmarks = (new_landmarks - new_center) * scale_factor + current_center
    
    # Update the vertices and landmarks
    state.verts2d = new_verts
    state.landmark_positions = new_landmarks
    
    # Update custom pins based on the reset mesh
    state.callbacks['update_custom_pins'](state)
    
    print("Reset mesh shape to default while preserving position and orientation")
    state.callbacks['redraw'](state)
