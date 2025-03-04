# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2 
from model.mesh import project_current_3d_to_2d

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
        # Also store the 3D position of the pin using barycentric coordinates
        i0, i1, i2 = state.faces[closest_face_idx]
        v0_3d = state.verts3d[i0]
        v1_3d = state.verts3d[i1]
        v2_3d = state.verts3d[i2]
        pin_pos_3d = bc_coords[0]*v0_3d + bc_coords[1]*v1_3d + bc_coords[2]*v2_3d
        
        # Add pin data to the current image's pin list
        # Now store: (2D x, 2D y, face index, barycentric coordinates, 3D position)
        state.pins_per_image[state.current_image_idx].append((x, y, closest_face_idx, bc_coords, pin_pos_3d))
        
        print(f"Added pin at ({x}, {y}) to face {closest_face_idx}")
        state.callbacks['redraw'](state)
    else:
        print("Could not find a face for the pin")

def update_custom_pins(state):
    """Update positions of custom pins after mesh movement"""
    updated_pins = []
    for _, _, face_idx, bc, pin_pos_3d in state.pins_per_image[state.current_image_idx]:
        i0, i1, i2 = state.faces[face_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        new_pos_2d = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        
        # Update the 3D position based on the current 3D mesh
        v0_3d, v1_3d, v2_3d = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        new_pos_3d = bc[0]*v0_3d + bc[1]*v1_3d + bc[2]*v2_3d
        
        updated_pins.append((new_pos_2d[0], new_pos_2d[1], face_idx, bc, new_pos_3d))
    
    state.pins_per_image[state.current_image_idx] = updated_pins

def synchronize_pins_across_views(state):
    """
    Ensure pins are consistent across all views by updating 2D positions
    based on the shared 3D model
    """
    current_view = state.current_image_idx
    
    # For each view
    for view_idx in range(len(state.images)):
        if view_idx == current_view:
            continue  # Skip current view as it's already up to date
            
        # Store the current view
        temp_view = state.current_image_idx
        
        # Switch to the view we're updating
        state.current_image_idx = view_idx
        state.overlay = state.images[view_idx].copy()
        state.img_h, state.img_w = state.overlay.shape[:2]
        
        # Project the current 3D model to this view
        project_current_3d_to_2d(state)
        
        # Update pins for this view
        update_custom_pins(state)
        
        # Switch back to original view
        state.current_image_idx = temp_view
        state.overlay = state.images[temp_view].copy()
        state.img_h, state.img_w = state.overlay.shape[:2]

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
    # Reset to default positions while preserving mesh modifications
    current_center_3d = np.mean(state.verts3d, axis=0)
    default_center_3d = np.mean(state.verts3d_default, axis=0)
    
    # Translate the 3D mesh to the default center
    translation = default_center_3d - current_center_3d
    state.verts3d += translation
    
    # Update the 2D projection
    state.verts2d = state.verts2d_default.copy()
    state.landmark_positions = state.landmark_positions_default.copy()
    
    # Clear camera parameters for the current image
    # ensure no alignment parameters remain
    state.camera_matrices[state.current_image_idx] = None
    state.rotations[state.current_image_idx] = None
    state.translations[state.current_image_idx] = None
    
    # Project the 3D model to 2D
    project_current_3d_to_2d(state)
    
    # Update landmarks based on new 2D projection
    from model.landmarks import update_all_landmarks
    update_all_landmarks(state)
    
    # Update pins
    update_custom_pins(state)
    
    print("Centered mesh position")
    state.callbacks['redraw'](state)

def reset_shape(state):
    """Reset only the FLAME shape parameters while preserving position and orientation"""
    
    # Store the current transformation
    current_center_3d = np.mean(state.verts3d, axis=0)
    current_scale_3d = np.mean(np.linalg.norm(state.verts3d - current_center_3d, axis=1))
    
    # Reset to default shape
    state.verts3d = state.verts3d_default.copy()
    
    # Apply the current scale and position to maintain orientation
    default_center_3d = np.mean(state.verts3d, axis=0)
    default_scale_3d = np.mean(np.linalg.norm(state.verts3d - default_center_3d, axis=1))
    
    # Scale factor to maintain current size
    scale_factor = current_scale_3d / default_scale_3d
    
    # Apply scaling and translation
    state.verts3d = (state.verts3d - default_center_3d) * scale_factor + current_center_3d
    
    # Project the updated 3D mesh to 2D
    if (state.camera_matrices[state.current_image_idx] is not None and 
        state.rotations[state.current_image_idx] is not None and 
        state.translations[state.current_image_idx] is not None):
        
        # If we have camera parameters, use them for projection
        project_current_3d_to_2d(state)
    else:
        # Otherwise use orthographic projection
        mn = state.verts3d[:, :2].min(axis=0)
        mx = state.verts3d[:, :2].max(axis=0)
        c3d = 0.5 * (mn + mx)
        s3d = (mx - mn).max()
        sc = 0.8 * min(state.img_w, state.img_h) / s3d
        c2d = np.array([state.img_w/2.0, state.img_h/2.0])
        
        from utils.geometry import ortho
        state.verts2d = ortho(state.verts3d, c3d, c2d, sc)
    
    # Update landmarks
    from model.landmarks import update_all_landmarks
    update_all_landmarks(state)
    
    # Update custom pins
    update_custom_pins(state)
    
    # Synchronize pins across all views
    synchronize_pins_across_views(state)
    
    print("Reset mesh shape to default while preserving position and orientation")
    state.callbacks['redraw'](state)