# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2 
from model.mesh import project_current_3d_to_2d, update_3d_vertices

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
        # Also calculate the 3D position of the pin using barycentric coordinates
        i0, i1, i2 = state.faces[closest_face_idx]
        v0_3d = state.verts3d[i0]
        v1_3d = state.verts3d[i1]
        v2_3d = state.verts3d[i2]
        pin_pos_3d = bc_coords[0]*v0_3d + bc_coords[1]*v1_3d + bc_coords[2]*v2_3d
        
        # Add pin data to the current image's pin list
        state.pins_per_image[state.current_image_idx].append((x, y, closest_face_idx, bc_coords, pin_pos_3d))
        
        print(f"Added pin at ({x}, {y}) to face {closest_face_idx}")
        state.callbacks['redraw'](state)
    else:
        print("Could not find a face for the pin")

def update_custom_pins(state):
    """Update positions of custom pins after mesh movement"""
    updated_pins = []
    for pin_data in state.pins_per_image[state.current_image_idx]:
        # Support both 4-tuple and 5-tuple pin formats for backward compatibility
        face_idx = pin_data[2]
        bc = pin_data[3]
        
        i0, i1, i2 = state.faces[face_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        new_pos_2d = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        
        # Update the 3D position based on the current 3D mesh
        v0_3d, v1_3d, v2_3d = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        new_pos_3d = bc[0]*v0_3d + bc[1]*v1_3d + bc[2]*v2_3d
        
        # Always use 5-tuple format for new pins
        updated_pins.append((new_pos_2d[0], new_pos_2d[1], face_idx, bc, new_pos_3d))
    
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
        # We'll use a simple flag since we can't really delete the landmarks
        # (they're needed for the underlying mesh structure)
        state.landmark_pins_hidden = True
        print("Landmark pins removed")
    
    # Redraw to update the display
    state.callbacks['redraw'](state)

def center_geo(state):
    """Reset the mesh to its default position and clear any camera parameters"""
    # Reset 3D vertices to default (original model)
    state.verts3d = state.verts3d_default.copy()
    
    # Reset 3D landmarks to default
    state.landmark3d = state.landmark3d_default.copy()
    
    # Project to 2D for the current view
    # Start with default 2D positions
    state.verts2d = state.verts2d_default.copy()
    state.landmark_positions = state.landmark_positions_default.copy()
    
    # Clear camera parameters for the current image
    # ensure no alignment parameters remain
    state.camera_matrices[state.current_image_idx] = None
    state.rotations[state.current_image_idx] = None
    state.translations[state.current_image_idx] = None
    
    # Update custom pins
    update_custom_pins(state)
    
    print("Reset mesh to default position")
    state.callbacks['redraw'](state)

# Copyright (c) CUBOX, Inc. and its affiliates.
# Only the reset_shape function - replace this in pins.py

def reset_shape(state):
    """Reset only the shape parameters while preserving position and orientation"""
    
    # Store the current transformation
    current_center_3d = np.mean(state.verts3d, axis=0)
    current_scale_3d = np.mean(np.linalg.norm(state.verts3d - current_center_3d, axis=1))
    
    # Reset to default shape in 3D space
    state.verts3d = state.verts3d_default.copy()
    
    # Check if we have camera parameters for this view
    if (state.camera_matrices[state.current_image_idx] is not None and 
        state.rotations[state.current_image_idx] is not None and 
        state.translations[state.current_image_idx] is not None):
        
        # Project default 3D mesh to 2D using current camera parameters
        project_current_3d_to_2d(state)
        
        # Update landmarks explicitly based on the new mesh
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        
        # Update custom pins
        update_custom_pins(state)
        
        print("Reset mesh shape to default while preserving position and orientation")
    else:
        # No camera parameters, use 2D transformation
        # Set 2D vertices to default positions
        state.verts2d = state.verts2d_default.copy()
        
        # Apply the stored 2D transformation
        new_center = np.mean(state.verts2d, axis=0)
        new_scale = np.mean(np.linalg.norm(state.verts2d - new_center, axis=1))
        
        # Scale and translate to match the previous transformation
        scale_factor = current_scale_3d / new_scale
        state.verts2d = (state.verts2d - new_center) * scale_factor + current_center_3d[:2]
        
        # Update the 3D vertices to match the 2D transformation
        update_3d_vertices(state)
        
        # Update landmarks explicitly based on the new mesh
        from model.landmarks import update_all_landmarks
        update_all_landmarks(state)
        
        # Update custom pins
        update_custom_pins(state)
        
        print("Reset mesh shape to default using 2D transformation")
    
    # Synchronize pins across all views to maintain consistency
    synchronize_pins_across_views(state)
    
    state.callbacks['redraw'](state)