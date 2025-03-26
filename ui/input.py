# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np
from config import Mode, PIN_SELECTION_THRESHOLD
from model.pins import add_custom_pin, update_custom_pins, synchronize_pins_across_views
from model.mesh import move_mesh_2d
from model.landmarks import update_all_landmarks

def on_mouse(event, x, y, flags, _, state):
    """Handle mouse events for the UI"""
    # Calculate UI dimensions based on current image size
    ui = state.ui_dimensions
    
    # Add zoom functionality with mouse wheel
    if event == cv2.EVENT_MOUSEWHEEL:
        if state.mode == Mode.VIEW_3D:
            # In OpenCV, flags contain the scroll information 
            # Positive for zoom in, negative for zoom out
            wheel_direction = 1 if flags > 0 else -1
            
            # Initialize zoom factor if not present
            if not hasattr(state, 'view_3d_zoom'):
                state.view_3d_zoom = 1.0
                
            # Update zoom factor (adjust the 0.1 multiplier for sensitivity)
            state.view_3d_zoom += wheel_direction * 0.1
            
            # Clamp zoom to reasonable range
            state.view_3d_zoom = max(0.5, min(3.0, state.view_3d_zoom))
            
            state.callbacks['redraw'](state)
            return
    
    # Check for button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Center Geo button 
        bx, by, bw, bh = ui['center_geo_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['center_geo'](state)
            return
            
        # Align Face button
        bx, by, bw, bh = ui['align_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['align_face'](state)
            return
            
        # Reset Shape button 
        bx, by, bw, bh = ui['reset_shape_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['reset_shape'](state)
            return
        
        # Remove Pins button 
        bx, by, bw, bh = ui['remove_pins_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['remove_pins'](state)
            return
            
        # Toggle Pins button 
        bx, by, bw, bh = ui['toggle_pins_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.mode = Mode.TOGGLE_PINS if state.mode != Mode.TOGGLE_PINS else Mode.MOVE
            state.callbacks['update_ui'](state)
            return
            
        # Save Mesh button 
        bx, by, bw, bh = ui['save_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['save_model'](state)
            return
            
        # Next Image button 
        bx, by, bw, bh = ui['next_img_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['next_image'](state)
            return
            
        # Previous Image button 
        bx, by, bw, bh = ui['prev_img_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.callbacks['prev_image'](state)
            return
            
        # 3D Visualizer button (new)
        bx, by, bw, bh = ui['visualizer_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            # Toggle 3D view mode
            state.mode = Mode.VIEW_3D if state.mode != Mode.VIEW_3D else Mode.MOVE
            
            # Initialize rotation angles if first time entering 3D mode
            if not hasattr(state, 'view_3d_rotation_x'):
                state.view_3d_rotation_x = 0.0
                state.view_3d_rotation_y = 0.0
                
            state.callbacks['update_ui'](state)
            return
        
        # In 3D view mode, start rotation
        if state.mode == Mode.VIEW_3D:
            state.drag_start_pos = (x, y)
            return
        
        # In MOVE mode, first check if the click is near an existing pin or landmark
        if state.mode == Mode.MOVE:
            # Calculate adaptive search radius based on image size and the PIN_SELECTION_THRESHOLD
            pin_selection_threshold_sq = PIN_SELECTION_THRESHOLD ** 2
            pin_radius_sq = (ui['pin_radius'] * 1.5) ** 2  # Slightly larger than visible radius
            landmark_radius_sq = (ui['landmark_radius'] * 1.5) ** 2
            
            # Initialize flag to track if we found a pin/landmark
            found_pin_or_landmark = False
            
            # Check if click is on a custom pin first
            for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
                # Handle both 4-tuple and 5-tuple pin formats for backward compatibility
                if len(pin_data) >= 2:  # At minimum we need x,y
                    px, py = pin_data[0], pin_data[1]
                    dx, dy = px - x, py - y
                    dist_sq = dx*dx + dy*dy
                    
                    # Check if click is within the pin selection threshold
                    if dist_sq < pin_selection_threshold_sq:
                        state.drag_index = i + len(state.landmark_positions)  # Offset by landmark count
                        state.drag_offset = (px - x, py - y)
                        found_pin_or_landmark = True
                        break
            
            # If not a custom pin, check if it's a landmark
            landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
            if not found_pin_or_landmark and not landmark_pins_hidden:
                for i, (lx, ly) in enumerate(state.landmark_positions):
                    dx, dy = lx - x, ly - y
                    dist_sq = dx*dx + dy*dy
                    
                    # Check if click is within the landmark selection threshold
                    if dist_sq < pin_selection_threshold_sq:
                        state.drag_index = i
                        state.drag_offset = (lx - x, ly - y)
                        found_pin_or_landmark = True
                        break
            
            # If no existing pin or landmark was found, create a new pin at the click location
            if not found_pin_or_landmark:
                add_custom_pin(x, y, state)
                return
        
        # Handle pin selection in TOGGLE_PINS mode
        elif state.mode == Mode.TOGGLE_PINS:
            # Calculate adaptive search radius for TOGGLE_PINS mode
            pin_radius_sq = (ui['pin_radius'] * 1.5) ** 2  # Slightly larger than visible radius
            
            # Check if click is on a custom pin
            for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
                # Handle both 4-tuple and 5-tuple pin formats for backward compatibility
                if len(pin_data) >= 2:  # At minimum we need x,y
                    px, py = pin_data[0], pin_data[1]
                    dx, dy = px - x, py - y
                    if dx*dx + dy*dy < pin_radius_sq:
                        state.drag_index = i + len(state.landmark_positions)  # Offset by landmark count
                        state.drag_offset = (px - x, py - y)
                        return

    elif event == cv2.EVENT_MOUSEMOVE:
        # Handle 3D view rotation
        if state.mode == Mode.VIEW_3D and hasattr(state, 'drag_start_pos') and state.drag_start_pos is not None:
            # Calculate rotation based on mouse movement
            dx = x - state.drag_start_pos[0]
            dy = y - state.drag_start_pos[1]
            
            # Update 3D view rotation (scaled for better control)
            state.view_3d_rotation_y += dx * 0.01
            state.view_3d_rotation_x += dy * 0.01
            
            # Save current position for next move
            state.drag_start_pos = (x, y)
            
            state.callbacks['redraw'](state)
            return
            
        if state.drag_index != -1:
            # Handle dragging landmarks
            if state.drag_index < len(state.landmark_positions) and state.mode != Mode.TOGGLE_PINS:
                ox, oy = state.landmark_positions[state.drag_index]
                nx = x + state.drag_offset[0]
                ny = y + state.drag_offset[1]
                dx, dy = nx - ox, ny - oy
                
                # Move mesh in 2D and update 3D vertices
                move_mesh_2d(state, ox, oy, dx, dy)
                
                # Update landmarks and custom pins
                update_all_landmarks(state)
                update_custom_pins(state)
            
            # Handle dragging custom pins
            else:
                pin_idx = state.drag_index - len(state.landmark_positions)
                if pin_idx < len(state.pins_per_image[state.current_image_idx]):
                    pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
                    
                    # Support both 4-tuple and 5-tuple pin formats
                    ox, oy = pin_data[0], pin_data[1]
                    face_idx = pin_data[2]
                    bc = pin_data[3]
                    
                    # Get 3D position if available (5-tuple format)
                    pin_pos_3d = None
                    if len(pin_data) >= 5:
                        pin_pos_3d = pin_data[4]
                    
                    nx = x + state.drag_offset[0]
                    ny = y + state.drag_offset[1]
                    
                    if state.mode == Mode.TOGGLE_PINS:
                        # In toggle pins mode, just move the pin without affecting the mesh
                        if pin_pos_3d is not None:
                            # 5-tuple format
                            state.pins_per_image[state.current_image_idx][pin_idx] = (nx, ny, face_idx, bc, pin_pos_3d)
                        else:
                            # 4-tuple format
                            state.pins_per_image[state.current_image_idx][pin_idx] = (nx, ny, face_idx, bc)
                    else:
                        # Check if we have multiple pins for rigid transformation
                        pins = state.pins_per_image[state.current_image_idx]
                        if len(pins) >= 2:
                            # Use rigid transformation for multiple pins
                            from model.mesh import transform_mesh_rigid
                            transform_mesh_rigid(state, pin_idx, ox, oy, nx, ny)
                        else:
                            # Use regular deformation for a single pin
                            dx, dy = nx - ox, ny - oy
                            move_mesh_2d(state, ox, oy, dx, dy)
                            
                            # Update landmarks and all pins
                            update_all_landmarks(state)
                            update_custom_pins(state)
            
            state.callbacks['redraw'](state)

    elif event == cv2.EVENT_LBUTTONUP:
        # Reset drag start position in 3D view mode
        if state.mode == Mode.VIEW_3D and hasattr(state, 'drag_start_pos'):
            state.drag_start_pos = None
            
        if state.drag_index != -1:
            # After finishing a drag operation, synchronize pins across all views
            if hasattr(state.callbacks, 'synchronize_pins') and callable(state.callbacks['synchronize_pins']):
                state.callbacks['synchronize_pins'](state)
            elif 'synchronize_pins_across_views' in globals():
                # Call it directly if it's in globals
                synchronize_pins_across_views(state)
                
        state.drag_index = -1