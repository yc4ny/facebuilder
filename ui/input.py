# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
from config import Mode
from model.pins import add_custom_pin, update_custom_pins
from model.mesh import move_mesh_2d
from model.landmarks import update_all_landmarks

def on_mouse(event, x, y, flags, _, state):
    """Handle mouse events for the UI"""
    # Calculate UI dimensions based on current image size
    ui = state.ui_dimensions
    
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
        
        # Add Pin button 
        bx, by, bw, bh = ui['add_pin_button_rect']
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state.mode = Mode.ADD_PIN if state.mode != Mode.ADD_PIN else Mode.MOVE
            state.callbacks['update_ui'](state)
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
            
        # Handle pin adding in ADD_PIN mode
        if state.mode == Mode.ADD_PIN:
            add_custom_pin(x, y, state)
            return
        
        # Handle landmark or pin dragging in MOVE mode
        if state.mode == Mode.MOVE or state.mode == Mode.TOGGLE_PINS:
            # Calculate adaptive search radius based on image size
            pin_radius_sq = (ui['pin_radius'] * 1.5) ** 2  # Slightly larger than visible radius
            landmark_radius_sq = (ui['landmark_radius'] * 1.5) ** 2
            
            # Check if click is on a custom pin first
            for i, (px, py, _, _) in enumerate(state.pins_per_image[state.current_image_idx]):
                dx, dy = px - x, py - y
                if dx*dx + dy*dy < pin_radius_sq:
                    state.drag_index = i + len(state.landmark_positions)  # Offset by landmark count
                    state.drag_offset = (px - x, py - y)
                    return
            
            # In TOGGLE_PINS mode, we don't check for landmarks
            if state.mode != Mode.TOGGLE_PINS:
                # If not a custom pin, check if it's a landmark
                for i, (lx, ly) in enumerate(state.landmark_positions):
                    dx, dy = lx - x, ly - y
                    if dx*dx + dy*dy < landmark_radius_sq:
                        state.drag_index = i
                        state.drag_offset = (lx - x, ly - y)
                        break

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drag_index != -1:
            # Handle dragging landmarks
            if state.drag_index < len(state.landmark_positions) and state.mode != Mode.TOGGLE_PINS:
                ox, oy = state.landmark_positions[state.drag_index]
                nx = x + state.drag_offset[0]
                ny = y + state.drag_offset[1]
                dx, dy = nx - ox, ny - oy
                move_mesh_2d(state, ox, oy, dx, dy)
                update_all_landmarks(state)
            # Handle dragging custom pins
            else:
                pin_idx = state.drag_index - len(state.landmark_positions)
                if pin_idx < len(state.pins_per_image[state.current_image_idx]):
                    ox, oy, face_idx, bc = state.pins_per_image[state.current_image_idx][pin_idx]
                    nx = x + state.drag_offset[0]
                    ny = y + state.drag_offset[1]
                    dx, dy = nx - ox, ny - oy
                    
                    if state.mode == Mode.TOGGLE_PINS:
                        # In toggle pins mode, just move the pin without affecting the mesh
                        state.pins_per_image[state.current_image_idx][pin_idx] = (nx, ny, face_idx, bc)
                    else:
                        # In regular mode, move the mesh with the pin
                        move_mesh_2d(state, ox, oy, dx, dy)
                        update_all_landmarks(state)
                        update_custom_pins(state)
            
            state.callbacks['redraw'](state)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drag_index = -1