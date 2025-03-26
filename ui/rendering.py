# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np
from config import WINDOW_NAME, Mode

def redraw(state):
    """Redraw the UI and mesh"""
    disp = state.overlay.copy()
    
    # Calculate UI dimensions based on current image size
    ui = state.ui_dimensions
    
    # Check if we have determined which faces are front-facing
    # If not, default to showing all faces
    front_facing = state.front_facing if state.front_facing is not None else np.ones(len(state.faces), dtype=bool)
    
    # Draw mesh triangles (only front-facing)
    for i, (i0, i1, i2) in enumerate(state.faces):
        if front_facing[i]:  # Only draw front-facing faces
            p0 = tuple(np.round(state.verts2d[i0]).astype(int))
            p1 = tuple(np.round(state.verts2d[i1]).astype(int))
            p2 = tuple(np.round(state.verts2d[i2]).astype(int))
            cv2.line(disp, p0, p1, (0,0,0), ui['line_thickness'])
            cv2.line(disp, p1, p2, (0,0,0), ui['line_thickness'])
            cv2.line(disp, p2, p0, (0,0,0), ui['line_thickness'])
    
    # Draw vertices (only those used by front-facing faces)
    visible_vertices = set()
    for i, face in enumerate(state.faces):
        if front_facing[i]:
            visible_vertices.update(face)
    
    for i in visible_vertices:
        x, y = state.verts2d[i]
        cv2.circle(disp, (int(round(x)), int(round(y))), ui['vertex_radius'], (255,0,0), -1)
    
    # Draw landmarks
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    if not landmark_pins_hidden:
        for (lx, ly) in state.landmark_positions:
            cv2.circle(disp, (int(round(lx)), int(round(ly))), ui['landmark_radius'], (0,0,255), -1)
    
    # Draw custom pins for the current image
    for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
        # Handle both 4-tuple and 5-tuple pin formats for backward compatibility
        if len(pin_data) >= 2:  # At minimum we need x,y
            px, py = pin_data[0], pin_data[1]
            pin_color = (0,255,0)
            cv2.circle(disp, (int(round(px)), int(round(py))), ui['pin_radius'], pin_color, -1)
    
    # Helper function to center text in button
    def draw_button_with_text(rect, text, is_active=False):
        bx, by, bw, bh = rect
        button_color = (100,100,255) if is_active else (50,50,50)
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), button_color, -1)
        
        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                     ui['text_size'], ui['text_thickness'])
        text_x = bx + (bw - text_width) // 2
        text_y = by + (bh + text_height) // 2
        
        cv2.putText(disp, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, ui['text_size'], (255,255,255), ui['text_thickness'])
    
    # Draw all buttons in the new order
    draw_button_with_text(ui['center_geo_button_rect'], "Center Geo")
    draw_button_with_text(ui['align_button_rect'], "Align Face")
    draw_button_with_text(ui['reset_shape_button_rect'], "Reset Shape")
    draw_button_with_text(ui['remove_pins_button_rect'], "Remove Pins")
    draw_button_with_text(ui['toggle_pins_button_rect'], "Toggle Pins", state.mode == Mode.TOGGLE_PINS)
    draw_button_with_text(ui['save_button_rect'], "Save Mesh")
    draw_button_with_text(ui['next_img_button_rect'], "Next Image")
    draw_button_with_text(ui['prev_img_button_rect'], "Prev Image")
    
    # Show current image index
    status_x, status_y1, status_y2 = ui['status_text_pos']
    cv2.putText(disp, f"Image {state.current_image_idx+1}/{len(state.images)}", 
                (status_x, status_y1), cv2.FONT_HERSHEY_SIMPLEX, 
                ui['text_size'] * 2, (255,120,0), ui['text_thickness']*2)
    
    # Show current mode
    if state.mode == Mode.TOGGLE_PINS:
        mode_text = "Mode: TOGGLE PINS"
    else:
        # Count active pins to determine if we're in single pin mode or regular move mode
        custom_pin_count = len(state.pins_per_image[state.current_image_idx])
        landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
        
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(state.landmark_positions)
        
        if total_active_pins == 1:
            mode_text = "Mode: SINGLE PIN (MOVE+ROTATE)"
        else:
            mode_text = "Mode: MULTI-PIN DEFORM"
    
    cv2.putText(disp, mode_text, (status_x, status_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, ui['text_size']*2, (255,120,0), ui['text_thickness']*2)
    
    # Add a user hint for adding pins
    # Calculate hint position based on the UI dimensions
    hint_y = status_y2 + int(ui['landmark_radius'] * 8)
    cv2.putText(disp, "Click on mesh to add pins", 
                (status_x, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 
                ui['text_size']*1.5, (0,200,0), ui['text_thickness'])
    
    cv2.imshow(WINDOW_NAME, disp)