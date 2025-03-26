# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
from config import UI_BUTTON_HEIGHT_RATIO, UI_BUTTON_MARGIN_RATIO, UI_TEXT_SIZE_RATIO, UI_LINE_THICKNESS_RATIO

def calculate_ui_dimensions(img_w, img_h):
    """Calculate UI element dimensions based on image size"""
    # Calculate image diagonal for scale reference
    img_diag = np.sqrt(img_w**2 + img_h**2)
    
    # Button dimensions
    button_height = int(img_h * UI_BUTTON_HEIGHT_RATIO)
    button_margin = int(img_w * UI_BUTTON_MARGIN_RATIO)
    
    # Use a smaller text size to prevent overflow
    text_size = max(0.4, img_diag * UI_TEXT_SIZE_RATIO * 0.6)
    text_thickness = max(1, int(img_diag * 0.0005))
    line_thickness = max(1, int(img_diag * UI_LINE_THICKNESS_RATIO))
    vertex_radius = max(1, int(img_diag * 0.0001))
    landmark_radius = max(3, int(img_diag * 0.002))
    pin_radius = max(3, int(img_diag * 0.002))
    
    # Simple fixed-width buttons
    button_width = int(button_height * 1.8)  # Standard width for most buttons
    center_geo_button_width = button_width
    align_button_width = button_width
    reset_shape_button_width = button_width
    remove_pins_button_width = button_width
    toggle_pins_button_width = button_width
    save_button_width = button_width
    next_img_button_width = button_width
    prev_img_button_width = button_width
    visualizer_button_width = button_width  # New 3D visualizer button
    
    # Calculate button positions - all in one row
    current_x = button_margin
    
    # Center Geo button (1st)
    center_geo_button_rect = (current_x, button_margin, center_geo_button_width, button_height)
    current_x += center_geo_button_width + button_margin
    
    # Align Face button (2nd)
    align_button_rect = (current_x, button_margin, align_button_width, button_height)
    current_x += align_button_width + button_margin
    
    # Reset Shape button (3rd)
    reset_shape_button_rect = (current_x, button_margin, reset_shape_button_width, button_height)
    current_x += reset_shape_button_width + button_margin
    
    # Remove Pins button (4th)
    remove_pins_button_rect = (current_x, button_margin, remove_pins_button_width, button_height)
    current_x += remove_pins_button_width + button_margin
    
    # Toggle Pins button (5th)
    toggle_pins_button_rect = (current_x, button_margin, toggle_pins_button_width, button_height)
    current_x += toggle_pins_button_width + button_margin
    
    # Save Mesh button (6th)
    save_button_rect = (current_x, button_margin, save_button_width, button_height)
    current_x += save_button_width + button_margin
    
    # Next Image button (7th)
    next_img_button_rect = (current_x, button_margin, next_img_button_width, button_height)
    current_x += next_img_button_width + button_margin
    
    # Prev Image button (8th)
    prev_img_button_rect = (current_x, button_margin, prev_img_button_width, button_height)
    current_x += prev_img_button_width + button_margin
    
    # 3D Visualizer button (9th - new)
    visualizer_button_rect = (current_x, button_margin, visualizer_button_width, button_height)
    
    # Status text position
    status_text_x = img_w - int(img_w * 0.25)
    status_text_y1 = button_margin + int(button_height * 0.75)
    status_text_y2 = status_text_y1 + button_height
    
    return {
        'center_geo_button_rect': center_geo_button_rect,
        'align_button_rect': align_button_rect,
        'reset_shape_button_rect': reset_shape_button_rect,
        'remove_pins_button_rect': remove_pins_button_rect,
        'toggle_pins_button_rect': toggle_pins_button_rect,
        'save_button_rect': save_button_rect,
        'next_img_button_rect': next_img_button_rect,
        'prev_img_button_rect': prev_img_button_rect,
        'visualizer_button_rect': visualizer_button_rect,  # Added 3D Visualizer button
        'text_size': text_size,
        'text_thickness': text_thickness,
        'line_thickness': line_thickness,
        'vertex_radius': vertex_radius,
        'landmark_radius': landmark_radius,
        'pin_radius': pin_radius,
        'status_text_pos': (status_text_x, status_text_y1, status_text_y2)
    }