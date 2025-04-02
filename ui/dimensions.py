# Copyright (c) CUBOX, Inc. and its affiliates.
"""
UI dimensions calculation module.

This module provides functions to compute UI element dimensions based on
the current display size, ensuring responsive layout across different screens.
"""
import numpy as np
from config import UI_BUTTON_HEIGHT_RATIO, UI_BUTTON_MARGIN_RATIO, UI_TEXT_SIZE_RATIO, UI_LINE_THICKNESS_RATIO

def calculate_ui_dimensions(img_w, img_h):
    """
    Calculate UI element dimensions based on image size.
    
    This function computes all dimensions for UI elements as a function of the
    current display size, ensuring consistent appearance and layout across
    different screen resolutions.
    
    Args:
        img_w: Image width in pixels
        img_h: Image height in pixels
        
    Returns:
        dict: Dictionary containing all UI element dimensions and positions
    """
    # Calculate image diagonal for scale reference
    # Used to scale UI elements proportionally to the display size
    # diagonal = √(width² + height²)
    img_diag = np.sqrt(img_w**2 + img_h**2)
    
    # Calculate basic UI element dimensions based on image size
    button_height = int(img_h * UI_BUTTON_HEIGHT_RATIO)
    button_margin = int(img_w * UI_BUTTON_MARGIN_RATIO)
    
    # Calculate text and graphics sizes based on image diagonal
    # Using a smaller multiplier (0.6) for text_size to prevent overflow
    text_size = max(0.4, img_diag * UI_TEXT_SIZE_RATIO * 0.6)
    text_thickness = max(1, int(img_diag * 0.0005))
    line_thickness = max(1, int(img_diag * UI_LINE_THICKNESS_RATIO))
    
    # Calculate vertex and pin sizes based on image diagonal
    # Small constant ensures minimum visible size on any display
    vertex_radius = max(1, int(img_diag * 0.0001))
    landmark_radius = max(3, int(img_diag * 0.002))
    pin_radius = max(3, int(img_diag * 0.002))
    
    # Set standard button width as a function of button height
    # Using a 1.8:1 aspect ratio for buttons
    button_width = int(button_height * 1.8)
    
    # Button widths (all using standard width for consistency)
    center_geo_button_width = button_width
    align_button_width = button_width
    reset_shape_button_width = button_width
    remove_pins_button_width = button_width
    toggle_pins_button_width = button_width
    save_button_width = button_width
    next_img_button_width = button_width
    prev_img_button_width = button_width
    visualizer_button_width = button_width
    
    # Calculate button positions - layout all buttons in a horizontal row
    # Each button's position depends on the previous button's position plus its width
    current_x = button_margin
    
    # Position each button and increment x position for next button
    # Button format: (x, y, width, height)
    
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
    
    # 3D Visualizer button (9th)
    visualizer_button_rect = (current_x, button_margin, visualizer_button_width, button_height)
    
    # Calculate status text positions
    # Position text in the right section of the screen
    status_text_x = img_w - int(img_w * 0.25)  # 25% from right edge
    status_text_y1 = button_margin + int(button_height * 0.75)  # Align with buttons
    status_text_y2 = status_text_y1 + button_height  # Second line of status text
    
    # Return dictionary containing all UI element dimensions and positions
    return {
        'center_geo_button_rect': center_geo_button_rect,
        'align_button_rect': align_button_rect,
        'reset_shape_button_rect': reset_shape_button_rect,
        'remove_pins_button_rect': remove_pins_button_rect,
        'toggle_pins_button_rect': toggle_pins_button_rect,
        'save_button_rect': save_button_rect,
        'next_img_button_rect': next_img_button_rect,
        'prev_img_button_rect': prev_img_button_rect,
        'visualizer_button_rect': visualizer_button_rect,
        'text_size': text_size,
        'text_thickness': text_thickness,
        'line_thickness': line_thickness,
        'vertex_radius': vertex_radius,
        'landmark_radius': landmark_radius,
        'pin_radius': pin_radius,
        'status_text_pos': (status_text_x, status_text_y1, status_text_y2)
    }