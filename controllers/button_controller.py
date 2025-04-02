"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Button controller module.

This module manages button callbacks and state updates for the application's UI,
connecting user interface components with their respective functionality.
"""
from config import Mode

class ButtonController:
    """
    Controller for managing button interactions and state.
    
    This class provides methods to initialize button callbacks and update button
    states based on the current application mode.
    """
    
    @staticmethod
    def initialize_button_callbacks(state):
        """
        Set up button callbacks to connect UI to functionality.
        
        Maps button text labels to their corresponding functions in the state object,
        allowing buttons to trigger the appropriate actions when clicked.
        
        Args:
            state: Application state containing button definitions and callback methods
        """
        # Map button text to methods
        button_callbacks = {
            "Center Geo": state.center_geo,
            "Align Face": state.align_face,
            "Reset Shape": state.reset_shape,
            "Remove Pins": state.remove_pins,
            "Toggle Pins": state.toggle_pins_mode,
            "Save Mesh": state.save_model,
            "Next Image": state.next_image,
            "Prev Image": state.prev_image,
            "3D View": state.toggle_3d_view_mode
        }
        
        # Apply callbacks to buttons
        for button in state.buttons:
            if button.text in button_callbacks:
                button.callback = button_callbacks[button.text]
    
    @staticmethod
    def update_button_states(state):
        """
        Update button states based on current application mode.
        
        This method ensures toggle buttons properly reflect the current mode
        of the application, such as whether pin toggling or 3D view is active.
        
        Args:
            state: Application state containing current mode and button references
        """
        for button in state.buttons:
            if button.text == "Toggle Pins":
                button.is_active = (state.mode == Mode.TOGGLE_PINS)
            elif button.text == "3D View":
                button.is_active = (state.mode == Mode.VIEW_3D)