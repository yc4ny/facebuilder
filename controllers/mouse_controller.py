# Copyright (c) CUBOX, Inc. and its affiliates.
"""
Mouse controller module.

This module handles all mouse interactions with the application, including
detecting clicks on buttons, managing mesh manipulation through dragging pins,
and handling 3D view control.
"""
import pygame
from pygame.locals import *
import numpy as np
from config import Mode, PIN_SELECTION_THRESHOLD
from model.pins import add_custom_pin, update_custom_pins, synchronize_pins_across_views
from model.landmarks import update_all_landmarks
from model.mesh import move_mesh_2d, transform_mesh_rigid

class MouseController:
    """
    Controller for managing mouse input events.
    
    This class provides methods to interpret mouse actions and trigger the
    appropriate responses in the application, including mesh manipulation,
    button clicks, and 3D view control.
    """
    
    @staticmethod
    def handle_mouse_event(event, state):
        """
        Entry point for handling all mouse events.
        
        Routes events to the appropriate handler based on the current application mode.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if state.mode == Mode.VIEW_3D:
            return MouseController._handle_3d_view_mouse_event(event, state)
        else:
            # Check button clicks first
            for button in state.buttons:
                if button.handle_event(event, state):
                    return True
            
            # Handle mesh and pin interactions
            return MouseController._handle_mesh_pin_mouse_event(event, state)
    
    @staticmethod
    def _handle_3d_view_mouse_event(event, state):
        """
        Handle mouse events in 3D view mode.
        
        Manages rotation and zooming of the 3D model view.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            # Start rotation on left click - store initial position
            state.drag_start_pos = event.pos
            return True
                
        elif event.type == MOUSEMOTION and state.drag_start_pos is not None:
            # Apply rotation based on mouse movement
            dx = event.pos[0] - state.drag_start_pos[0]
            dy = event.pos[1] - state.drag_start_pos[1]
            
            # Update rotation angles - movement speed factor of 0.01 radians per pixel
            state.view_3d_rotation_y += dx * 0.01  # Horizontal movement → Y-axis rotation
            state.view_3d_rotation_x += dy * 0.01  # Vertical movement → X-axis rotation
            
            # Save current position for next move
            state.drag_start_pos = event.pos
            return True
                
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            # End rotation
            state.drag_start_pos = None
            return True
                
        elif event.type == MOUSEWHEEL:
            # Apply zoom - adjust by 0.1 per scroll unit
            state.view_3d_zoom += event.y * 0.1
            # Clamp zoom to reasonable range (0.5x to 3.0x)
            state.view_3d_zoom = max(0.5, min(3.0, state.view_3d_zoom))
            return True
        
        return False
    
    @staticmethod
    def _handle_mesh_pin_mouse_event(event, state):
        """
        Handle mouse events related to mesh and pins.
        
        Manages pin selection, creation, and dragging to deform the mesh.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            return MouseController._handle_mouse_button_down(event, state)
            
        elif event.type == MOUSEMOTION and state.drag_index != -1:
            return MouseController._handle_mouse_drag(event, state)
                
        elif event.type == MOUSEBUTTONUP and event.button == 1 and state.drag_index != -1:
            return MouseController._handle_mouse_button_up(event, state)
        
        return False
    
    @staticmethod
    def _handle_mouse_button_down(event, state):
        """
        Handle mouse button down events for mesh interaction.
        
        Manages pin selection and creation based on the current mode.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True if event was handled
        """
        x, y = event.pos
        
        if state.mode == Mode.MOVE:
            # Check if clicking on an existing pin
            if MouseController._try_select_pin(x, y, state, True):
                return True
            
            # If no pin selected, create a new one
            add_custom_pin(x, y, state)
            
        elif state.mode == Mode.TOGGLE_PINS:
            # In toggle mode, only allow selecting pins (not creating)
            MouseController._try_select_pin(x, y, state, False)
        
        return True
    
    @staticmethod
    def _try_select_pin(x, y, state, check_landmarks):
        """
        Try to select a pin or landmark at the given position.
        
        Args:
            x, y: Mouse coordinates
            state: Application state
            check_landmarks: Whether to check landmarks in addition to custom pins
            
        Returns:
            bool: True if a pin was selected, False otherwise
        """
        # Calculate adaptive search radius for pin selection
        pin_selection_threshold_sq = PIN_SELECTION_THRESHOLD ** 2
        
        # Check for custom pins first
        for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
            if len(pin_data) >= 2:
                px, py = pin_data[0], pin_data[1]
                dx, dy = px - x, py - y
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < pin_selection_threshold_sq:
                    state.drag_index = i + len(state.landmark_positions)
                    state.drag_offset = (px - x, py - y)
                    return True
        
        # Check for landmarks if requested
        if check_landmarks:
            landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
            if not landmark_pins_hidden:
                for i, (lx, ly) in enumerate(state.landmark_positions):
                    dx, dy = lx - x, ly - y
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq < pin_selection_threshold_sq:
                        state.drag_index = i
                        state.drag_offset = (lx - x, ly - y)
                        return True
        
        return False
    
    @staticmethod
    def _handle_mouse_drag(event, state):
        """
        Handle mouse dragging for mesh manipulation.
        
        Applies mesh deformation based on pin movement.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True as event is handled
        """
        x, y = event.pos
        
        # Mark that pins have been moved (affects coloring)
        if not hasattr(state, 'pins_moved'):
            state.pins_moved = False
        state.pins_moved = True
        
        # Get original position and calculate delta
        orig_pos, new_pos, delta = MouseController._get_drag_positions(state, x, y)
        
        # Handle pin/landmark dragging based on mode
        if state.drag_index < len(state.landmark_positions):
            # Dragging a landmark
            MouseController._handle_landmark_drag(state, orig_pos, new_pos, delta)
        else:
            # Dragging a custom pin
            MouseController._handle_custom_pin_drag(state, orig_pos, new_pos, delta)
        
        return True
    
    @staticmethod
    def _get_drag_positions(state, mouse_x, mouse_y):
        """
        Calculate drag positions and delta for the current drag operation.
        
        Args:
            state: Application state
            mouse_x, mouse_y: Current mouse position
            
        Returns:
            tuple: (orig_pos, new_pos, delta) - original position, new position, and displacement
        """
        if state.drag_index < len(state.landmark_positions):
            # For landmarks
            orig_x, orig_y = state.landmark_positions[state.drag_index]
        else:
            # For custom pins
            pin_idx = state.drag_index - len(state.landmark_positions)
            pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
            orig_x, orig_y = pin_data[0], pin_data[1]
        
        # Calculate new position with offset
        new_x = mouse_x + state.drag_offset[0]
        new_y = mouse_y + state.drag_offset[1]
        
        # Calculate displacement
        dx = new_x - orig_x
        dy = new_y - orig_y
        
        return (orig_x, orig_y), (new_x, new_y), (dx, dy)
    
    @staticmethod
    def _handle_landmark_drag(state, orig_pos, new_pos, delta):
        """
        Handle dragging a landmark.
        
        Args:
            state: Application state
            orig_pos: Original position (x, y)
            new_pos: New position (x, y)
            delta: Displacement (dx, dy)
        """
        ox, oy = orig_pos
        nx, ny = new_pos
        dx, dy = delta
        
        if state.mode == Mode.TOGGLE_PINS:
            # Just move the landmark without affecting the mesh
            state.landmark_positions[state.drag_index] = (nx, ny)
        else:
            # Determine number of active pins for mode selection
            landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
            landmark_count = 0 if landmark_pins_hidden else len(state.landmark_positions)
            custom_pin_count = len(state.pins_per_image[state.current_image_idx])
            total_active_pins = landmark_count + custom_pin_count
            
            if total_active_pins >= 2:
                # Use rigid transformation with multiple pins
                transform_mesh_rigid(state, state.drag_index, ox, oy, nx, ny)
            else:
                # Single pin movement
                move_mesh_2d(state, ox, oy, dx, dy)
                update_all_landmarks(state)
                update_custom_pins(state)
    
    @staticmethod
    def _handle_custom_pin_drag(state, orig_pos, new_pos, delta):
        """
        Handle dragging a custom pin.
        
        Args:
            state: Application state
            orig_pos: Original position (x, y)
            new_pos: New position (x, y)
            delta: Displacement (dx, dy)
        """
        ox, oy = orig_pos
        nx, ny = new_pos
        dx, dy = delta
        
        # Get pin data
        pin_idx = state.drag_index - len(state.landmark_positions)
        pin_data = state.pins_per_image[state.current_image_idx][pin_idx]
        face_idx = pin_data[2]
        bc = pin_data[3]
        pin_pos_3d = pin_data[4] if len(pin_data) >= 5 else None
        
        if state.mode == Mode.TOGGLE_PINS:
            # Just move the pin without affecting the mesh
            if pin_pos_3d is not None:
                state.pins_per_image[state.current_image_idx][pin_idx] = (nx, ny, face_idx, bc, pin_pos_3d)
            else:
                state.pins_per_image[state.current_image_idx][pin_idx] = (nx, ny, face_idx, bc)
        else:
            # Determine transformation mode based on pin count
            landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
            landmark_count = 0 if landmark_pins_hidden else len(state.landmark_positions)
            custom_pin_count = len(state.pins_per_image[state.current_image_idx])
            total_active_pins = landmark_count + custom_pin_count
            
            if total_active_pins >= 2:
                # Use rigid transformation for multiple pins
                transform_mesh_rigid(state, state.drag_index, ox, oy, nx, ny)
            else:
                # Single pin movement
                move_mesh_2d(state, ox, oy, dx, dy)
                
                # Update landmarks and pins
                update_all_landmarks(state)
                update_custom_pins(state)
    
    @staticmethod
    def _handle_mouse_button_up(event, state):
        """
        Handle mouse button up after dragging.
        
        Finalizes the deformation and updates pins across all views.
        
        Args:
            event: Pygame event object
            state: Application state
            
        Returns:
            bool: True as event is handled
        """
        # Remember if we were dragging a landmark
        was_dragging_landmark = state.drag_index < len(state.landmark_positions)
        
        # Synchronize pins across all views for consistency
        synchronize_pins_across_views(state)
        
        # Make sure landmarks are still visible after synchronization
        if was_dragging_landmark:
            # Ensure landmark pins are visible
            if hasattr(state, 'landmark_pins_hidden'):
                state.landmark_pins_hidden = False
            
            # Force landmark update to ensure they're positioned correctly
            update_all_landmarks(state)
        
        # End dragging
        state.drag_index = -1
        return True