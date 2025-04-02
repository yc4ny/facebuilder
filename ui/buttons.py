"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Button UI component module.

This module defines the Button class for handling interactive UI elements.
"""
import pygame
from pygame.locals import *

class Button:
    """
    Button class for Pygame UI.
    
    Handles rendering, state management, and event handling for interactive
    button elements in the UI.
    
    Attributes:
        rect: Rectangle defining button position and size
        text: Button label text
        font: Pygame font for rendering text
        callback: Function to call when button is clicked
        is_toggle: Whether button is a toggle button that maintains state
        is_active: Current state for toggle buttons
    """
    def __init__(self, rect, text, font, callback=None, is_toggle=False):
        """
        Initialize a new Button.
        
        Args:
            rect: Tuple (x, y, width, height) defining button rectangle
            text: Button label text
            font: Pygame font for rendering text
            callback: Function to call when button is clicked (optional)
            is_toggle: Whether button is a toggle button (optional)
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.is_toggle = is_toggle
        self.is_active = False
        
    def draw(self, surface):
        """
        Draw the button on the given surface.
        
        Renders the button with appropriate colors based on its state,
        and centers the text within the button rectangle.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Choose button background color based on state
        color = (100, 100, 255) if self.is_active else (50, 50, 50)
        
        # Draw button background and border
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 1)  # Border
        
        # Draw text centered within button
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event, state):
        """
        Handle mouse events for this button.
        
        Checks if the button was clicked and triggers the callback function
        if one is assigned. Toggle buttons also update their state.
        
        Args:
            event: Pygame event to handle
            state: Application state to pass to the callback
            
        Returns:
            bool: True if button was clicked, False otherwise
        """
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # Toggle state if this is a toggle button
                if self.is_toggle:
                    self.is_active = not self.is_active
                
                # Call the callback function if assigned
                if self.callback:
                    self.callback(state)
                    
                return True
        return False