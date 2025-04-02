# Copyright (c) CUBOX, Inc. and its affiliates.
"""
Application configuration module.

This module defines global constants and enumerations used throughout the
application to ensure consistent configuration.
"""
from enum import Enum

# UI Constants
WINDOW_NAME = "Face Builder"

# UI scaling constants
# These values define UI element sizes relative to the image dimensions
UI_BUTTON_HEIGHT_RATIO = 0.05  # Button height as a ratio of image height
UI_BUTTON_MARGIN_RATIO = 0.01  # Margin between buttons as a ratio of image width
UI_TEXT_SIZE_RATIO = 0.00005   # Text size as a ratio of image diagonal
UI_LINE_THICKNESS_RATIO = 0.0000005  # Line thickness as a ratio of image diagonal

# Interaction constants
PIN_SELECTION_THRESHOLD = 15   # Distance in pixels to select a pin instead of creating a new one

class Mode(Enum):
    """
    Application operation modes.
    
    Defines the different interaction modes that control how mouse
    events are interpreted and processed.
    """
    MOVE = 0        # Normal mesh manipulation mode
    TOGGLE_PINS = 1 # Mode for moving pins without affecting the mesh
    VIEW_3D = 2     # Mode for 3D visualization and rotation