# Copyright (c) CUBOX, Inc. and its affiliates.
from enum import Enum

# UI Constants
WINDOW_NAME = "Face Builder"
UI_BUTTON_HEIGHT_RATIO = 0.05  # Button height as a ratio of image height
UI_BUTTON_MARGIN_RATIO = 0.01  # Margin between buttons as a ratio of image width
UI_TEXT_SIZE_RATIO = 0.00005    # Text size as a ratio of image diagonal
UI_LINE_THICKNESS_RATIO = 0.0000005  # Line thickness as a ratio of image diagonal

# Pin selection threshold - distance in pixels to select a pin instead of creating a new one
PIN_SELECTION_THRESHOLD = 15

# Operation modes
class Mode(Enum):
    MOVE = 0
    TOGGLE_PINS = 1  # Mode for toggling/selecting pins
    VIEW_3D = 2      # Mode for 3D visualization