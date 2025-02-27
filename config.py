# Copyright (c) CUBOX, Inc. and its affiliates.
from enum import Enum

# UI Constants
WINDOW_NAME = "Face Builder"
UI_BUTTON_HEIGHT_RATIO = 0.05  # Button height as a ratio of image height
UI_BUTTON_MARGIN_RATIO = 0.01  # Margin between buttons as a ratio of image width
UI_TEXT_SIZE_RATIO = 0.00005    # Text size as a ratio of image diagonal
UI_LINE_THICKNESS_RATIO = 0.0000005  # Line thickness as a ratio of image diagonal

# Operation modes
class Mode(Enum):
    MOVE = 0
    ADD_PIN = 1
    TOGGLE_PINS = 2  # Mode for toggling/selecting pins