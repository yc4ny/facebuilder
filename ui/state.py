# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2
import dlib
import mediapipe as mp
import pygame
from config import Mode, UI_BUTTON_HEIGHT_RATIO, UI_BUTTON_MARGIN_RATIO, UI_TEXT_SIZE_RATIO, UI_LINE_THICKNESS_RATIO
from ui.buttons import Button
from model.pins import synchronize_pins_across_views
from model.landmarks import update_all_landmarks, align_face
from model.mesh import project_current_3d_to_2d
from model.pins import center_geo, reset_shape, remove_pins, update_custom_pins
from utils.file_io import save_model

class FaceBuilderState:
    """Class to hold the state of the Face Builder application"""
    def __init__(self):
        # Operation modes
        self.mode = Mode.MOVE
        self.drag_index = -1
        self.drag_offset = (0, 0)
        self.custom_pins = []
        self.custom_pin_colors = []
        self.front_facing = None
        self.pins_moved = False  # Track if pins have been moved at least once
        
        # Face model variables
        self.landmark_positions = None
        self.landmark3d = None
        self.verts2d = None
        self.verts3d = None
        self.faces = None
        self.face_for_lmk = None
        self.lmk_b_coords = None
        self.overlay = None
        self.img_h = 0
        self.img_w = 0
        self.alpha = 1e-4
        self.weights = None
        
        # Default pose variables - never modified, used for resetting
        self.verts2d_default = None
        self.landmark_positions_default = None
        self.verts3d_default = None
        self.landmark3d_default = None
        
        # Multi-view support
        self.images = []
        self.pygame_surfaces = []  # To store Pygame surface versions of the images
        self.current_image_idx = 0
        self.pins_per_image = []
        self.camera_matrices = []
        self.rotations = []
        self.translations = []
        
        # 3D visualization state variables
        self.view_3d_rotation_x = np.pi
        self.view_3d_rotation_y = 0
        self.view_3d_zoom = 0.7
        self.drag_start_pos = None
        
        # Flag to track if we're in spherical rotation mode
        self.is_spherical_rotation = False
        
        # Initialize face detector and predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        # Try to load the shape predictor
        try:
            self.dlib_predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
            print("Loaded dlib shape predictor")
        except RuntimeError as e:
            print(f"Error loading dlib shape predictor: {e}")
            print("Please download the shape predictor from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract it to the data directory and restart the application.")
            exit(1)
        
        # Initialize MediaPipe for better face detection
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.3,
                model_selection=1
            )
            print("MediaPipe face detection initialized successfully")
        except ImportError:
            print("MediaPipe not found. Install with: pip install mediapipe")
            print("Continuing with dlib detector only")
            self.mp_face_detection = None
            self.mp_face_detector = None
        
        # UI dimensions and elements
        self.ui_dimensions = None
        self.buttons = []
        self.fonts = {}
        
        # Set up callbacks for UI interactions
        self.callbacks = {}
    
    def initialize_ui(self):
        """Initialize UI elements for Pygame"""
        # Calculate UI dimensions
        self.ui_dimensions = self.calculate_ui_dimensions()
        
        # Create fonts
        pygame.font.init()
        font_size = int(self.ui_dimensions['text_size'] * 24)
        self.fonts['button'] = pygame.font.SysFont('Arial', font_size)
        self.fonts['status'] = pygame.font.SysFont('Arial', font_size * 2)
        
        # Clear existing buttons
        self.buttons = []
        
        # Create buttons with their callbacks
        self.buttons.append(Button(
            self.ui_dimensions['center_geo_button_rect'], 
            "Center Geo", 
            self.fonts['button'],
            callback=self.center_geo
        ))
        
        self.buttons.append(Button(
            self.ui_dimensions['align_button_rect'], 
            "Align Face", 
            self.fonts['button'],
            callback=self.align_face
        ))
        
        self.buttons.append(Button(
            self.ui_dimensions['reset_shape_button_rect'], 
            "Reset Shape", 
            self.fonts['button'],
            callback=self.reset_shape
        ))
        
        self.buttons.append(Button(
            self.ui_dimensions['remove_pins_button_rect'], 
            "Remove Pins", 
            self.fonts['button'],
            callback=self.remove_pins
        ))
        
        toggle_button = Button(
            self.ui_dimensions['toggle_pins_button_rect'], 
            "Toggle Pins", 
            self.fonts['button'],
            callback=self.toggle_pins_mode,
            is_toggle=True
        )
        self.buttons.append(toggle_button)
        
        self.buttons.append(Button(
            self.ui_dimensions['save_button_rect'], 
            "Save Mesh", 
            self.fonts['button'],
            callback=self.save_model
        ))
        
        self.buttons.append(Button(
            self.ui_dimensions['next_img_button_rect'], 
            "Next Image", 
            self.fonts['button'],
            callback=self.next_image
        ))
        
        self.buttons.append(Button(
            self.ui_dimensions['prev_img_button_rect'], 
            "Prev Image", 
            self.fonts['button'],
            callback=self.prev_image
        ))
        
        visualizer_button = Button(
            self.ui_dimensions['visualizer_button_rect'], 
            "3D View", 
            self.fonts['button'],
            callback=self.toggle_3d_view_mode,
            is_toggle=True
        )
        self.buttons.append(visualizer_button)
    
    def calculate_ui_dimensions(self):
        """Calculate UI element dimensions based on image size"""
        # Calculate image diagonal for scale reference
        img_w, img_h = self.img_w, self.img_h
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
        
        # Calculate button positions - all in one row
        current_x = button_margin
        
        # Center Geo button (1st)
        center_geo_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Align Face button (2nd)
        align_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Reset Shape button (3rd)
        reset_shape_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Remove Pins button (4th)
        remove_pins_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Toggle Pins button (5th)
        toggle_pins_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Save Mesh button (6th)
        save_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Next Image button (7th)
        next_img_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # Prev Image button (8th)
        prev_img_button_rect = (current_x, button_margin, button_width, button_height)
        current_x += button_width + button_margin
        
        # 3D Visualizer button (9th)
        visualizer_button_rect = (current_x, button_margin, button_width, button_height)
        
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
            'visualizer_button_rect': visualizer_button_rect,
            'text_size': text_size,
            'text_thickness': text_thickness,
            'line_thickness': line_thickness,
            'vertex_radius': vertex_radius,
            'landmark_radius': landmark_radius,
            'pin_radius': pin_radius,
            'status_text_pos': (status_text_x, status_text_y1, status_text_y2)
        }
    
    def update_ui(self):
        """Update UI after changing image or mode"""
        self.ui_dimensions = self.calculate_ui_dimensions()
        self.initialize_ui()
        
        # Update button states based on current mode
        for button in self.buttons:
            if button.text == "Toggle Pins":
                button.is_active = (self.mode == Mode.TOGGLE_PINS)
            elif button.text == "3D View":
                button.is_active = (self.mode == Mode.VIEW_3D)
    
    def next_image(self, state=None):
        """Switch to the next image"""
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            # Convert OpenCV image to Pygame surface
            self.update_current_image()
            
            # Project the current 3D model to 2D for this view
            project_current_3d_to_2d(self)
            
            # Update landmarks for the new view
            update_all_landmarks(self)
            
            # Update UI dimensions for the new image size
            self.update_ui()
    
    def prev_image(self, state=None):
        """Switch to the previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            # Convert OpenCV image to Pygame surface
            self.update_current_image()
            
            # Project the current 3D model to 2D for this view
            project_current_3d_to_2d(self)
            
            # Update landmarks for the new view
            update_all_landmarks(self)
            
            # Update UI dimensions for the new image size
            self.update_ui()

    def update_current_image(self):
        """Update the current image and overlay surface"""
        self.overlay = self.images[self.current_image_idx].copy()
        self.img_h, self.img_w = self.overlay.shape[:2]
    
    def save_model(self, state=None):
        """Save the current 3D model to file"""
        # Make sure pins are up-to-date across all views before saving
        synchronize_pins_across_views(self)
        
        save_model(
            self.verts2d, 
            self.verts3d,
            self.faces, 
            self.pins_per_image, 
            self.camera_matrices, 
            self.rotations, 
            self.translations
        )
    
    def toggle_pins_mode(self, state=None):
        """Toggle between pin modes"""
        self.mode = Mode.TOGGLE_PINS if self.mode != Mode.TOGGLE_PINS else Mode.MOVE
        self.update_ui()
    
    def toggle_3d_view_mode(self, state=None):
        """Toggle 3D view mode"""
        self.mode = Mode.VIEW_3D if self.mode != Mode.VIEW_3D else Mode.MOVE
        
        # Initialize rotation angles if first time entering 3D mode
        if not hasattr(self, 'view_3d_rotation_x'):
            self.view_3d_rotation_x = 0.0
            self.view_3d_rotation_y = 0.0
        
        self.update_ui()
    
    def center_geo(self, state=None):
        """Reset the mesh to its default position"""
        center_geo(self)
        
    def align_face(self, state=None):
        """Align the face mesh to detected landmarks"""
        align_face(self)
        
    def reset_shape(self, state=None):
        """Reset the mesh shape while preserving position"""
        reset_shape(self)
        
    def remove_pins(self, state=None):
        """Remove custom pins from the current image"""
        remove_pins(self)
        
    def is_single_pin_active(self):
        """Check if only a single pin is active in the current view"""
        custom_pin_count = len(self.pins_per_image[self.current_image_idx])
        
        # Check if landmarks are hidden
        landmark_pins_hidden = hasattr(self, 'landmark_pins_hidden') and self.landmark_pins_hidden
        
        # Count total active pins
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(self.landmark_positions)
        
        # A single pin is active if there's exactly one pin total (custom or landmark)
        return total_active_pins == 1