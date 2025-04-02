"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Application state module.

This module defines the FaceBuilderState class which serves as the central
repository for all application data and state management.
"""
import numpy as np
import cv2
import dlib
import mediapipe as mp
import pygame
from config import Mode, UI_BUTTON_HEIGHT_RATIO, UI_BUTTON_MARGIN_RATIO, UI_TEXT_SIZE_RATIO, UI_LINE_THICKNESS_RATIO
from ui.buttons import Button

class FaceBuilderState:
    """
    Application state container class.
    
    This class maintains all data related to the application state, including:
    - 3D model vertices, faces, and landmarks
    - Camera parameters and transformations
    - UI elements and configuration
    - Current view and interaction mode
    
    It also provides methods for common state operations like switching images,
    updating UI, and handling mode changes.
    """
    
    def __init__(self):
        """
        Initialize a new application state.
        
        Creates a new state with default values for all properties.
        External initialization is required for mesh data and images.
        """
        # Operation modes
        self.mode = Mode.MOVE           # Current operation mode
        self.drag_index = -1            # Index of pin being dragged (-1 = none)
        self.drag_offset = (0, 0)       # Offset from mouse to pin position
        self.custom_pins = []           # List of custom (user-added) pins
        self.custom_pin_colors = []     # Color for each custom pin
        self.front_facing = None        # Boolean array indicating which faces are front-facing
        self.pins_moved = False         # Track if pins have been moved (affects coloring)
        
        # Face model variables
        self.landmark_positions = None  # 2D positions of facial landmarks
        self.landmark3d = None          # 3D positions of facial landmarks
        self.verts2d = None             # 2D positions of mesh vertices
        self.verts3d = None             # 3D positions of mesh vertices
        self.faces = None               # Face indices (triangles)
        self.face_for_lmk = None        # Face index for each landmark
        self.lmk_b_coords = None        # Barycentric coordinates for landmarks
        self.overlay = None             # Current image being processed
        self.img_h = 0                  # Height of current image
        self.img_w = 0                  # Width of current image
        self.alpha = 1e-4               # Weight parameter for deformation
        self.weights = None             # Skinning weights for deformation
        
        # Default pose variables - never modified, used for resetting
        self.verts2d_default = None     # Default 2D vertex positions
        self.landmark_positions_default = None  # Default 2D landmark positions
        self.verts3d_default = None     # Default 3D vertex positions
        self.landmark3d_default = None  # Default 3D landmark positions
        
        # Multi-view support
        self.images = []                # List of input images
        self.pygame_surfaces = []       # Pygame surfaces for each image
        self.current_image_idx = 0      # Index of currently active image
        self.pins_per_image = []        # Custom pins for each image
        self.camera_matrices = []       # Camera intrinsic matrix for each image
        self.rotations = []             # Rotation vectors for each image
        self.translations = []          # Translation vectors for each image
        
        # 3D visualization state variables
        self.view_3d_rotation_x = np.pi # Rotation angle around X axis
        self.view_3d_rotation_y = 0     # Rotation angle around Y axis
        self.view_3d_zoom = 0.7         # Zoom level for 3D view
        self.drag_start_pos = None      # Starting position for 3D view drag
        
        # Face detection components
        self._initialize_face_detection()
        
        # UI elements and dimensions
        self.ui_dimensions = None       # Calculated dimensions for UI elements
        self.buttons = []               # List of UI buttons
        self.fonts = {}                 # Dictionary of fonts for UI rendering
        
        # Callbacks for circular dependencies
        self.callbacks = {}             # Functions to handle specific operations
    
    def _initialize_face_detection(self):
        """
        Initialize face detection and landmark detection components.
        
        Sets up dlib face detector and shape predictor, and MediaPipe face
        detection if available. These are used for aligning the mesh to faces.
        """
        # Initialize face detector
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        # Try to load the shape predictor for facial landmarks
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
                model_selection=1  # Use full range detector
            )
            print("MediaPipe face detection initialized successfully")
        except ImportError:
            print("MediaPipe not found. Install with: pip install mediapipe")
            print("Continuing with dlib detector only")
            self.mp_face_detection = None
            self.mp_face_detector = None
    
    def initialize_ui(self):
        """
        Initialize UI elements for Pygame.
        
        Calculates UI dimensions, creates fonts, and initializes buttons
        with appropriate positions and sizes.
        """
        # Calculate UI dimensions based on current image size
        self.ui_dimensions = self.calculate_ui_dimensions()
        
        # Initialize Pygame fonts
        pygame.font.init()
        font_size = int(self.ui_dimensions['text_size'] * 24)
        self.fonts['button'] = pygame.font.SysFont('Arial', font_size)
        self.fonts['status'] = pygame.font.SysFont('Arial', font_size * 2)
        
        # Clear existing buttons
        self.buttons = []
        
        # Create buttons with their callbacks
        self._initialize_buttons()
    
    def _initialize_buttons(self):
        """
        Create all UI buttons with default parameters.
        
        The callbacks for these buttons will be set later by the ButtonController.
        """
        ui = self.ui_dimensions
        
        # Center Geometry button
        self.buttons.append(Button(
            ui['center_geo_button_rect'], 
            "Center Geo", 
            self.fonts['button'],
            callback=self.center_geo
        ))
        
        # Align Face button
        self.buttons.append(Button(
            ui['align_button_rect'], 
            "Align Face", 
            self.fonts['button'],
            callback=self.align_face
        ))
        
        # Reset Shape button
        self.buttons.append(Button(
            ui['reset_shape_button_rect'], 
            "Reset Shape", 
            self.fonts['button'],
            callback=self.reset_shape
        ))
        
        # Remove Pins button
        self.buttons.append(Button(
            ui['remove_pins_button_rect'], 
            "Remove Pins", 
            self.fonts['button'],
            callback=self.remove_pins
        ))
        
        # Toggle Pins button (toggle button)
        toggle_button = Button(
            ui['toggle_pins_button_rect'], 
            "Toggle Pins", 
            self.fonts['button'],
            callback=self.toggle_pins_mode,
            is_toggle=True
        )
        self.buttons.append(toggle_button)
        
        # Save Mesh button
        self.buttons.append(Button(
            ui['save_button_rect'], 
            "Save Mesh", 
            self.fonts['button'],
            callback=self.save_model
        ))
        
        # Next Image button
        self.buttons.append(Button(
            ui['next_img_button_rect'], 
            "Next Image", 
            self.fonts['button'],
            callback=self.next_image
        ))
        
        # Previous Image button
        self.buttons.append(Button(
            ui['prev_img_button_rect'], 
            "Prev Image", 
            self.fonts['button'],
            callback=self.prev_image
        ))
        
        # 3D View button (toggle button)
        visualizer_button = Button(
            ui['visualizer_button_rect'], 
            "3D View", 
            self.fonts['button'],
            callback=self.toggle_3d_view_mode,
            is_toggle=True
        )
        self.buttons.append(visualizer_button)
    
    def calculate_ui_dimensions(self):
        """
        Calculate UI element dimensions based on image size.
        
        This method computes the size and position of all UI elements
        relative to the current image dimensions to ensure responsive layout.
        
        Returns:
            dict: Dictionary of UI element dimensions and positions
        """
        # Calculate image diagonal for scale reference
        img_w, img_h = self.img_w, self.img_h
        img_diag = np.sqrt(img_w**2 + img_h**2)
        
        # Button dimensions
        button_height = int(img_h * UI_BUTTON_HEIGHT_RATIO)
        button_margin = int(img_w * UI_BUTTON_MARGIN_RATIO)
        
        # Calculate sizes based on image diagonal
        text_size = max(0.4, img_diag * UI_TEXT_SIZE_RATIO * 0.6)
        text_thickness = max(1, int(img_diag * 0.0005))
        line_thickness = max(1, int(img_diag * UI_LINE_THICKNESS_RATIO))
        vertex_radius = max(1, int(img_diag * 0.0001))
        landmark_radius = max(3, int(img_diag * 0.002))
        pin_radius = max(3, int(img_diag * 0.002))
        
        # Standard button width
        button_width = int(button_height * 1.8)
        
        # Calculate button positions - all in one row
        current_x = button_margin
        
        # Position each button sequentially
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
        
        # Return dictionary of all UI dimensions
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
        """
        Update UI after changing image or mode.
        
        Recalculates UI dimensions based on the current image size and
        updates button states to reflect the current application mode.
        """
        # Recalculate UI dimensions for new image
        self.ui_dimensions = self.calculate_ui_dimensions()
        
        # Reinitialize all UI elements
        self.initialize_ui()
        
        # Update button states based on current mode
        for button in self.buttons:
            if button.text == "Toggle Pins":
                button.is_active = (self.mode == Mode.TOGGLE_PINS)
            elif button.text == "3D View":
                button.is_active = (self.mode == Mode.VIEW_3D)
    
    def next_image(self, state=None):
        """
        Switch to the next image in the sequence.
        
        Updates current image index, loads the image, and ensures
        the mesh is properly projected for the new view.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        if self.current_image_idx < len(self.images) - 1:
            # Increment image index
            self.current_image_idx += 1
            
            # Update the current image
            self.update_current_image()
            
            # Project the current 3D model to 2D for this view
            # Import here to avoid circular import
            from model.mesh import project_current_3d_to_2d
            project_current_3d_to_2d(self)
            
            # Update landmarks for the new view
            # Import here to avoid circular import
            from model.landmarks import update_all_landmarks
            update_all_landmarks(self)
            
            # Update UI dimensions for the new image size
            self.update_ui()
    
    def prev_image(self, state=None):
        """
        Switch to the previous image in the sequence.
        
        Updates current image index, loads the image, and ensures
        the mesh is properly projected for the new view.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        if self.current_image_idx > 0:
            # Decrement image index
            self.current_image_idx -= 1
            
            # Update the current image
            self.update_current_image()
            
            # Project the current 3D model to 2D for this view
            # Import here to avoid circular import
            from model.mesh import project_current_3d_to_2d
            project_current_3d_to_2d(self)
            
            # Update landmarks for the new view
            # Import here to avoid circular import
            from model.landmarks import update_all_landmarks
            update_all_landmarks(self)
            
            # Update UI dimensions for the new image size
            self.update_ui()

    def update_current_image(self):
        """
        Update the current image and overlay surface.
        
        Sets the current overlay image based on the current image index
        and updates image dimensions.
        """
        # Copy the current image to the overlay
        self.overlay = self.images[self.current_image_idx].copy()
        
        # Update image dimensions
        self.img_h, self.img_w = self.overlay.shape[:2]
    
    def save_model(self, state=None):
        """
        Save the current 3D model to file.
        
        Synchronizes pins across all views before saving to ensure
        consistency, then saves the model data to disk.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Make sure pins are up-to-date across all views before saving
        # Import here to avoid circular import
        from model.pins import synchronize_pins_across_views
        synchronize_pins_across_views(self)
        
        # Import here to avoid circular import
        from utils.file_io import save_model
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
        """
        Toggle between pin manipulation modes.
        
        Switches between normal mesh manipulation mode and pin toggle mode,
        which allows moving pins without affecting the mesh.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Toggle between MOVE and TOGGLE_PINS modes
        self.mode = Mode.TOGGLE_PINS if self.mode != Mode.TOGGLE_PINS else Mode.MOVE
        
        # Update UI to reflect mode change
        self.update_ui()
    
    def toggle_3d_view_mode(self, state=None):
        """
        Toggle 3D view mode.
        
        Switches between normal mesh editing mode and 3D visualization mode,
        initializing rotation angles if entering 3D mode for the first time.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Toggle between MOVE and VIEW_3D modes
        self.mode = Mode.VIEW_3D if self.mode != Mode.VIEW_3D else Mode.MOVE
        
        # Initialize rotation angles if first time entering 3D mode
        if not hasattr(self, 'view_3d_rotation_x'):
            self.view_3d_rotation_x = 0.0
            self.view_3d_rotation_y = 0.0
        
        # Update UI to reflect mode change
        self.update_ui()
    
    def center_geo(self, state=None):
        """
        Reset the mesh to its default position.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Import here to avoid circular import
        from model.pins import center_geo
        center_geo(self)
        
    def align_face(self, state=None):
        """
        Align the face mesh to detected landmarks.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Import here to avoid circular import
        from model.landmarks import align_face
        align_face(self)
        
    def reset_shape(self, state=None):
        """
        Reset the mesh shape while preserving position.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Import here to avoid circular import
        from model.pins import reset_shape
        reset_shape(self)
        
    def remove_pins(self, state=None):
        """
        Remove custom pins from the current image.
        
        Args:
            state: Optional state parameter for button callback compatibility
        """
        # Import here to avoid circular import
        from model.pins import remove_pins
        remove_pins(self)
        
    def is_single_pin_active(self):
        """
        Check if only a single pin is active in the current view.
        
        This is used to determine the appropriate manipulation mode
        based on the number of active pins.
        
        Returns:
            bool: True if exactly one pin is active, False otherwise
        """
        # Count custom pins
        custom_pin_count = len(self.pins_per_image[self.current_image_idx])
        
        # Check if landmarks are hidden
        landmark_pins_hidden = hasattr(self, 'landmark_pins_hidden') and self.landmark_pins_hidden
        
        # Count total active pins
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(self.landmark_positions)
        
        # A single pin is active if there's exactly one pin total (custom or landmark)
        return total_active_pins == 1