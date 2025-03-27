# Copyright (c) CUBOX, Inc. and its affiliates.
import pickle
import numpy as np
import cv2
import dlib
import os
import mediapipe as mp
import pygame
import pygame.gfxdraw
from pygame.locals import *
from config import Mode
from utils.file_io import load_images_from_directory, save_model
from model.mesh import move_mesh_2d, project_current_3d_to_2d
from model.landmarks import update_all_landmarks
from model.pins import add_custom_pin, update_custom_pins, synchronize_pins_across_views

WINDOW_NAME = "Face Builder"
UI_BUTTON_HEIGHT_RATIO = 0.05
UI_BUTTON_MARGIN_RATIO = 0.01
UI_TEXT_SIZE_RATIO = 0.00005
UI_LINE_THICKNESS_RATIO = 0.0000005
PIN_SELECTION_THRESHOLD = 15

class Button:
    """Button class for Pygame UI"""
    def __init__(self, rect, text, font, callback=None, is_toggle=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.is_toggle = is_toggle
        self.is_active = False
        
    def draw(self, surface):
        # Draw button background
        color = (100, 100, 255) if self.is_active else (50, 50, 50)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 1)  # Border
        
        # Draw text centered
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event, state):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.callback:
                    self.callback(state)
                return True
        return False

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
        # Import function here to avoid circular imports
        from model.pins import center_geo
        center_geo(self)
        
    def align_face(self, state=None):
        """Align the face mesh to detected landmarks"""
        # Import function here to avoid circular imports
        from model.landmarks import align_face
        align_face(self)
        
    def reset_shape(self, state=None):
        """Reset the mesh shape while preserving position"""
        # Import function here to avoid circular imports
        from model.pins import reset_shape
        reset_shape(self)
        
    def remove_pins(self, state=None):
        """Remove custom pins from the current image"""
        from model.pins import remove_pins
        remove_pins(self)
    
    def handle_mouse_event(self, event):
        """Handle Pygame mouse events"""
        if self.mode == Mode.VIEW_3D:
            # Handle 3D view mode interactions
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                # Start rotation on left click
                self.drag_start_pos = event.pos
                return True
                
            elif event.type == MOUSEMOTION and self.drag_start_pos is not None:
                # Apply rotation based on mouse movement
                dx = event.pos[0] - self.drag_start_pos[0]
                dy = event.pos[1] - self.drag_start_pos[1]
                
                # Update rotation angles
                self.view_3d_rotation_y += dx * 0.01
                self.view_3d_rotation_x += dy * 0.01
                
                # Save current position for next move
                self.drag_start_pos = event.pos
                return True
                
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                # End rotation
                self.drag_start_pos = None
                return True
                
            elif event.type == MOUSEWHEEL:
                # Apply zoom
                self.view_3d_zoom += event.y * 0.1
                self.view_3d_zoom = max(0.5, min(3.0, self.view_3d_zoom))
                return True
                
        else:
            # Check button clicks first
            for button in self.buttons:
                if button.handle_event(event, self):
                    return True
            
            # Handle mesh and pin interactions
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if self.mode == Mode.MOVE:
                    # Calculate adaptive search radius for pin selection
                    pin_selection_threshold_sq = PIN_SELECTION_THRESHOLD ** 2
                    pin_radius_sq = (self.ui_dimensions['pin_radius'] * 1.5) ** 2
                    landmark_radius_sq = (self.ui_dimensions['landmark_radius'] * 1.5) ** 2
                    
                    # Check if we're clicking on a pin or landmark
                    found_pin_or_landmark = False
                    
                    # Check for custom pins first
                    for i, pin_data in enumerate(self.pins_per_image[self.current_image_idx]):
                        if len(pin_data) >= 2:
                            px, py = pin_data[0], pin_data[1]
                            dx, dy = px - x, py - y
                            dist_sq = dx*dx + dy*dy
                            
                            if dist_sq < pin_selection_threshold_sq:
                                self.drag_index = i + len(self.landmark_positions)
                                self.drag_offset = (px - x, py - y)
                                found_pin_or_landmark = True
                                break
                    
                    # Check for landmarks if no pin was found
                    landmark_pins_hidden = hasattr(self, 'landmark_pins_hidden') and self.landmark_pins_hidden
                    if not found_pin_or_landmark and not landmark_pins_hidden:
                        for i, (lx, ly) in enumerate(self.landmark_positions):
                            dx, dy = lx - x, ly - y
                            dist_sq = dx*dx + dy*dy
                            
                            if dist_sq < pin_selection_threshold_sq:
                                self.drag_index = i
                                self.drag_offset = (lx - x, ly - y)
                                found_pin_or_landmark = True
                                break
                    
                    # Create a new pin if no existing pin/landmark was found
                    if not found_pin_or_landmark:
                        add_custom_pin(x, y, self)
                
                elif self.mode == Mode.TOGGLE_PINS:
                    # Handle pin selection in toggle mode
                    pin_radius_sq = (self.ui_dimensions['pin_radius'] * 1.5) ** 2
                    
                    for i, pin_data in enumerate(self.pins_per_image[self.current_image_idx]):
                        if len(pin_data) >= 2:
                            px, py = pin_data[0], pin_data[1]
                            dx, dy = px - x, py - y
                            if dx*dx + dy*dy < pin_radius_sq:
                                self.drag_index = i + len(self.landmark_positions)
                                self.drag_offset = (px - x, py - y)
                                break
            
            elif event.type == MOUSEMOTION and self.drag_index != -1:
                # Handle dragging pins/landmarks
                x, y = event.pos
                
                if self.drag_index < len(self.landmark_positions) and self.mode != Mode.TOGGLE_PINS:
                    # Dragging a landmark
                    ox, oy = self.landmark_positions[self.drag_index]
                    nx = x + self.drag_offset[0]
                    ny = y + self.drag_offset[1]
                    dx, dy = nx - ox, ny - oy
                    
                    # Move mesh and update
                    move_mesh_2d(self, ox, oy, dx, dy)
                    update_all_landmarks(self)
                    update_custom_pins(self)
                else:
                    # Dragging a custom pin
                    pin_idx = self.drag_index - len(self.landmark_positions)
                    if pin_idx < len(self.pins_per_image[self.current_image_idx]):
                        pin_data = self.pins_per_image[self.current_image_idx][pin_idx]
                        
                        ox, oy = pin_data[0], pin_data[1]
                        face_idx = pin_data[2]
                        bc = pin_data[3]
                        
                        pin_pos_3d = None
                        if len(pin_data) >= 5:
                            pin_pos_3d = pin_data[4]
                        
                        nx = x + self.drag_offset[0]
                        ny = y + self.drag_offset[1]
                        
                        if self.mode == Mode.TOGGLE_PINS:
                            # Just move the pin without affecting the mesh
                            if pin_pos_3d is not None:
                                self.pins_per_image[self.current_image_idx][pin_idx] = (nx, ny, face_idx, bc, pin_pos_3d)
                            else:
                                self.pins_per_image[self.current_image_idx][pin_idx] = (nx, ny, face_idx, bc)
                        else:
                            # Check for multiple pins for rigid transformation
                            pins = self.pins_per_image[self.current_image_idx]
                            if len(pins) >= 2:
                                # Use rigid transformation
                                from model.mesh import transform_mesh_rigid
                                transform_mesh_rigid(self, pin_idx, ox, oy, nx, ny)
                            else:
                                # Use regular deformation
                                dx, dy = nx - ox, ny - oy
                                move_mesh_2d(self, ox, oy, dx, dy)
                                
                                # Update landmarks and pins
                                update_all_landmarks(self)
                                update_custom_pins(self)
                
                return True
                
            elif event.type == MOUSEBUTTONUP and event.button == 1 and self.drag_index != -1:
                # Finish dragging
                if hasattr(self.callbacks, 'synchronize_pins') and callable(self.callbacks['synchronize_pins']):
                    self.callbacks['synchronize_pins'](self)
                else:
                    synchronize_pins_across_views(self)
                
                self.drag_index = -1
                return True
        
        return False
    
    def is_single_pin_active(self):
        """Check if only a single pin is active in the current view"""
        custom_pin_count = len(self.pins_per_image[self.current_image_idx])
        
        # Check if landmarks are hidden
        landmark_pins_hidden = hasattr(self, 'landmark_pins_hidden') and self.landmark_pins_hidden
        
        # Count total active pins
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(self.landmark_positions)
        
        # A single pin is active if there's exactly one pin total
        return total_active_pins == 1

def cv2_to_pygame(cv2_image):
    """Convert OpenCV image (BGR) to Pygame surface (RGB)"""
    # Convert color format from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Create Pygame surface from the RGB image
    pygame_surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
    
    return pygame_surface

def draw_mesh_pygame(surface, state):
    """Draw the mesh and UI using Pygame"""
    # Clear surface with the background image
    surface.blit(cv2_to_pygame(state.overlay), (0, 0))
    
    # Calculate UI dimensions
    ui = state.ui_dimensions
    
    # Check if we're in 3D view mode
    if state.mode == Mode.VIEW_3D:
        draw_3d_view_pygame(surface, state)
        return
    
    # Calculate which faces are front-facing using the current camera parameters
    from utils.geometry import calculate_front_facing
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    state.front_facing = calculate_front_facing(
        state.verts3d, state.faces, 
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Draw mesh triangles (only front-facing)
    for i, (i0, i1, i2) in enumerate(state.faces):
        if state.front_facing[i]:
            p0 = (int(round(state.verts2d[i0][0])), int(round(state.verts2d[i0][1])))
            p1 = (int(round(state.verts2d[i1][0])), int(round(state.verts2d[i1][1])))
            p2 = (int(round(state.verts2d[i2][0])), int(round(state.verts2d[i2][1])))
            
            # Draw lines using pygame
            pygame.draw.line(surface, (0, 0, 0), p0, p1, ui['line_thickness'])
            pygame.draw.line(surface, (0, 0, 0), p1, p2, ui['line_thickness'])
            pygame.draw.line(surface, (0, 0, 0), p2, p0, ui['line_thickness'])
    
    # Draw vertices (only those used by front-facing faces)
    visible_vertices = set()
    for i, face in enumerate(state.faces):
        if state.front_facing[i]:
            visible_vertices.update(face)
    
    for i in visible_vertices:
        x, y = state.verts2d[i]
        pygame.draw.circle(surface, (0, 0, 255), 
                          (int(round(x)), int(round(y))), 
                          ui['vertex_radius'])
    
    # Draw landmarks
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    if not landmark_pins_hidden:
        for (lx, ly) in state.landmark_positions:
            pygame.draw.circle(surface, (255, 0, 0), 
                              (int(round(lx)), int(round(ly))), 
                              ui['landmark_radius'])
    
    # Draw custom pins for the current image
    for pin_data in state.pins_per_image[state.current_image_idx]:
        if len(pin_data) >= 2:
            px, py = pin_data[0], pin_data[1]
            pygame.draw.circle(surface, (0, 255, 0), 
                              (int(round(px)), int(round(py))), 
                              ui['pin_radius'])
    
    # Draw all buttons
    for button in state.buttons:
        button.draw(surface)
    
    # Show current image index
    status_x, status_y1, status_y2 = ui['status_text_pos']
    img_text = f"Image {state.current_image_idx+1}/{len(state.images)}"
    img_text_surf = state.fonts['status'].render(img_text, True, (255, 120, 0))
    surface.blit(img_text_surf, (status_x, status_y1))
    
    # Show current mode
    if state.mode == Mode.TOGGLE_PINS:
        mode_text = "Mode: TOGGLE PINS"
    else:
        # Count active pins to determine mode
        custom_pin_count = len(state.pins_per_image[state.current_image_idx])
        landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
        
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(state.landmark_positions)
        
        if total_active_pins == 1:
            mode_text = "Mode: SINGLE PIN (MOVE+ROTATE)"
        else:
            mode_text = "Mode: MULTI-PIN DEFORM"
    
    mode_text_surf = state.fonts['status'].render(mode_text, True, (255, 120, 0))
    surface.blit(mode_text_surf, (status_x, status_y2))
    
    # Add a user hint for adding pins
    hint_y = status_y2 + int(ui['landmark_radius'] * 8)
    hint_text = "Click on mesh to add pins"
    hint_text_surf = state.fonts['button'].render(hint_text, True, (0, 200, 0))
    surface.blit(hint_text_surf, (status_x, hint_y))

def draw_3d_view_pygame(surface, state):
    """Draw 3D visualization of the mesh using Pygame"""
    ui = state.ui_dimensions
    
    # Create a gradient background
    for y in range(state.img_h):
        blue_val = int(220 - 50 * (y / state.img_h))
        pygame.draw.line(surface, (240, 240, blue_val), (0, y), (state.img_w, y))
    
    # Set up camera parameters
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create rotation matrices for x and y axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(state.view_3d_rotation_x), -np.sin(state.view_3d_rotation_x)],
        [0, np.sin(state.view_3d_rotation_x), np.cos(state.view_3d_rotation_x)]
    ])
    
    Ry = np.array([
        [np.cos(state.view_3d_rotation_y), 0, np.sin(state.view_3d_rotation_y)],
        [0, 1, 0],
        [-np.sin(state.view_3d_rotation_y), 0, np.cos(state.view_3d_rotation_y)]
    ])
    
    # Combined rotation matrix
    R = Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    
    # Center the model and scale it appropriately
    verts_center = np.mean(state.verts3d, axis=0)
    verts_centered = state.verts3d - verts_center
    
    # Calculate appropriate scale to fit in window
    base_scale = min(state.img_w, state.img_h) * 0.3 / np.max(np.abs(verts_centered))
    zoom_scale = base_scale * state.view_3d_zoom
    verts_scaled = verts_centered * zoom_scale
    
    # Position the model in front of the camera
    tvec = np.array([[0, 0, max(state.img_w, state.img_h) * 0.7]], dtype=np.float32).T
    
    # Project the 3D vertices to 2D
    projected_verts, _ = cv2.projectPoints(
        verts_scaled, rvec, tvec, camera_matrix, np.zeros((4, 1))
    )
    projected_verts = projected_verts.reshape(-1, 2)
    
    # Calculate which faces are front-facing
    from utils.geometry import calculate_front_facing
    state.front_facing = calculate_front_facing(
        verts_scaled, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Sort faces by depth for better rendering
    face_depths = []
    for i, (i0, i1, i2) in enumerate(state.faces):
        # Only add front-facing faces to the render list
        if state.front_facing[i]:
            # Calculate depth of face center in camera space
            center_3d = (verts_scaled[i0] + verts_scaled[i1] + verts_scaled[i2]) / 3
            center_rotated = R @ center_3d
            depth = center_rotated[2] + tvec[2, 0]  # Z in camera space
            face_depths.append((i, depth))
    
    # Sort faces back-to-front (only front-facing faces)
    sorted_faces = [idx for idx, _ in sorted(face_depths, key=lambda x: x[1], reverse=True)]
    
    # Draw the mesh triangles with filled polygons
    for face_idx in sorted_faces:
        i0, i1, i2 = state.faces[face_idx]
        p0 = (int(round(projected_verts[i0][0])), int(round(projected_verts[i0][1])))
        p1 = (int(round(projected_verts[i1][0])), int(round(projected_verts[i1][1])))
        p2 = (int(round(projected_verts[i2][0])), int(round(projected_verts[i2][1])))
        
        # Calculate face normal for shading
        v0 = verts_scaled[i0]
        v1 = verts_scaled[i1]
        v2 = verts_scaled[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)  # Normalize
        
        # Direction to camera (for lighting calculation)
        view_dir = np.array([0, 0, 1])
        
        # Calculate diffuse lighting
        light_intensity = np.dot(normal, view_dir)
        
        # Add ambient lighting to ensure all faces are visible
        light_intensity = max(0.3, min(0.95, light_intensity + 0.6))
        
        # Silver/metallic color for the mesh
        base_color = np.array([192, 192, 192])  # Silver gray
        
        # Apply lighting
        colored = base_color * light_intensity
        color = (int(colored[0]), int(colored[1]), int(colored[2]))
        
        # Fill the triangle with this color
        pygame.draw.polygon(surface, color, [p0, p1, p2])
        
        # Draw edges of triangles
        pygame.draw.line(surface, (80, 80, 80), p0, p1, ui['line_thickness'])
        pygame.draw.line(surface, (80, 80, 80), p1, p2, ui['line_thickness'])
        pygame.draw.line(surface, (80, 80, 80), p2, p0, ui['line_thickness'])
    
    # Draw all buttons
    for button in state.buttons:
        button.draw(surface)
    
    # Add instruction text for 3D view
    instruction_y = ui['landmark_radius'] * 6 + ui['center_geo_button_rect'][1] + ui['center_geo_button_rect'][3] + 10
    instruction_text = "DRAG to rotate | SCROLL to zoom"
    instruction_surf = state.fonts['button'].render(instruction_text, True, (0, 0, 0))
    surface.blit(instruction_surf, (state.img_w - 350, instruction_y))
    
    view_mode_text = "3D VIEW MODE"
    view_mode_surf = state.fonts['status'].render(view_mode_text, True, (0, 0, 120))
    view_mode_rect = view_mode_surf.get_rect(center=(state.img_w // 2, state.img_h - 50))
    surface.blit(view_mode_surf, view_mode_rect)
    
def main():
    # Initialize Pygame
    pygame.init()
    
    # Initialize the state
    state = FaceBuilderState()
    
    print("Loading FLAME model data...")
    try:
        # Load FLAME model data
        with open("data/flame2023.pkl", "rb") as f:
            flame_data = pickle.load(f, encoding='latin1')
        verts_3d = flame_data['v_template']
        faces_ = flame_data['f']
        weights_ = flame_data['weights']
        
        # Save original face indices before removing eyeballs
        original_faces = faces_.copy()
        
        # Keep a record of which faces don't use eyeball vertices
        valid_face_mask = np.all((faces_ < 3931) | (faces_ > 5022), axis=1)
        valid_face_indices_original = np.where(valid_face_mask)[0]
        
        # Remove eyeball vertices
        print("Removing eyeball vertices...")
        from utils.geometry import remove_eyeballs
        verts_3d, faces_ = remove_eyeballs(verts_3d, faces_, 3931, 5022)
        
        # Direct mapping from original valid face indices to new face indices
        old_to_new_face = {idx: i for i, idx in enumerate(valid_face_indices_original)}
        
        # Update weights
        keep_mask = np.ones(flame_data['weights'].shape[0], dtype=bool)
        keep_mask[3931:5023] = False
        weights_ = weights_[keep_mask]
        
        state.faces = np.array(faces_)
        state.weights = weights_
        
        # Load landmark embedding data
        with open("data/flame_static_embedding.pkl", "rb") as f:
            emb_data = pickle.load(f, encoding='latin1')
        f_idx = emb_data["lmk_face_idx"]
        b_coords = emb_data["lmk_b_coords"]
        
        # Filter landmarks based on faces that still exist
        valid_landmarks = []
        valid_face_indices = []
        valid_barycentric_coords = []
        
        for i, (face_idx, bc) in enumerate(zip(f_idx, b_coords)):
            if face_idx in old_to_new_face:
                valid_landmarks.append(i)
                valid_face_indices.append(old_to_new_face[face_idx])
                valid_barycentric_coords.append(bc)
        
        state.lmk_b_coords = np.array(valid_barycentric_coords)
        state.face_for_lmk = np.array(valid_face_indices)
        
        print(f"Retained {len(valid_landmarks)} landmarks after eyeball removal")
        print("Model data loaded successfully!")
    except Exception as e:
        print(f"Error loading model data: {e}")
        return
    
    # Load all images from the images directory
    print("Loading images...")
    state.images = load_images_from_directory()
    state.current_image_idx = 0
    state.overlay = state.images[state.current_image_idx].copy()
    state.img_h, state.img_w = state.overlay.shape[:2]
    print(f"Current image size: {state.img_w}x{state.img_h}")
    
    # Initialize pins, camera matrices, rotations, and translations for each image
    state.pins_per_image = [[] for _ in range(len(state.images))]
    state.camera_matrices = [None for _ in range(len(state.images))]
    state.rotations = [None for _ in range(len(state.images))]
    state.translations = [None for _ in range(len(state.images))]
    
    # Calculate initial camera parameters for perspective projection
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Calculate center of the model
    center = np.mean(verts_3d, axis=0)

    # Initialize rotation (identity rotation)
    R = np.eye(3, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)

    # Position the model in front of the camera
    distance = -0.5 
    tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)

    # Set up initial camera parameters for the current view
    state.camera_matrices[state.current_image_idx] = camera_matrix
    state.rotations[state.current_image_idx] = rvec
    state.translations[state.current_image_idx] = tvec

    # Calculate 3D landmark positions based on the barycentric coordinates
    lmk_3d = []
    for face_idx, bc in zip(state.face_for_lmk, state.lmk_b_coords):
        i0, i1, i2 = state.faces[face_idx]
        v0 = verts_3d[i0]
        v1 = verts_3d[i1]
        v2 = verts_3d[i2]
        xyz = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        lmk_3d.append(xyz)

    # Initialize 3D landmarks
    state.landmark3d = np.array(lmk_3d)

    # Project 3D to 2D using perspective projection
    try:
        projected_verts, _ = cv2.projectPoints(
            verts_3d, rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        state.verts2d = projected_verts.reshape(-1, 2)
        
        # Project 3D landmarks to 2D
        projected_landmarks, _ = cv2.projectPoints(
            np.array(lmk_3d, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        state.landmark_positions = projected_landmarks.reshape(-1, 2)
    except Exception as e:
        print(f"Error initializing projection: {e}")
        # Provide default values in case of projection failure
        state.verts2d = np.zeros((len(verts_3d), 2))
        state.landmark_positions = np.zeros((len(lmk_3d), 2))
    
    # Initialize the 3D vertices that will be modified by user interactions
    state.verts3d = verts_3d.copy()
    
    # Store default positions - these are never modified, used for resetting
    state.verts2d_default = state.verts2d.copy()
    state.landmark_positions_default = state.landmark_positions.copy()
    state.verts3d_default = verts_3d.copy()
    state.landmark3d_default = np.array(lmk_3d).copy()
    
    # Setup callbacks for circular dependencies
    state.callbacks = {
        'synchronize_pins': synchronize_pins_across_views,
        'update_landmarks': update_all_landmarks,
        'project_2d': project_current_3d_to_2d
    }
    
    # Initialize UI
    state.initialize_ui()
    
    # Set up Pygame window
    pygame.display.set_caption(WINDOW_NAME)
    screen = pygame.display.set_mode((state.img_w, state.img_h), pygame.RESIZABLE)
    
    print("\nEnhanced Face Builder Controls:")
    print("- Click 'Align Face' to automatically detect and align the 3D mesh to the face")
    print("- Click directly on the mesh to add pins automatically")
    print("- Click 'Toggle Pins' to move pins without affecting the mesh")
    print("- Click 'Remove Pins' to clear all custom pins from the current image")
    print("- Click 'Center Geo' to reset the mesh to its default position")
    print("- Use 'Next Image' and 'Prev Image' to switch between input images")
    print("- Click 'Save' to save the current model to file")
    print("- Click '3D View' to switch to interactive 3D visualization mode")
    print("- In 3D View mode, drag with the mouse to rotate the model")
    print("- Drag landmarks or custom pins to manipulate the 3D mesh")
    print("- When using a single pin, the mesh will move with the pin and rotate based on position")
    print("- When dragged to left/right edge, the head rotates to face the opposite direction")
    print("- Press ESC to exit\n")
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
            elif event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL):
                state.handle_mouse_event(event)
        
        # Clear screen and draw mesh
        draw_mesh_pygame(screen, state)
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()