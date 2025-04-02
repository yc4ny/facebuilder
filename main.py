# Copyright (c) CUBOX, Inc. and its affiliates.
import pickle
import numpy as np
import cv2
import pygame
from pygame.locals import *
import os

# Import configuration
from config import WINDOW_NAME, Mode

# Import UI components
from ui.state import FaceBuilderState
from ui.pygame_renderer import draw_mesh_pygame

# Import controllers
from controllers.mouse_controller import MouseController
from controllers.button_controller import ButtonController

# Import utility functions
from utils.file_io import load_images_from_directory
from utils.geometry import remove_eyeballs

# Import model management functions
from model.mesh import project_current_3d_to_2d
from model.landmarks import update_all_landmarks
from model.pins import synchronize_pins_across_views

def initialize_state():
    """Initialize the application state with the 3D model and images"""
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
        return None
    
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
    
    return state
    
def main():
    # Initialize Pygame
    pygame.init()
    
    # Initialize the state
    state = initialize_state()
    if state is None:
        return
    
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
                MouseController.handle_mouse_event(event, state)
        
        # Clear screen and draw mesh
        draw_mesh_pygame(screen, state)
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()