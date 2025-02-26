import pickle
import numpy as np
import cv2
import dlib
import os

from config import WINDOW_NAME, Mode
from utils.file_io import load_images_from_directory, save_model, save_to_obj
from utils.geometry import ortho, remove_eyeballs
from ui.dimensions import calculate_ui_dimensions
from ui.rendering import redraw
from ui.input import on_mouse
from model.mesh import move_mesh_2d
from model.landmarks import update_all_landmarks, align_face
from model.pins import add_custom_pin, update_custom_pins, remove_pins, center_geo, reset_shape

class FaceBuilderState:
    """Class to hold the state of the Face Builder application"""
    def __init__(self):
        # Operation modes
        self.mode = Mode.MOVE
        self.drag_index = -1
        self.drag_offset = (0, 0)
        self.custom_pins = []  # List of custom pins [(x, y, face_idx, barycentric_coords), ...]
        self.custom_pin_colors = []  # Color for each custom pin
        
        # Face model variables
        self.landmark_positions = None
        self.verts2d = None
        self.faces = None
        self.face_for_lmk = None
        self.lmk_b_coords = None
        self.overlay = None
        self.img_h = 0
        self.img_w = 0
        self.alpha = 1e-4
        self.weights = None
        
        # Default pose variables
        self.verts2d_default = None
        self.landmark_positions_default = None
        self.verts3d_default = None
        self.landmark3d_default = None
        
        # Multi-view support
        self.images = []  # List of input images
        self.current_image_idx = 0  # Current image index
        self.pins_per_image = []  # Custom pins for each image
        self.camera_matrices = []  # Camera matrices for each view
        self.rotations = []  # Rotation vectors for each view
        self.translations = []  # Translation vectors for each view
        
        # Initialize face detector and predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        
        # UI dimensions
        self.ui_dimensions = None
        
        # Callbacks for UI interactions
        self.callbacks = {
            'redraw': redraw,
            'align_face': align_face,
            'update_ui': self.update_ui,
            'next_image': self.next_image,
            'prev_image': self.prev_image,
            'save_model': self.save_model,
            'remove_pins': remove_pins,
            'center_geo': center_geo,
            'reset_shape': reset_shape,  # Add this line
            'update_custom_pins': update_custom_pins
        }
    
    def update_ui(self, state=None):
        """Update UI after changing image or mode"""
        self.ui_dimensions = calculate_ui_dimensions(self.img_w, self.img_h)
        self.callbacks['redraw'](self)

    def next_image(self, state=None):
        """Switch to the next image"""
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.overlay = self.images[self.current_image_idx].copy()
            self.img_h, self.img_w = self.overlay.shape[:2]
            self.update_ui()
    
    def prev_image(self, state=None):
        """Switch to the previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.overlay = self.images[self.current_image_idx].copy()
            self.img_h, self.img_w = self.overlay.shape[:2]
            self.update_ui()
    
    def save_model(self, state=None):
        """Save the current 3D model to file"""
        save_model(
            self.verts2d, 
            self.verts3d_default, 
            self.faces, 
            self.pins_per_image, 
            self.camera_matrices, 
            self.rotations, 
            self.translations
        )

def main():
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
    
    # Calculate initial 2D projection using orthographic projection
    mn = verts_3d[:, :2].min(axis=0)
    mx = verts_3d[:, :2].max(axis=0)
    c3d = 0.5 * (mn + mx)
    s3d = (mx - mn).max()
    sc = 0.8 * min(state.img_w, state.img_h) / s3d
    c2d = np.array([state.img_w/2.0, state.img_h/2.0])
    
    state.verts2d = ortho(verts_3d, c3d, c2d, sc)
    
    # Calculate 3D landmark positions and their 2D projections
    lmk_3d = []
    for face_idx, bc in zip(state.face_for_lmk, state.lmk_b_coords):
        i0, i1, i2 = state.faces[face_idx]
        v0 = verts_3d[i0]
        v1 = verts_3d[i1]
        v2 = verts_3d[i2]
        xyz = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        lmk_3d.append(xyz)
    
    # Project landmarks to 2D
    lmk_2d = ortho(np.array(lmk_3d), c3d, c2d, sc)
    state.landmark_positions = lmk_2d
    
    # Store default positions
    state.verts2d_default = state.verts2d.copy()
    state.landmark_positions_default = state.landmark_positions.copy()
    state.verts3d_default = verts_3d.copy()
    state.landmark3d_default = np.array(lmk_3d).copy()
    
    # Calculate UI dimensions
    state.ui_dimensions = calculate_ui_dimensions(state.img_w, state.img_h)
    
    # Create window and set mouse callback
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, _: on_mouse(event, x, y, flags, _, state))
    
    print("\nEnhanced Face Builder Controls:")
    print("- Click 'Align Face' to automatically detect and align the 3D mesh to the face")
    print("- Click 'Add Pin' to toggle between adding custom pins and moving points")
    print("- Click 'Toggle Pins' to move pins without affecting the mesh")
    print("- Click 'Remove Pins' to clear all custom pins from the current image")
    print("- Click 'Center Geo' to reset the mesh to its default position")
    print("- Use 'Next Image' and 'Prev Image' to switch between input images")
    print("- Click 'Save' to save the current model to file")
    print("- Drag landmarks or custom pins to manipulate the 3D mesh")
    print("- Press ESC to exit\n")
    
    redraw(state)
    
    while True:
        if cv2.waitKey(50) == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()