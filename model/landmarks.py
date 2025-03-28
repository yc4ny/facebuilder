# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np
import dlib
import mediapipe as mp
from model.pins import update_custom_pins

def update_all_landmarks(state):
    """
    Update all predefined landmarks based on the current 3D mesh
    """
    # First update the 3D positions of landmarks
    for i in range(len(state.landmark3d)):
        f_idx = state.face_for_lmk[i]
        bc = state.lmk_b_coords[i]
        i0, i1, i2 = state.faces[f_idx]
        v0, v1, v2 = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        state.landmark3d[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
    
    # Then project to 2D for the current view
    camera_matrix = state.camera_matrices[state.current_image_idx]
    rvec = state.rotations[state.current_image_idx]
    tvec = state.translations[state.current_image_idx]
    
    # Ensure we have valid camera parameters
    if camera_matrix is None or rvec is None or tvec is None:
        # Initialize default camera parameters if not present
        focal_length = max(state.img_w, state.img_h)
        camera_matrix = np.array([
            [focal_length, 0, state.img_w / 2],
            [0, focal_length, state.img_h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Calculate center of the model
        center = np.mean(state.verts3d, axis=0)
        
        # Initialize rotation (identity rotation)
        R = np.eye(3, dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)
        
        # Position the model in front of the camera
        distance = focal_length * 1.5
        tvec = np.array([[0, 0, distance]], dtype=np.float32).T - R @ center.reshape(3, 1)
        
        # Save camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
    
    # Use perspective projection
    try:
        projected_landmarks, _ = cv2.projectPoints(
            np.array(state.landmark3d, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        state.landmark_positions = projected_landmarks.reshape(-1, 2)
    except cv2.error as e:
        print(f"Error projecting landmarks: {e}")
        # Fallback to direct calculation from 2D vertices if projection fails
        update_landmarks_from_2d(state)

def update_landmarks_from_2d(state):
    """Update landmarks directly from 2D vertices (fallback method)"""
    for i in range(len(state.landmark_positions)):
        f_idx = state.face_for_lmk[i]
        bc = state.lmk_b_coords[i]
        i0, i1, i2 = state.faces[f_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        state.landmark_positions[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2

def align_face(state):
    """Align the 3D face model to detected face landmarks with proper 3D rotation"""
    # Use a global variable outside the state object for stability
    global ALIGNMENT_CACHE
    
    # Initialize the global cache if it doesn't exist
    if 'ALIGNMENT_CACHE' not in globals():
        global ALIGNMENT_CACHE
        ALIGNMENT_CACHE = {}
        print("Initialized global alignment cache")
    
    # Generate a unique key for this image
    cache_key = f"image_{state.current_image_idx}"
    
    # Check if we have cached data for this image
    if cache_key in ALIGNMENT_CACHE:
        print(f"Using cached alignment for image {state.current_image_idx+1}")
        cached_data = ALIGNMENT_CACHE[cache_key]
        
        # Apply cached camera parameters
        state.camera_matrices[state.current_image_idx] = cached_data['camera_matrix']
        state.rotations[state.current_image_idx] = cached_data['rvec']
        state.translations[state.current_image_idx] = cached_data['tvec']
        
        # Project current 3D vertices to 2D using these parameters
        if 'project_2d' in state.callbacks:
            state.callbacks['project_2d'](state)
        
        # Update landmarks
        update_all_landmarks(state)
        
        # Make landmarks visible again if they were hidden
        if hasattr(state, 'landmark_pins_hidden'):
            state.landmark_pins_hidden = False
            print("Landmarks made visible again after using cached alignment")
        
        # Update custom pins
        update_custom_pins(state)
        return
    
    # If we're here, we need to perform a new alignment
    print(f"Performing new alignment for image {state.current_image_idx+1}")
    
    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2RGB)
    # Convert to grayscale for landmark detection with dlib
    gray = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2GRAY)
    
    # Set a fixed random seed for deterministic behavior
    np.random.seed(42 + state.current_image_idx)
    
    # Process the image with MediaPipe
    results = state.mp_face_detector.process(rgb_img)
    
    detected = False
    # Check if MediaPipe detected any faces
    if results and results.detections and len(results.detections) > 0:
        # Get the first detection
        detection = results.detections[0]
        
        # Extract bounding box
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = state.overlay.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                     int(bbox.width * iw), int(bbox.height * ih)
        
        # Ensure valid coordinates
        x = max(0, x)
        y = max(0, y)
        w = min(iw - x, w)
        h = min(ih - y, h)
        
        # Create dlib rectangle from MediaPipe detection
        rect = dlib.rectangle(x, y, x + w, y + h)
        detected = True
        print(f"MediaPipe detected face with confidence {detection.score[0]:.2f}")
    
    # If MediaPipe failed, try dlib as fallback
    if not detected:
        # Try with dlib detector as fallback
        rects = state.dlib_detector(gray, 1)  # Use upsampling for better detection
        
        if len(rects) == 0:
            # If still no detection, try with histogram equalization
            equalized = cv2.equalizeHist(gray)
            rects = state.dlib_detector(equalized, 1)
            
            if len(rects) == 0:
                print("No face detected with either MediaPipe or dlib!")
                
                # Create a manual rectangle in the center of the image as last resort
                center_x = state.img_w // 2
                center_y = state.img_h // 2
                rect_size = min(state.img_w, state.img_h) * 2 // 3
                
                rect = dlib.rectangle(
                    left=center_x - rect_size//2,
                    top=center_y - rect_size//2,
                    right=center_x + rect_size//2,
                    bottom=center_y + rect_size//2
                )
                print("Using manual central rectangle as fallback")
            else:
                rect = rects[0]
                print("Using dlib detection with equalized image")
                gray = equalized  # Use equalized image for landmark detection
        else:
            rect = rects[0]
            print("Using dlib detection as fallback")
    
    # Get the landmarks using dlib's predictor with fixed random seed
    shape = state.dlib_predictor(gray, rect)
    dlib_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    
    # Use only corresponding landmarks
    if dlib_landmarks.shape[0] == 68 and len(state.landmark3d_default) == 51:
        dlib_landmarks = dlib_landmarks[17:68]  # Use only face contour landmarks
    
    if len(state.landmark3d_default) != dlib_landmarks.shape[0]:
        print(f"Mismatch in number of landmarks: FLAME has {len(state.landmark3d_default)} and dlib detected {dlib_landmarks.shape[0]}.")
        max_landmarks = min(len(dlib_landmarks), len(state.landmark3d_default))
        dlib_landmarks = dlib_landmarks[:max_landmarks]
        flame_landmarks_3d = state.landmark3d_default[:max_landmarks]
        print(f"Using {max_landmarks} landmarks for partial alignment")
    else:
        flame_landmarks_3d = state.landmark3d_default
    
    # Try to use 3D pose estimation for better alignment
    try:
        # Create camera matrix - using focal length based on image size
        focal_length = max(state.img_w, state.img_h)
        camera_matrix = np.array([
            [focal_length, 0, state.img_w / 2],
            [0, focal_length, state.img_h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # No lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Use deterministic EPNP algorithm
        success, rvec, tvec = cv2.solvePnP(
            flame_landmarks_3d, 
            dlib_landmarks, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        # Only refine if successful, using fixed iterations
        if success:
            rvec, tvec = cv2.solvePnPRefineVVS(
                flame_landmarks_3d, 
                dlib_landmarks, 
                camera_matrix, 
                dist_coeffs,
                rvec,
                tvec,
                criteria=(cv2.TERM_CRITERIA_COUNT, 20, 0.0001)  # Fixed iterations for determinism
            )
            
            # Project all 3D vertices to 2D using the estimated pose
            # Use the current (potentially modified) 3D vertices
            vertices_3d = np.array(state.verts3d, dtype=np.float32)
            projected_vertices, _ = cv2.projectPoints(
                vertices_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            
            # Get 2D positions
            new_verts = projected_vertices.reshape(-1, 2)
            
            # Project landmarks as well
            projected_landmarks, _ = cv2.projectPoints(
                flame_landmarks_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            new_landmarks = projected_landmarks.reshape(-1, 2)
            
            # Check if projection is valid
            verts_centroid = np.mean(new_verts, axis=0)
            x_inside = 0 <= verts_centroid[0] <= state.img_w
            y_inside = 0 <= verts_centroid[1] <= state.img_h
            
            if x_inside and y_inside:
                # Update 2D projection with full 3D rotation
                state.verts2d = new_verts
                state.landmark_positions = new_landmarks
                
                # Store camera parameters for this view
                state.camera_matrices[state.current_image_idx] = camera_matrix
                state.rotations[state.current_image_idx] = rvec
                state.translations[state.current_image_idx] = tvec
                
                print("Applied full 3D rotation to mesh")
            else:
                print("3D projection went outside image bounds, using 2D alignment instead")
                raise cv2.error("Projection out of bounds")
        else:
            print("Failed to estimate 3D pose, falling back to 2D alignment")
            raise cv2.error("SolvePnP failed")
            
    except cv2.error as e:
        print(f"Error in 3D pose estimation: {e}")
        print("Using 2D alignment as fallback")
        
        # Set up default camera parameters
        focal_length = max(state.img_w, state.img_h)
        camera_matrix = np.array([
            [focal_length, 0, state.img_w / 2],
            [0, focal_length, state.img_h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Calculate centers of both landmark sets
        detected_center = np.mean(dlib_landmarks, axis=0)
        model_center = np.mean(state.landmark_positions_default, axis=0)
        
        # Calculate scale based on average distances from center
        detected_centered = dlib_landmarks - detected_center
        model_centered = state.landmark_positions_default[:len(dlib_landmarks)] - model_center
        
        detected_scale = np.mean(np.linalg.norm(detected_centered, axis=1)) if len(detected_centered) > 0 else 1.0
        model_scale = np.mean(np.linalg.norm(model_centered, axis=1)) if len(model_centered) > 0 else 1.0
        scale_factor = detected_scale / model_scale if model_scale > 0 else 1.0
        
        # Create initial rotation (identity rotation)
        R = np.eye(3, dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)
        
        # Calculate model center in 3D
        model_center_3d = np.mean(state.verts3d_default, axis=0)
        
        # Create translation to position the model
        translation = np.array([0, 0, focal_length * 1.5])
        tvec = translation.reshape(3, 1)
        
        # Store camera parameters
        state.camera_matrices[state.current_image_idx] = camera_matrix
        state.rotations[state.current_image_idx] = rvec
        state.translations[state.current_image_idx] = tvec
        
        # Project 3D vertices to 2D
        try:
            projected_verts, _ = cv2.projectPoints(
                state.verts3d, rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            state.verts2d = projected_verts.reshape(-1, 2)
            
            # Apply 2D scaling and translation to better match detected landmarks
            state.verts2d = (state.verts2d - model_center) * scale_factor + detected_center
            
            # Also update landmark positions
            update_all_landmarks(state)
        except Exception as e:
            print(f"Error in 2D fallback alignment: {e}")
    
    # Cache the alignment results in the global cache
    ALIGNMENT_CACHE[cache_key] = {
        'camera_matrix': state.camera_matrices[state.current_image_idx],
        'rvec': state.rotations[state.current_image_idx],
        'tvec': state.translations[state.current_image_idx]
    }
    
    # Make landmarks visible again if they were hidden
    if hasattr(state, 'landmark_pins_hidden'):
        state.landmark_pins_hidden = False
        print("Landmarks made visible again")
    
    # Update custom pins
    update_custom_pins(state)
    
    print(f"Alignment complete for image {state.current_image_idx+1}")

def reset_alignment_cache():
    """Reset the alignment cache to force re-alignment of all images"""
    global ALIGNMENT_CACHE
    ALIGNMENT_CACHE = {}
    print("Alignment cache cleared for all images")


def clear_image_alignment(state):
    """Clear the alignment cache for the current image"""
    global ALIGNMENT_CACHE
    cache_key = f"image_{state.current_image_idx}"
    if cache_key in ALIGNMENT_CACHE:
        del ALIGNMENT_CACHE[cache_key]
        print(f"Cleared alignment cache for image {state.current_image_idx+1}")