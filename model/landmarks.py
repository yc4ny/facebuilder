# Copyright (c) CUBOX, Inc. and its affiliates.
"""
Facial landmarks module.

This module manages the detection, tracking, and updating of facial landmarks
used to align and manipulate the 3D mesh to match the face in images.
"""
import cv2
import numpy as np
import dlib
from model.pins import update_custom_pins

# Global cache to store alignment results across application sessions
ALIGNMENT_CACHE = {}

def update_all_landmarks(state):
    """
    Update all predefined landmarks based on the current 3D mesh.
    
    This function recalculates the positions of landmarks using the current mesh
    state and projects them to 2D using the current camera parameters, preserving
    any landmarks that are being dragged by the user.
    
    Args:
        state: Application state containing mesh data and camera parameters
    """
    # Check if a landmark is being dragged - don't update that one
    is_dragging_landmark = (hasattr(state, 'drag_index') and 
                           state.drag_index != -1 and 
                           state.drag_index < len(state.landmark_positions))
    dragged_landmark_idx = state.drag_index if is_dragging_landmark else -1
    
    # First update the 3D positions of landmarks using barycentric coordinates
    for i in range(len(state.landmark3d)):
        # Skip if this is the landmark being dragged
        if i == dragged_landmark_idx:
            continue
            
        # Get the face and barycentric coordinates for this landmark
        f_idx = state.face_for_lmk[i]
        bc = state.lmk_b_coords[i]
        
        # Get the 3D vertices of the triangle
        i0, i1, i2 = state.faces[f_idx]
        v0, v1, v2 = state.verts3d[i0], state.verts3d[i1], state.verts3d[i2]
        
        # Compute 3D position using barycentric interpolation:
        # p = α*v₀ + β*v₁ + γ*v₂
        # where (α,β,γ) are barycentric coordinates and (v₀,v₁,v₂) are vertices
        state.landmark3d[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
    
    # Project updated 3D landmarks to 2D for the current view
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
    
    # Project non-dragged landmarks to 2D using perspective projection
    try:
        # Create array of 3D landmarks to project (excluding dragged landmark)
        landmarks_to_project = []
        landmark_indices = []
        
        for i, landmark_3d in enumerate(state.landmark3d):
            if i != dragged_landmark_idx:
                landmarks_to_project.append(landmark_3d)
                landmark_indices.append(i)
        
        if landmarks_to_project:  # Only project if we have landmarks to update
            landmarks_to_project = np.array(landmarks_to_project, dtype=np.float32)
            
            # Project 3D points to 2D using equation: x' = K[R|t]X
            projected_landmarks, _ = cv2.projectPoints(
                landmarks_to_project,
                rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            
            # Update the non-dragged landmark positions
            for idx, proj_idx in enumerate(landmark_indices):
                state.landmark_positions[proj_idx] = projected_landmarks[idx].reshape(2)
    
    except cv2.error as e:
        print(f"Error projecting landmarks: {e}")
        # Fallback to direct calculation from 2D vertices if projection fails
        update_landmarks_from_2d(state, dragged_landmark_idx)


def update_landmarks_from_2d(state, skip_idx=-1):
    """
    Update landmarks directly from 2D vertices using barycentric coordinates.
    
    This is a fallback method when 3D projection fails.
    
    Args:
        state: Application state
        skip_idx: Index of landmark to skip updating (e.g., being dragged)
    """
    for i in range(len(state.landmark_positions)):
        # Skip the landmark being dragged
        if i == skip_idx:
            continue
            
        # Get face and barycentric coordinates for this landmark
        f_idx = state.face_for_lmk[i]
        bc = state.lmk_b_coords[i]
        
        # Get 2D vertices of the face
        i0, i1, i2 = state.faces[f_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        
        # Compute 2D position using barycentric interpolation
        state.landmark_positions[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2


def align_face(state):
    """
    Align the 3D face model to detected face landmarks with proper 3D rotation.
    
    This function detects facial landmarks in the current image and uses them
    to calculate camera parameters that align the 3D model with the detected face.
    Results are cached to speed up subsequent alignments of the same image.
    
    Args:
        state: Application state
    """
    global ALIGNMENT_CACHE
    
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
    
    # Prepare image for detection
    rgb_img = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2RGB)  # For MediaPipe
    gray = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2GRAY)    # For dlib
    
    # Set a fixed random seed for deterministic behavior
    np.random.seed(42 + state.current_image_idx)
    
    # Get face bounding box using MediaPipe or dlib
    rect = _detect_face_bounding_box(state, rgb_img, gray)
    
    # Get the landmarks using dlib's predictor with fixed random seed
    shape = state.dlib_predictor(gray, rect)
    dlib_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    
    # Extract corresponding landmarks between dlib and FLAME model
    dlib_landmarks, flame_landmarks_3d = _extract_corresponding_landmarks(state, dlib_landmarks)
    
    # Try to use 3D pose estimation for better alignment
    try:
        _apply_3d_pose_alignment(state, dlib_landmarks, flame_landmarks_3d)
    except cv2.error as e:
        print(f"Error in 3D pose estimation: {e}")
        print("Using 2D alignment as fallback")
        _apply_2d_fallback_alignment(state, dlib_landmarks)
    
    # Cache the alignment results
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


def _detect_face_bounding_box(state, rgb_img, gray):
    """
    Detect face bounding box using MediaPipe or dlib.
    
    Args:
        state: Application state
        rgb_img: RGB image for MediaPipe
        gray: Grayscale image for dlib
        
    Returns:
        dlib.rectangle: Detected face bounding box
    """
    detected = False
    
    # Try MediaPipe detection first (usually more robust)
    results = state.mp_face_detector.process(rgb_img)
    
    # Check if MediaPipe detected any faces
    if results and results.detections and len(results.detections) > 0:
        # Get the first detection
        detection = results.detections[0]
        
        # Extract bounding box
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = rgb_img.shape
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
    
    return rect


def _extract_corresponding_landmarks(state, dlib_landmarks):
    """
    Extract corresponding landmarks between dlib and FLAME model.
    
    Args:
        state: Application state
        dlib_landmarks: Landmarks detected by dlib
        
    Returns:
        tuple: (dlib_landmarks, flame_landmarks_3d) - matching sets of landmarks
    """
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
    
    return dlib_landmarks, flame_landmarks_3d


def _apply_3d_pose_alignment(state, dlib_landmarks, flame_landmarks_3d):
    """
    Apply 3D pose estimation to align mesh with detected landmarks.
    
    Uses the PnP (Perspective-n-Point) algorithm to solve for the camera pose.
    
    Args:
        state: Application state
        dlib_landmarks: 2D landmarks detected in the image
        flame_landmarks_3d: Corresponding 3D landmarks from the FLAME model
        
    Raises:
        cv2.error: If 3D pose estimation fails
    """
    # Create camera matrix - using focal length based on image size
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # No lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Use deterministic EPNP algorithm to solve the PnP problem:
    # The PnP problem finds the camera pose (R,t) that minimizes the reprojection error:
    # Σ ||p_i - K[R|t]P_i||², where p_i are 2D points and P_i are 3D points
    success, rvec, tvec = cv2.solvePnP(
        flame_landmarks_3d, 
        dlib_landmarks, 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    
    # Only refine if successful, using fixed iterations for determinism
    if success:
        # Refine the pose using Virtual Visual Servoing (VVS)
        rvec, tvec = cv2.solvePnPRefineVVS(
            flame_landmarks_3d, 
            dlib_landmarks, 
            camera_matrix, 
            dist_coeffs,
            rvec,
            tvec,
            criteria=(cv2.TERM_CRITERIA_COUNT, 20, 0.0001)
        )
        
        # Project all 3D vertices to 2D using the estimated pose
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
        
        # Check if projection is valid (centroid inside image)
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


def _apply_2d_fallback_alignment(state, dlib_landmarks):
    """
    Apply 2D scaling and translation alignment as fallback.
    
    Args:
        state: Application state
        dlib_landmarks: 2D landmarks detected in the image
    """
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
    # This finds the scaling that best matches the size of the detected face
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
        # x' = s(x - cx) + tx, y' = s(y - cy) + ty
        # where s is scale, (cx,cy) is model center, (tx,ty) is detected center
        state.verts2d = (state.verts2d - model_center) * scale_factor + detected_center
        
        # Also update landmark positions
        update_all_landmarks(state)
    except Exception as e:
        print(f"Error in 2D fallback alignment: {e}")


def reset_alignment_cache():
    """
    Reset the alignment cache to force re-alignment of all images.
    """
    global ALIGNMENT_CACHE
    ALIGNMENT_CACHE = {}
    print("Alignment cache cleared for all images")


def clear_image_alignment(state):
    """
    Clear the alignment cache for the current image.
    
    Args:
        state: Application state
    """
    global ALIGNMENT_CACHE
    cache_key = f"image_{state.current_image_idx}"
    if cache_key in ALIGNMENT_CACHE:
        del ALIGNMENT_CACHE[cache_key]
        print(f"Cleared alignment cache for image {state.current_image_idx+1}")