# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np
import dlib  # Still needed for landmark prediction
import mediapipe as mp

def update_all_landmarks(state):
    """Update all predefined landmarks"""
    # Update all predefined landmarks
    for i in range(len(state.landmark_positions)):
        f_idx = state.face_for_lmk[i]
        bc = state.lmk_b_coords[i]
        i0, i1, i2 = state.faces[f_idx]
        v0, v1, v2 = state.verts2d[i0], state.verts2d[i1], state.verts2d[i2]
        state.landmark_positions[i] = bc[0]*v0 + bc[1]*v1 + bc[2]*v2
        
def align_face(state):
    """Align the 3D face model to detected face landmarks with proper 3D rotation"""
    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2RGB)
    # Convert to grayscale for landmark detection with dlib
    gray = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2GRAY)
    
    # Initialize MediaPipe face detection if not already initialized
    if not hasattr(state, 'mp_face_detection'):
        mp_face_detection = mp.solutions.face_detection
        state.mp_face_detection = mp_face_detection
        state.mp_face_detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.3,  # Lower threshold for better side view detection
            model_selection=1  # Use full-range model for profile views
        )
    
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
    
    # Get the landmarks using dlib's predictor
    shape = state.dlib_predictor(gray, rect)
    dlib_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    
    # Use only corresponding landmarks
    if dlib_landmarks.shape[0] == 68 and len(state.landmark3d_default) == 51:
        dlib_landmarks = dlib_landmarks[17:68]
    
    if len(state.landmark3d_default) != dlib_landmarks.shape[0]:
        print(f"Mismatch in number of landmarks: FLAME has {len(state.landmark3d_default)} and dlib detected {dlib_landmarks.shape[0]}.")
        # Use any landmarks we have - better partial alignment than none
        max_landmarks = min(len(dlib_landmarks), len(state.landmark3d_default))
        dlib_landmarks = dlib_landmarks[:max_landmarks]
        flame_landmarks_3d = state.landmark3d_default[:max_landmarks]
        print(f"Using {max_landmarks} landmarks for partial alignment")
    else:
        flame_landmarks_3d = state.landmark3d_default
    
    # Now use 3D pose estimation to get rotation and translation
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
        
        # Solve for 3D pose using solvePnP
        # Use SOLVEPNP_EPNP algorithm which is more stable for this case
        success, rvec, tvec = cv2.solvePnP(
            flame_landmarks_3d, 
            dlib_landmarks, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if success:
            print("Successfully estimated 3D pose!")
            
            # Project all 3D vertices to 2D using the estimated pose
            vertices_3d = np.array(state.verts3d_default, dtype=np.float32)
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
            
            # Check if projection is valid (sometimes projection can send points far outside image)
            verts_centroid = np.mean(new_verts, axis=0)
            x_inside = 0 <= verts_centroid[0] <= state.img_w
            y_inside = 0 <= verts_centroid[1] <= state.img_h
            
            if x_inside and y_inside:
                # Update mesh with full 3D rotation
                state.verts2d = new_verts
                state.landmark_positions = new_landmarks
                
                # Store camera parameters for this view
                state.camera_matrices[state.current_image_idx] = camera_matrix
                state.rotations[state.current_image_idx] = rvec
                state.translations[state.current_image_idx] = tvec
                
                print("Applied full 3D rotation to mesh")
            else:
                # If projection is invalid, fall back to 2D alignment
                print("3D projection went outside image bounds, using 2D alignment instead")
                raise cv2.error("Projection out of bounds")
        else:
            # If solvePnP fails, fall back to 2D alignment
            print("Failed to estimate 3D pose, falling back to 2D alignment")
            raise cv2.error("SolvePnP failed")
            
    except cv2.error as e:
        print(f"Error in 3D pose estimation: {e}")
        print("Using 2D alignment as fallback")
        
        # Fall back to 2D alignment (original method)
        # Calculate centers of both landmark sets
        dlib_center = np.mean(dlib_landmarks, axis=0)
        model_center = np.mean(state.landmark_positions_default, axis=0)
        
        # Calculate scale based on average distances from center
        dlib_centered = dlib_landmarks - dlib_center
        model_centered = state.landmark_positions_default[:len(dlib_landmarks)] - model_center
        
        dlib_scale = np.mean(np.linalg.norm(dlib_centered, axis=1)) if len(dlib_centered) > 0 else 1.0
        model_scale = np.mean(np.linalg.norm(model_centered, axis=1)) if len(model_centered) > 0 else 1.0
        scale_factor = dlib_scale / model_scale if model_scale > 0 else 1.0
        
        # Apply transformation to vertices and landmarks
        translation = dlib_center - model_center * scale_factor
        state.verts2d = (state.verts2d_default - model_center) * scale_factor + dlib_center
        state.landmark_positions = (state.landmark_positions_default - model_center) * scale_factor + dlib_center
        
        # Store transformation parameters
        transform_matrix = np.array([
            [scale_factor, 0, translation[0]],
            [0, scale_factor, translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        state.camera_matrices[state.current_image_idx] = transform_matrix
        state.translations[state.current_image_idx] = translation
        state.rotations[state.current_image_idx] = None
    
    # Update custom pins
    state.callbacks['update_custom_pins'](state)
    
    # Restore landmark visibility
    if hasattr(state, 'landmark_pins_hidden'):
        state.landmark_pins_hidden = False
    
    print(f"Alignment complete for image {state.current_image_idx+1}")
    state.callbacks['redraw'](state)