# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np

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
    """Align the 3D face model to detected face landmarks"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(state.overlay, cv2.COLOR_BGR2GRAY)
    rects = state.dlib_detector(gray, 1)
    
    if len(rects) == 0:
        print("No face detected!")
        return
    
    rect = rects[0]
    shape = state.dlib_predictor(gray, rect)
    dlib_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    
    # Use only corresponding landmarks
    if dlib_landmarks.shape[0] == 68 and len(state.landmark3d_default) == 51:
        dlib_landmarks = dlib_landmarks[17:68]
    
    if len(state.landmark3d_default) != dlib_landmarks.shape[0]:
        print(f"Mismatch in number of landmarks: FLAME has {len(state.landmark3d_default)} and dlib detected {dlib_landmarks.shape[0]}.")
        return
    
    # Create camera matrix
    f = max(state.img_w, state.img_h)
    camera_matrix = np.array([[f, 0, state.img_w/2.0],
                              [0, f, state.img_h/2.0],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    # Get 3D and 2D correspondence points
    object_points = np.array(state.landmark3d_default, dtype=np.float32)
    image_points = dlib_landmarks
    
    # Solve for pose
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        print("Could not solve PnP")
        return
    
    # Store camera parameters for this view
    state.camera_matrices[state.current_image_idx] = camera_matrix
    state.rotations[state.current_image_idx] = rvec
    state.translations[state.current_image_idx] = tvec
    
    # Project 3D vertices and landmarks to 2D
    projected_verts, _ = cv2.projectPoints(np.array(state.verts3d_default, dtype=np.float32),
                                           rvec, tvec, camera_matrix, dist_coeffs)
    projected_verts = projected_verts.reshape(-1, 2)
    
    projected_landmarks, _ = cv2.projectPoints(np.array(state.landmark3d_default, dtype=np.float32),
                                              rvec, tvec, camera_matrix, dist_coeffs)
    projected_landmarks = projected_landmarks.reshape(-1, 2)
    
    # Update the mesh and landmarks
    state.verts2d = projected_verts
    state.landmark_positions = projected_landmarks
    
    # Update custom pins based on the new mesh
    state.callbacks['update_custom_pins'](state)
    
    print(f"Alignment complete for image {state.current_image_idx+1}.")
    state.callbacks['redraw'](state)
