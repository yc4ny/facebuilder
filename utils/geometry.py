# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import cv2

# Import the ear vertex indices if available
try:
    from ear_indices import LEFT_EAR_VERTICES, RIGHT_EAR_VERTICES
    PREDEFINED_EARS = True
except ImportError:
    PREDEFINED_EARS = False
    print("No predefined ear indices found. Will detect ears based on geometry.")

def ortho(v3d, c3d, c2d, sc):
    """
    Orthographic projection of 3D points to 2D
    """
    xy = v3d[:, :2] - c3d
    xy *= sc
    xy[:, 1] = -xy[:, 1]
    xy += c2d
    return xy

def inverse_ortho(v2d, v3d_ref, c3d, c2d, sc):
    """
    Inverse orthographic projection from 2D to 3D
    Preserves the z-coordinate from the reference 3D points
    
    Args:
        v2d: 2D points after user modification
        v3d_ref: Original 3D points (used to get z values)
        c3d: 3D center used during projection
        c2d: 2D center used during projection
        sc: Scale factor used during projection
        
    Returns:
        Updated 3D points
    """
    v3d_new = np.zeros_like(v3d_ref)
    
    # Invert the projection for x and y coordinates
    xy = (v2d - c2d)
    xy[:, 1] = -xy[:, 1]  # Invert y-axis
    xy /= sc
    xy += c3d
    
    # Copy the x and y coordinates
    v3d_new[:, 0] = xy[:, 0]
    v3d_new[:, 1] = xy[:, 1]
    
    # Keep the original z coordinates
    v3d_new[:, 2] = v3d_ref[:, 2]
    
    return v3d_new

def project_3d_to_2d(verts3d, camera_matrix, rvec, tvec):
    """
    Project 3D vertices to 2D using camera parameters
    
    Args:
        verts3d: 3D vertices
        camera_matrix: Camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        
    Returns:
        2D projected vertices
    """
    if rvec is None or tvec is None:
        return None
        
    try:
        projected_verts, _ = cv2.projectPoints(
            np.array(verts3d, dtype=np.float32),
            rvec, tvec, camera_matrix, np.zeros((4, 1))
        )
        return projected_verts.reshape(-1, 2)
    except cv2.error:
        return None

def back_project_2d_to_3d(verts2d, verts3d_ref, camera_matrix, rvec, tvec):
    """
    Back-project 2D points to 3D using reference 3D points
    For spherical rotation, preserves the distance from center
    
    Args:
        verts2d: Current 2D vertices
        verts3d_ref: Reference 3D vertices (original shape)
        camera_matrix: Camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        
    Returns:
        Updated 3D vertices
    """
    if rvec is None or tvec is None:
        return None
    
    # Convert to numpy arrays
    verts2d = np.array(verts2d, dtype=np.float32)
    verts3d_ref = np.array(verts3d_ref, dtype=np.float32)
    
    # Detect if we're in spherical rotation mode by checking landmark movements
    # In spherical rotation, vertices maintain distance from center
    # First, compute center of mass for original and current 3D vertices
    com_orig = np.mean(verts3d_ref, axis=0)
    
    # Get the rotation matrix from rotation vector
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    
    # Check if we're doing a spherical rotation
    # In that case, we'll preserve distances from the center
    is_spherical_rotation = False
    
    # Count how many vertices have significantly changed distance from center in 2D
    # If fewer than 5% have changed, we're likely doing a spherical rotation
    center_2d = np.mean(verts2d, axis=0)
    dist_2d = np.linalg.norm(verts2d - center_2d, axis=1)
    
    # Create a new 3D vertices array based on the back-projection
    verts3d_new = verts3d_ref.copy()
    
    # Determine transformation type based on movement patterns
    # Get camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    
    # Camera center in world coordinates
    camera_center = -R_inv.dot(tvec).ravel()
    
    # For each vertex, find the best 3D position
    for i in range(len(verts2d)):
        # Get 2D target point
        x, y = verts2d[i]
        
        # Create ray direction in camera coordinates
        ray_camera = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
        
        # Transform ray to world coordinates
        ray_world = R_inv.dot(ray_camera)
        ray_world = ray_world / np.linalg.norm(ray_world)  # Normalize
        
        # Original 3D point position
        orig_pos = verts3d_ref[i]
        
        # Vector from camera center to reference point
        w = orig_pos - camera_center
        
        # Distance from center to original vertex
        dist_from_center = np.linalg.norm(orig_pos - com_orig)
        
        # Project w onto ray_world to find closest point on ray
        proj_length = np.dot(w, ray_world)
        
        # Calculate the closest point on the ray
        closest_point = camera_center + proj_length * ray_world
        
        # For spherical rotation: preserve distance from center
        # Find the point on the ray that preserves the distance from the center
        if is_spherical_rotation:
            # Direction from center to the point on the ray
            dir_to_ray = closest_point - com_orig
            dir_to_ray = dir_to_ray / np.linalg.norm(dir_to_ray)
            
            # Place point at the correct distance from center
            spherical_point = com_orig + dir_to_ray * dist_from_center
            
            # Update the vertex
            verts3d_new[i] = spherical_point
        else:
            # Standard back-projection - use closest point on ray
            verts3d_new[i] = closest_point
    
    return verts3d_new

def calculate_face_normals(vertices, faces):
    """
    Calculate the face normals for all faces in the mesh.
    
    Args:
        vertices: Array of 3D vertex positions
        faces: Array of face indices
        
    Returns:
        normals: Array of face normals (M x 3)
    """
    normals = np.zeros((len(faces), 3))
    for i, (i0, i1, i2) in enumerate(faces):
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize the normal
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        
        normals[i] = normal
    
    return normals

def identify_ear_vertices(vertices):
    """
    Identify which vertices belong to ear regions based on their position.
    Uses predefined indices if available, otherwise falls back to position-based detection.
    
    Args:
        vertices: Array of 3D vertex positions
        
    Returns:
        ear_vertices: Indices of vertices that are part of ears
        left_ear_vertices: Indices of left ear vertices
        right_ear_vertices: Indices of right ear vertices
    """
    if PREDEFINED_EARS:
        # Use predefined ear vertex indices
        left_ear_vertices = np.array(LEFT_EAR_VERTICES)
        right_ear_vertices = np.array(RIGHT_EAR_VERTICES)
        ear_vertices = np.concatenate([left_ear_vertices, right_ear_vertices])
        return ear_vertices, left_ear_vertices, right_ear_vertices
    
    # Fallback method: detect ears based on position
    # Calculate center of the head
    center = np.mean(vertices, axis=0)
    
    # Threshold for ear identification (distance from center in x direction)
    # This may need to be tuned for your specific model
    threshold = 0.06  # This is a reasonable value for FLAME models
    
    # Identify left and right ear vertices
    left_ear_vertices = np.where((vertices[:, 0] - center[0]) < -threshold)[0]
    right_ear_vertices = np.where((vertices[:, 0] - center[0]) > threshold)[0]
    
    # Combine all ear vertices
    ear_vertices = np.concatenate([left_ear_vertices, right_ear_vertices])
    
    return ear_vertices, left_ear_vertices, right_ear_vertices

def identify_ear_faces(faces, left_ear_vertices, right_ear_vertices):
    """
    Identify faces that are part of the left and right ears.
    
    Args:
        faces: Array of face indices
        left_ear_vertices: Indices of vertices that are part of left ear
        right_ear_vertices: Indices of vertices that are part of right ear
        
    Returns:
        left_ear_faces: Indices of faces that are part of left ear
        right_ear_faces: Indices of faces that are part of right ear
    """
    left_ear_faces = []
    right_ear_faces = []
    
    # Convert vertex arrays to sets for faster membership testing
    left_ear_set = set(left_ear_vertices)
    right_ear_set = set(right_ear_vertices)
    
    for i, face in enumerate(faces):
        i0, i1, i2 = face
        # A face belongs to an ear if at least one of its vertices is in the ear set
        if i0 in left_ear_set or i1 in left_ear_set or i2 in left_ear_set:
            left_ear_faces.append(i)
        if i0 in right_ear_set or i1 in right_ear_set or i2 in right_ear_set:
            right_ear_faces.append(i)
    
    return left_ear_faces, right_ear_faces

def calculate_front_facing(vertices, faces, camera_matrix=None, rvec=None, tvec=None):
    """
    Determine which faces are front-facing using OpenGL-style backface culling.
    Uses winding order to determine front-facing triangles (clockwise = front-facing).
    
    Args:
        vertices: Array of 3D vertex positions
        faces: Array of face indices
        camera_matrix: Camera intrinsic matrix for perspective projection.
        rvec: Rotation vector for perspective projection.
        tvec: Translation vector for perspective projection.
        
    Returns:
        mask: Boolean array indicating which faces are front-facing (True) and which are back-facing (False)
    """
    # Get 2D projected vertices
    verts2d = None
    if camera_matrix is not None and rvec is not None and tvec is not None:
        # Use perspective projection if we have camera parameters
        verts2d = project_3d_to_2d(vertices, camera_matrix, rvec, tvec)
    
    if verts2d is None:
        # Fall back to orthographic projection
        center = np.mean(vertices[:, :2], axis=0)
        max_dim = np.max(np.abs(vertices[:, :2] - center)) * 2
        scale = 1000.0 / max_dim  # Arbitrary scale to get reasonable 2D coordinates
        verts2d = vertices[:, :2] * scale
        verts2d += np.array([1000, 1000])  # Center in a 2000x2000 viewport
    
    # Calculate camera position
    if camera_matrix is not None and rvec is not None and tvec is not None:
        # Using OpenCV camera model
        R, _ = cv2.Rodrigues(rvec)
        camera_position = -R.T @ tvec
        camera_position = camera_position.ravel()
    else:
        # For orthographic projection, assume camera is looking from positive z direction
        camera_position = np.array([0, 0, 100000])  # Very far along z-axis
    
    # Determine front-facing triangles using OpenGL-style winding order test
    # In OpenGL with GL_CW, clockwise winding is front-facing
    front_facing = np.zeros(len(faces), dtype=bool)
    
    for i, face in enumerate(faces):
        i0, i1, i2 = face
        
        # Get 2D vertices of this face
        v0 = verts2d[i0]
        v1 = verts2d[i1]
        v2 = verts2d[i2]
        
        # Calculate signed area to determine winding order
        # Positive area = counterclockwise winding in screen space
        # Negative area = clockwise winding in screen space
        area = 0.5 * ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
        
        # In OpenGL with GL_CW, clockwise winding (negative area) is front-facing
        # Negate the comparison because we want clockwise to be front-facing
        # Note: OpenCV's Y axis points down, which is the opposite of OpenGL
        # This affects winding order, so we adapt accordingly
        front_facing[i] = area < 0  # Clockwise winding = front-facing
    
    # Special handling for ears
    _, left_ear_vertices, right_ear_vertices = identify_ear_vertices(vertices)
    left_ear_faces, right_ear_faces = identify_ear_faces(faces, left_ear_vertices, right_ear_vertices)
    
    # Calculate view direction for each face for ear culling
    face_centers = np.zeros((len(faces), 3))
    for i, (i0, i1, i2) in enumerate(faces):
        face_centers[i] = (vertices[i0] + vertices[i1] + vertices[i2]) / 3
    
    # Get vector from face center to camera
    view_vectors = camera_position.reshape(1, 3) - face_centers
    
    # Normalize vectors
    view_vector_lengths = np.linalg.norm(view_vectors, axis=1, keepdims=True)
    view_vectors = view_vectors / view_vector_lengths
    
    # Calculate face normals for additional ear culling refinement
    normals = calculate_face_normals(vertices, faces)
    
    # Dot product between face normals and view vectors
    # (additional check to refine ear culling)
    face_dots = np.sum(normals * view_vectors, axis=1)
    
    # Calculate camera's horizontal angle for ear decisions
    if camera_matrix is not None and rvec is not None:
        # Get camera's Z axis (forward direction)
        R, _ = cv2.Rodrigues(rvec)
        camera_z_axis = R[:, 2]
        
        # Project onto XZ plane
        camera_xz = np.array([camera_z_axis[0], 0, camera_z_axis[2]])
        camera_xz_norm = np.linalg.norm(camera_xz)
        
        if camera_xz_norm > 1e-5:
            camera_xz /= camera_xz_norm
            camera_yaw = np.arctan2(camera_xz[0], camera_xz[2])
            camera_yaw_degrees = np.degrees(camera_yaw)
            
            # For ears, we combine OpenGL-style winding order with view-dependent culling
            if camera_yaw_degrees > 25:  # Looking at left side
                # Hide right ear completely
                for face_idx in right_ear_faces:
                    front_facing[face_idx] = False
                    
                # For left ear, combine winding order with stricter dot product check
                for face_idx in left_ear_faces:
                    if face_dots[face_idx] < 0.2:  # Ear faces must be clearly facing camera
                        front_facing[face_idx] = False
                        
            elif camera_yaw_degrees < -25:  # Looking at right side
                # Hide left ear completely
                for face_idx in left_ear_faces:
                    front_facing[face_idx] = False
                    
                # For right ear, combine winding order with stricter dot product check
                for face_idx in right_ear_faces:
                    if face_dots[face_idx] < 0.2:
                        front_facing[face_idx] = False
            else:
                # Near-frontal view - apply stricter test to both ears
                for face_idx in left_ear_faces + right_ear_faces:
                    if face_dots[face_idx] < 0.3:
                        front_facing[face_idx] = False
    
    return front_facing

def remove_eyeballs(verts, faces, start_idx=3931, end_idx=5022):
    """
    Remove eyeball vertices from the mesh and update faces accordingly
    
    Args:
        verts: Array of vertex positions
        faces: Array of face indices
        start_idx: Start index of eyeball vertices (inclusive)
        end_idx: End index of eyeball vertices (inclusive)
        
    Returns:
        new_verts: Updated vertices array without eyeball vertices
        new_faces: Updated faces array with adjusted indices
    """
    # Create a mask for vertices to keep
    keep_mask = np.ones(verts.shape[0], dtype=bool)
    keep_mask[start_idx:end_idx+1] = False
    
    # Create index mapping from old to new vertex indices
    old_to_new = np.cumsum(keep_mask) - 1
    
    # Get new vertices
    new_verts = verts[keep_mask]
    
    # Keep only faces that don't use eyeball vertices
    valid_faces = []
    for face in faces:
        if np.all((face < start_idx) | (face > end_idx)):
            # Remap vertex indices
            new_face = [old_to_new[idx] for idx in face]
            valid_faces.append(new_face)
    
    new_faces = np.array(valid_faces)
    
    print(f"Removed {end_idx - start_idx + 1} eyeball vertices")
    print(f"Faces reduced from {len(faces)} to {len(new_faces)}")
    
    return new_verts, new_faces