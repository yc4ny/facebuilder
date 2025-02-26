import numpy as np

def move_mesh_2d(state, old_lx, old_ly, dx, dy):
    """Move the mesh vertices based on the dragged point"""
    # Calculate distance from each vertex to the dragged point
    diffs = state.verts2d - np.array([old_lx, old_ly])
    dists_sq = np.sum(diffs**2, axis=1)
    
    # Calculate radial weights - inverse distance weighting
    radial_w = 1.0 / (1.0 + state.alpha * dists_sq)
    
    # If dragging a landmark, use skinning weights for better deformation
    if state.drag_index != -1 and state.drag_index < len(state.landmark_positions):
        f_idx = state.face_for_lmk[state.drag_index]
        v_indices = state.faces[f_idx]
        avg_weight = np.mean(state.weights[v_indices, :], axis=0)
        norm_avg = avg_weight / np.sum(avg_weight)
        joint_w = np.dot(state.weights, norm_avg)
        combined_w = radial_w * joint_w
    elif state.drag_index != -1:  # Dragging a custom pin
        pin_idx = state.drag_index - len(state.landmark_positions)
        if pin_idx < len(state.pins_per_image[state.current_image_idx]):
            _, _, face_idx, _ = state.pins_per_image[state.current_image_idx][pin_idx]
            v_indices = state.faces[face_idx]
            avg_weight = np.mean(state.weights[v_indices, :], axis=0)
            norm_avg = avg_weight / np.sum(avg_weight)
            joint_w = np.dot(state.weights, norm_avg)
            combined_w = radial_w * joint_w
    else:
        combined_w = radial_w
    
    # Apply the calculated weights to move vertices
    shift = np.column_stack((combined_w * dx, combined_w * dy))
    state.verts2d += shift
