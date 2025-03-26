# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import numpy as np
from config import WINDOW_NAME, Mode

def redraw(state):
    """Redraw the UI and mesh"""
    # If in 3D view mode, render 3D visualization
    if state.mode == Mode.VIEW_3D:
        draw_3d_view(state)
        return
        
    disp = state.overlay.copy()
    
    # Calculate UI dimensions based on current image size
    ui = state.ui_dimensions
    
    # Check if we have determined which faces are front-facing
    # If not, default to showing all faces
    front_facing = state.front_facing if state.front_facing is not None else np.ones(len(state.faces), dtype=bool)
    
    # Draw mesh triangles (only front-facing)
    for i, (i0, i1, i2) in enumerate(state.faces):
        if front_facing[i]:  # Only draw front-facing faces
            p0 = tuple(np.round(state.verts2d[i0]).astype(int))
            p1 = tuple(np.round(state.verts2d[i1]).astype(int))
            p2 = tuple(np.round(state.verts2d[i2]).astype(int))
            cv2.line(disp, p0, p1, (0,0,0), ui['line_thickness'])
            cv2.line(disp, p1, p2, (0,0,0), ui['line_thickness'])
            cv2.line(disp, p2, p0, (0,0,0), ui['line_thickness'])
    
    # Draw vertices (only those used by front-facing faces)
    visible_vertices = set()
    for i, face in enumerate(state.faces):
        if front_facing[i]:
            visible_vertices.update(face)
    
    for i in visible_vertices:
        x, y = state.verts2d[i]
        cv2.circle(disp, (int(round(x)), int(round(y))), ui['vertex_radius'], (255,0,0), -1)
    
    # Draw landmarks
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    if not landmark_pins_hidden:
        for (lx, ly) in state.landmark_positions:
            cv2.circle(disp, (int(round(lx)), int(round(ly))), ui['landmark_radius'], (0,0,255), -1)
    
    # Draw custom pins for the current image
    for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
        # Handle both 4-tuple and 5-tuple pin formats for backward compatibility
        if len(pin_data) >= 2:  # At minimum we need x,y
            px, py = pin_data[0], pin_data[1]
            pin_color = (0,255,0)
            cv2.circle(disp, (int(round(px)), int(round(py))), ui['pin_radius'], pin_color, -1)
    
    # Helper function to center text in button
    def draw_button_with_text(rect, text, is_active=False):
        bx, by, bw, bh = rect
        button_color = (100,100,255) if is_active else (50,50,50)
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), button_color, -1)
        
        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                     ui['text_size'], ui['text_thickness'])
        text_x = bx + (bw - text_width) // 2
        text_y = by + (bh + text_height) // 2
        
        cv2.putText(disp, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, ui['text_size'], (255,255,255), ui['text_thickness'])
    
    # Draw all buttons in the new order including 3D View
    draw_button_with_text(ui['center_geo_button_rect'], "Center Geo")
    draw_button_with_text(ui['align_button_rect'], "Align Face")
    draw_button_with_text(ui['reset_shape_button_rect'], "Reset Shape")
    draw_button_with_text(ui['remove_pins_button_rect'], "Remove Pins")
    draw_button_with_text(ui['toggle_pins_button_rect'], "Toggle Pins", state.mode == Mode.TOGGLE_PINS)
    draw_button_with_text(ui['save_button_rect'], "Save Mesh")
    draw_button_with_text(ui['next_img_button_rect'], "Next Image")
    draw_button_with_text(ui['prev_img_button_rect'], "Prev Image")
    draw_button_with_text(ui['visualizer_button_rect'], "3D View", state.mode == Mode.VIEW_3D)
    
    # Show current image index
    status_x, status_y1, status_y2 = ui['status_text_pos']
    cv2.putText(disp, f"Image {state.current_image_idx+1}/{len(state.images)}", 
                (status_x, status_y1), cv2.FONT_HERSHEY_SIMPLEX, 
                ui['text_size'] * 2, (255,120,0), ui['text_thickness']*2)
    
    # Show current mode
    if state.mode == Mode.TOGGLE_PINS:
        mode_text = "Mode: TOGGLE PINS"
    else:
        # Count active pins to determine if we're in single pin mode or regular move mode
        custom_pin_count = len(state.pins_per_image[state.current_image_idx])
        landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
        
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(state.landmark_positions)
        
        if total_active_pins == 1:
            mode_text = "Mode: SINGLE PIN (MOVE+ROTATE)"
        else:
            mode_text = "Mode: MULTI-PIN DEFORM"
    
    cv2.putText(disp, mode_text, (status_x, status_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, ui['text_size']*2, (255,120,0), ui['text_thickness']*2)
    
    # Add a user hint for adding pins
    # Calculate hint position based on the UI dimensions
    hint_y = status_y2 + int(ui['landmark_radius'] * 8)
    cv2.putText(disp, "Click on mesh to add pins", 
                (status_x, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 
                ui['text_size']*1.5, (0,200,0), ui['text_thickness'])
    
    cv2.imshow(WINDOW_NAME, disp)


def draw_3d_view(state):
    """Render a 3D visualization of the mesh"""
    ui = state.ui_dimensions
    
    # Create a blank background with gradient
    disp = np.ones((state.img_h, state.img_w, 3), dtype=np.uint8) * 240
    
    # Create gradient background (light blue to darker blue)
    for y in range(state.img_h):
        blue_val = int(220 - 50 * (y / state.img_h))
        disp[y, :] = (240, 240, blue_val)
    
    # Set up camera parameters
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize rotation angles if not set
    if not hasattr(state, 'view_3d_rotation_x'):
        state.view_3d_rotation_x = 0.0
        state.view_3d_rotation_y = 0.0
    
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
    R = Ry @ Rx  # Apply Y rotation first, then X rotation
    rvec, _ = cv2.Rodrigues(R)
    
    # Center the model and scale it appropriately
    verts_center = np.mean(state.verts3d, axis=0)
    verts_centered = state.verts3d - verts_center
    
    # Calculate appropriate scale to fit in window
    base_scale = min(state.img_w, state.img_h) * 0.3 / np.max(np.abs(verts_centered))
    
    # Apply zoom factor if it exists
    if hasattr(state, 'view_3d_zoom'):
        zoom_scale = base_scale * state.view_3d_zoom
    else:
        # Initialize zoom factor and use base scale
        state.view_3d_zoom = 0.7
        zoom_scale = base_scale * 0.7
        
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
    front_facing = calculate_front_facing(
        verts_scaled, state.faces, 
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Sort faces by depth for better rendering
    face_depths = []
    for i, (i0, i1, i2) in enumerate(state.faces):
        # Calculate depth of face center in camera space
        center_3d = (verts_scaled[i0] + verts_scaled[i1] + verts_scaled[i2]) / 3
        center_rotated = R @ center_3d
        depth = center_rotated[2] + tvec[2, 0]  # Z in camera space
        face_depths.append((i, depth))
    
    # Sort faces back-to-front
    sorted_faces = [idx for idx, _ in sorted(face_depths, key=lambda x: x[1], reverse=True)]
    
    # Draw the mesh triangles with filled polygons
    for face_idx in sorted_faces:
        # Render all faces, not just front-facing ones
        i0, i1, i2 = state.faces[face_idx]
        p0 = tuple(np.round(projected_verts[i0]).astype(int))
        p1 = tuple(np.round(projected_verts[i1]).astype(int))
        p2 = tuple(np.round(projected_verts[i2]).astype(int))
        
        # Create a triangle polygon
        triangle = np.array([p0, p1, p2], dtype=np.int32)
        
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
        
        # Calculate diffuse lighting based on normal and view direction
        # Cosine between normal and view direction
        light_intensity = np.dot(normal, view_dir)
        
        # Add ambient lighting to ensure all faces are visible
        light_intensity = max(0.3, min(0.95, light_intensity + 0.6))  # Add ambient and clamp
        
        # Silver/metallic color for the mesh (uniform for all triangles)
        base_color = np.array([192, 192, 192])  # Silver gray
        
        # Apply lighting - explicitly convert to proper BGR tuple format
        colored = base_color * light_intensity
        color = (int(colored[0]), int(colored[1]), int(colored[2]))
        
        # Fill the triangle with this color
        cv2.fillPoly(disp, [triangle], color)
        
        # Draw edges of triangles
        cv2.line(disp, p0, p1, (80, 80, 80), ui['line_thickness'])
        cv2.line(disp, p1, p2, (80, 80, 80), ui['line_thickness'])
        cv2.line(disp, p2, p0, (80, 80, 80), ui['line_thickness'])
    
    # Helper function to center text in button
    def draw_button_with_text(rect, text, is_active=False):
        bx, by, bw, bh = rect
        button_color = (100,100,255) if is_active else (50,50,50)
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), button_color, -1)
        
        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                     ui['text_size'], ui['text_thickness'])
        text_x = bx + (bw - text_width) // 2
        text_y = by + (bh + text_height) // 2
        
        cv2.putText(disp, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, ui['text_size'], (255,255,255), ui['text_thickness'])
    
    # Draw all buttons in the new order including 3D View
    draw_button_with_text(ui['center_geo_button_rect'], "Center Geo")
    draw_button_with_text(ui['align_button_rect'], "Align Face")
    draw_button_with_text(ui['reset_shape_button_rect'], "Reset Shape")
    draw_button_with_text(ui['remove_pins_button_rect'], "Remove Pins")
    draw_button_with_text(ui['toggle_pins_button_rect'], "Toggle Pins", state.mode == Mode.TOGGLE_PINS)
    draw_button_with_text(ui['save_button_rect'], "Save Mesh")
    draw_button_with_text(ui['next_img_button_rect'], "Next Image")
    draw_button_with_text(ui['prev_img_button_rect'], "Prev Image")
    draw_button_with_text(ui['visualizer_button_rect'], "3D View", state.mode == Mode.VIEW_3D)
    
    # Add instruction text for 3D view
    instruction_y = ui['landmark_radius'] * 6 + ui['center_geo_button_rect'][1] + ui['center_geo_button_rect'][3] + 10
    cv2.putText(disp, "DRAG to rotate | SCROLL to zoom", 
                (state.img_w - 350, instruction_y), 
                cv2.FONT_HERSHEY_SIMPLEX, ui['text_size']*1.5, 
                (0, 0, 0), ui['text_thickness'])
    
    cv2.putText(disp, "3D VIEW MODE", 
                (state.img_w // 2 - 100, state.img_h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, ui['text_size']*2, 
                (0, 0, 120), ui['text_thickness']*2)
    
    cv2.imshow(WINDOW_NAME, disp)