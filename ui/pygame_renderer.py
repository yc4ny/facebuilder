"""
Copyright (c) SECERN AI, Inc. and its affiliates.
Pygame rendering module.

This module handles rendering of the 3D mesh, UI elements, and 3D visualization
using the Pygame library.
"""
import pygame
import numpy as np
import cv2
from config import Mode
from utils.geometry import calculate_front_facing

def cv2_to_pygame(cv2_image):
    """
    Convert OpenCV image (BGR) to Pygame surface (RGB).
    
    Args:
        cv2_image: OpenCV image in BGR format
        
    Returns:
        pygame.Surface: Pygame surface in RGB format
    """
    # Convert color format from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Create Pygame surface from the RGB image
    # Note: Pygame expects dimensions in (width, height) but numpy arrays are (height, width)
    # so we need to swap axes
    pygame_surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
    
    return pygame_surface

def draw_mesh_pygame(surface, state):
    """
    Draw the mesh and UI using Pygame.
    
    This is the main rendering function that draws the current view, including
    the mesh, pins, landmarks, and UI elements.
    
    Args:
        surface: Pygame surface to draw on
        state: Application state containing mesh and UI data
    """
    # Clear surface with the background image
    surface.blit(cv2_to_pygame(state.overlay), (0, 0))
    
    # Get UI dimensions for consistent rendering
    ui = state.ui_dimensions
    
    # Check if we're in 3D view mode
    if state.mode == Mode.VIEW_3D:
        draw_3d_view_pygame(surface, state)
        return
    
    # Calculate which faces are front-facing using the current camera parameters
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
            # Get vertex coordinates and convert to integers for drawing
            p0 = (int(round(state.verts2d[i0][0])), int(round(state.verts2d[i0][1])))
            p1 = (int(round(state.verts2d[i1][0])), int(round(state.verts2d[i1][1])))
            p2 = (int(round(state.verts2d[i2][0])), int(round(state.verts2d[i2][1])))
            
            # Draw triangle edges as lines
            pygame.draw.line(surface, (0, 0, 0), p0, p1, ui['line_thickness'])
            pygame.draw.line(surface, (0, 0, 0), p1, p2, ui['line_thickness'])
            pygame.draw.line(surface, (0, 0, 0), p2, p0, ui['line_thickness'])
    
    # Draw vertices (only those used by front-facing faces)
    # This avoids drawing invisible vertices that are part of back-facing triangles
    visible_vertices = set()
    for i, face in enumerate(state.faces):
        if state.front_facing[i]:
            visible_vertices.update(face)
    
    for i in visible_vertices:
        x, y = state.verts2d[i]
        pygame.draw.circle(surface, (0, 0, 255), 
                          (int(round(x)), int(round(y))), 
                          ui['vertex_radius'])
    
    # Check if we're dragging a pin (to update pin colors) or if pins have been moved previously
    is_dragging = state.drag_index != -1
    pins_have_been_moved = hasattr(state, 'pins_moved') and state.pins_moved
    
    # Draw landmarks (facial feature points)
    landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
    if not landmark_pins_hidden:
        for i, (lx, ly) in enumerate(state.landmark_positions):
            # Determine color: green for the dragged landmark, red for others
            color = (0, 255, 0) if is_dragging and state.drag_index == i else (255, 0, 0)
            pygame.draw.circle(surface, color, 
                              (int(round(lx)), int(round(ly))), 
                              ui['landmark_radius'])
    
    # Draw custom pins for the current image
    for i, pin_data in enumerate(state.pins_per_image[state.current_image_idx]):
        if len(pin_data) >= 2:
            px, py = pin_data[0], pin_data[1]
            # Calculate actual pin index (offset by landmark count)
            actual_pin_idx = i + len(state.landmark_positions)
            
            # Determine color based on multiple factors:
            # 1. Green for the currently dragged pin
            # 2. Red for all pins if any pin has been moved previously (pins_moved is True)
            # 3. Red for all pins except the dragged one when dragging
            # 4. Green for new pins if no dragging has occurred yet
            if is_dragging and state.drag_index == actual_pin_idx:
                # Currently dragged pin is always green
                color = (0, 255, 0)
            elif pins_have_been_moved or is_dragging:
                # All pins are red after any movement or during dragging (except dragged pin)
                color = (255, 0, 0)
            else:
                # Default green for new pins
                color = (0, 255, 0)
                
            pygame.draw.circle(surface, color, 
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
    
    # Show current mode - Include landmarks in pin count for mode determination
    if state.mode == Mode.TOGGLE_PINS:
        mode_text = "Mode: TOGGLE PINS"
    else:
        # Count active pins including landmarks to determine mode
        custom_pin_count = len(state.pins_per_image[state.current_image_idx])
        landmark_pins_hidden = hasattr(state, 'landmark_pins_hidden') and state.landmark_pins_hidden
        
        total_active_pins = custom_pin_count
        if not landmark_pins_hidden:
            total_active_pins += len(state.landmark_positions)
        
        # Different modes based on number of active pins
        if total_active_pins == 1:
            mode_text = "Mode: SINGLE PIN (MOVE+ROTATE)"
        elif total_active_pins == 2:
            mode_text = "Mode: DUAL PIN (SCALE+ROTATE)"
        elif total_active_pins == 3:
            mode_text = "Mode: TRIPLE PIN (3D ROTATION)"
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
    """
    Render a 3D visualization of the mesh.
    
    This function draws a 3D view of the mesh with proper lighting, shading,
    and perspective, allowing the user to view the mesh from any angle.
    
    Args:
        surface: Pygame surface to draw on
        state: Application state containing mesh and camera data
    """
    ui = state.ui_dimensions
    
    # Create a gradient background for better visual effect
    for y in range(state.img_h):
        blue_val = int(220 - 50 * (y / state.img_h))
        pygame.draw.line(surface, (240, 240, blue_val), (0, y), (state.img_w, y))
    
    # Set up camera parameters for 3D projection
    focal_length = max(state.img_w, state.img_h)
    camera_matrix = np.array([
        [focal_length, 0, state.img_w / 2],
        [0, focal_length, state.img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create rotation matrices for x and y axes based on view rotation angles
    # Rx rotates around x-axis:
    # [1      0       0    ]
    # [0  cos(θx) -sin(θx) ]
    # [0  sin(θx)  cos(θx) ]
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(state.view_3d_rotation_x), -np.sin(state.view_3d_rotation_x)],
        [0, np.sin(state.view_3d_rotation_x), np.cos(state.view_3d_rotation_x)]
    ])
    
    # Ry rotates around y-axis:
    # [ cos(θy)  0  sin(θy) ]
    # [    0     1     0    ]
    # [-sin(θy)  0  cos(θy) ]
    Ry = np.array([
        [np.cos(state.view_3d_rotation_y), 0, np.sin(state.view_3d_rotation_y)],
        [0, 1, 0],
        [-np.sin(state.view_3d_rotation_y), 0, np.cos(state.view_3d_rotation_y)]
    ])
    
    # Combined rotation matrix: R = Ry × Rx (apply Y rotation first, then X rotation)
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
    
    # Project the 3D vertices to 2D using equation: x' = K[R|t]X
    projected_verts, _ = cv2.projectPoints(
        verts_scaled, rvec, tvec, camera_matrix, np.zeros((4, 1))
    )
    projected_verts = projected_verts.reshape(-1, 2)
    
    # Calculate which faces are front-facing to implement backface culling
    state.front_facing = calculate_front_facing(
        verts_scaled, state.faces,
        camera_matrix=camera_matrix, rvec=rvec, tvec=tvec
    )
    
    # Sort faces by depth for better rendering (painter's algorithm)
    # This ensures faces further away are drawn first, then closer faces on top
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
    
    # Draw the mesh triangles with filled polygons and shading
    for face_idx in sorted_faces:
        i0, i1, i2 = state.faces[face_idx]
        p0 = (int(round(projected_verts[i0][0])), int(round(projected_verts[i0][1])))
        p1 = (int(round(projected_verts[i1][0])), int(round(projected_verts[i1][1])))
        p2 = (int(round(projected_verts[i2][0])), int(round(projected_verts[i2][1])))
        
        # Calculate face normal for lighting using cross product of edges
        # n = (v1-v0) × (v2-v0) / ||(v1-v0) × (v2-v0)||
        v0 = verts_scaled[i0]
        v1 = verts_scaled[i1]
        v2 = verts_scaled[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)  # Normalize with epsilon to avoid division by zero
        
        # Direction to camera (for lighting calculation)
        view_dir = np.array([0, 0, 1])
        
        # Calculate diffuse lighting based on normal and view direction
        # I = N·V (dot product of normal and view direction)
        light_intensity = np.dot(normal, view_dir)
        
        # Add ambient lighting to ensure all faces are visible
        # I' = min(0.95, max(0.3, I + 0.6))
        light_intensity = max(0.3, min(0.95, light_intensity + 0.6))
        
        # Silver/metallic color for the mesh (uniform for all triangles)
        base_color = np.array([192, 192, 192])  # Silver gray
        
        # Apply lighting: final_color = base_color * light_intensity
        colored = base_color * light_intensity
        color = (int(colored[0]), int(colored[1]), int(colored[2]))
        
        # Fill the triangle with this color
        pygame.draw.polygon(surface, color, [p0, p1, p2])
        
        # Draw edges of triangles for better definition
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
    
    # Add 3D view mode indicator
    view_mode_text = "3D VIEW MODE"
    view_mode_surf = state.fonts['status'].render(view_mode_text, True, (0, 0, 120))
    view_mode_rect = view_mode_surf.get_rect(center=(state.img_w // 2, state.img_h - 50))
    surface.blit(view_mode_surf, view_mode_rect)