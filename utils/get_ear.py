# Copyright (c) CUBOX, Inc. and its affiliates.
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import networkx as nx

def identify_ear_regions(flame_path="data/flame2023.pkl", threshold=0.06, proximity_radius=0.01):

    # Load FLAME model
    with open(flame_path, "rb") as f:
        flame_data = pickle.load(f, encoding='latin1')
    
    vertices = flame_data['v_template']
    faces = flame_data['f']
    
    # Calculate center of the head
    center = np.mean(vertices, axis=0)
    
    # Find vertices with high absolute x values (ears are on the sides)
    x_distances = np.abs(vertices[:, 0] - center[0])
    
    # Identify potential left and right ear vertices based on X coordinate
    potential_left_ear = np.where((vertices[:, 0] - center[0]) < -threshold)[0]
    potential_right_ear = np.where((vertices[:, 0] - center[0]) > threshold)[0]
    
    # Build KD-Trees for efficient proximity queries
    kdtree_left = KDTree(vertices[potential_left_ear])
    kdtree_right = KDTree(vertices[potential_right_ear])
    
    # Function to find connected components in a region
    def extract_connected_components(indices, all_vertices):
        # Build graph with vertices as nodes and edges for vertices within proximity_radius
        G = nx.Graph()
        
        # Add all vertices as nodes
        for i, idx in enumerate(indices):
            G.add_node(idx)
        
        # Add edges between nodes that are close to each other
        for i, idx1 in enumerate(indices):
            # Find all neighbors within proximity radius
            neighbors = kdtree_left.query_ball_point(all_vertices[idx1], proximity_radius) if idx1 in potential_left_ear else \
                       kdtree_right.query_ball_point(all_vertices[idx1], proximity_radius)
            
            # Convert neighbor indices from KDTree to original vertex indices
            if idx1 in potential_left_ear:
                neighbor_indices = [potential_left_ear[j] for j in neighbors]
            else:
                neighbor_indices = [potential_right_ear[j] for j in neighbors]
            
            # Add edges
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in indices and neighbor_idx != idx1:
                    G.add_edge(idx1, neighbor_idx)
        
        # Find connected components
        connected_components = list(nx.connected_components(G))
        
        # Sort components by size (largest first)
        connected_components.sort(key=len, reverse=True)
        
        return connected_components
    
    # Extract connected components for left and right sides
    left_connected = extract_connected_components(potential_left_ear, vertices)
    right_connected = extract_connected_components(potential_right_ear, vertices)
    
    # The ear is typically the largest connected component on each side
    # However, we should verify this isn't part of the face by checking its position
    
    # Function to check if a component is likely an ear
    def is_ear_component(component_vertices, is_left):
        # Calculate average Y and Z positions
        avg_y = np.mean(vertices[list(component_vertices)][:, 1])
        avg_z = np.mean(vertices[list(component_vertices)][:, 2])
        
        # Ears are typically in the middle third of the head's height (Y-axis)
        # and behind the center point (Z-axis for many FLAME models)
        y_center = center[1]
        z_center = center[2]
        
        # Check if component is at appropriate height and depth
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])
        y_range = y_max - y_min
        
        # Y position check: should be in middle third of head height
        y_ok = (avg_y > y_min + y_range * 0.3) and (avg_y < y_max - y_range * 0.3)
        
        # Z position check: should be behind the center of the head
        z_ok = avg_z < z_center
        
        return y_ok and z_ok
    
    # Extract actual ear vertices
    left_ear_components = [comp for comp in left_connected if is_ear_component(comp, True)]
    right_ear_components = [comp for comp in right_connected if is_ear_component(comp, False)]
    
    # If we found any components that match our criteria, use the largest one on each side
    left_ear_vertices = list(left_ear_components[0]) if left_ear_components else []
    right_ear_vertices = list(right_ear_components[0]) if right_ear_components else []
    
    # If we have mesh connectivity information (faces), we can further refine by adding
    # vertices that are directly connected to ear vertices through mesh edges
    
    # Function to expand component using face connectivity
    def expand_component_with_faces(component_vertices, all_faces, expansion_steps=1):
        expanded = set(component_vertices)
        for _ in range(expansion_steps):
            neighbors = set()
            for face in all_faces:
                # If at least one vertex of the face is in our component
                # and not all vertices are in the component, add the missing ones
                face_verts = set(face)
                if face_verts & expanded and not face_verts.issubset(expanded):
                    neighbors.update(face_verts - expanded)
            expanded.update(neighbors)
        return expanded
    
    # Add connected vertices based on mesh topology
    expanded_left = expand_component_with_faces(left_ear_vertices, faces, expansion_steps=1)
    expanded_right = expand_component_with_faces(right_ear_vertices, faces, expansion_steps=1)
    
    # Combined ear vertices
    ear_vertices = np.array(list(expanded_left) + list(expanded_right))
    left_ear_vertices = np.array(list(expanded_left))
    right_ear_vertices = np.array(list(expanded_right))
    
    # Find faces that contain ear vertices
    ear_faces = []
    for i, face in enumerate(faces):
        if any(vertex in ear_vertices for vertex in face):
            ear_faces.append(i)
    
    return ear_vertices, np.array(ear_faces), left_ear_vertices, right_ear_vertices

def visualize_ear_regions(verts, faces, ear_vertices, left_ear=None, right_ear=None, save_path="ear_regions.png"):
    """
    Visualize the identified ear regions for verification.
    
    Args:
        verts: Vertex array from FLAME model
        faces: Face array from FLAME model
        ear_vertices: Indices of vertices identified as ear regions
        left_ear: Indices of left ear vertices (optional)
        right_ear: Indices of right ear vertices (optional)
        save_path: Path to save the visualization
    """
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot full head with ears highlighted
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create color array - 0 for non-ear vertices
    colors = np.zeros(len(verts))
    
    # Color ear vertices
    if left_ear is not None and right_ear is not None:
        colors[left_ear] = 1  # Left ear (red)
        colors[right_ear] = 2  # Right ear (blue)
    else:
        colors[ear_vertices] = 1  # All ears (red)
    
    # Plot vertices
    ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=colors, s=1, cmap='coolwarm')
    ax1.set_title("Full Head with Ear Regions Highlighted")
    
    # Plot just the ear vertices
    ax2 = fig.add_subplot(122, projection='3d')
    if left_ear is not None and right_ear is not None:
        # Plot left ear in red
        ax2.scatter(verts[left_ear, 0], verts[left_ear, 1], verts[left_ear, 2], 
                   c='red', s=3, label='Left Ear')
        # Plot right ear in blue
        ax2.scatter(verts[right_ear, 0], verts[right_ear, 1], verts[right_ear, 2],
                   c='blue', s=3, label='Right Ear')
        ax2.legend()
    else:
        # Plot all ear vertices in red
        ax2.scatter(verts[ear_vertices, 0], verts[ear_vertices, 1], verts[ear_vertices, 2], 
                   c='red', s=3)
    ax2.set_title("Isolated Ear Vertices")
    
    # Add overall title
    plt.suptitle("FLAME Model - Ear Region Detection")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {save_path}")
    
    # Return the number of ear vertices for reference
    return len(ear_vertices)

def write_ear_indices_to_file(left_ear, right_ear, file_path="data/ear_indices.json"):
    """
    Write the ear vertex indices to a Python file for easy import in other scripts.
    
    Args:
        left_ear: Left ear vertex indices
        right_ear: Right ear vertex indices
        file_path: Path to save the Python file
    """
    with open(file_path, "w") as f:
        f.write("# Ear vertex indices for FLAME model\n\n")
        f.write("# Left ear vertex indices\n")
        f.write(f"LEFT_EAR_VERTICES = {list(left_ear)}\n\n")
        f.write("# Right ear vertex indices\n")
        f.write(f"RIGHT_EAR_VERTICES = {list(right_ear)}\n")
    
    print(f"Ear indices written to {file_path}")

if __name__ == "__main__":
    try:
        # Try to import networkx, which is needed for connected component analysis
        import networkx as nx
    except ImportError:
        print("Networkx library is required. Please install it using:")
        print("pip install networkx")
        exit(1)
        
    ear_verts, ear_faces, left_ear, right_ear = identify_ear_regions()
    
    # Load FLAME model to visualize
    with open("data/flame2023.pkl", "rb") as f:
        flame_data = pickle.load(f, encoding='latin1')
    
    verts = flame_data['v_template']
    faces = flame_data['f']
    
    print(f"Total vertices in model: {len(verts)}")
    
    # Apply eyeball removal to match your main application
    from utils.geometry import remove_eyeballs
    verts, faces = remove_eyeballs(verts, faces, 3931, 5022)
    
    # Visualize ears
    print(f"Left ear vertices: {len(left_ear)}")
    print(f"Right ear vertices: {len(right_ear)}")
    print(f"Total ear vertices: {len(ear_verts)}")
    print(f"Total ear faces: {len(ear_faces)}")
    
    ear_verts_count = visualize_ear_regions(verts, faces, ear_verts, left_ear, right_ear)
    
    print(f"{ear_verts_count} ear vertices identified and visualized")
    
    # Save ear indices to file for easy use in other scripts
    write_ear_indices_to_file(left_ear, right_ear)