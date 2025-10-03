import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cuboid(ax, position, size, color='lightgray', alpha=0.7):
    """
    Draws a 3D cuboid (rectangular prism) on the given axes.
    
    Args:
        ax: The matplotlib 3D axes object.
        position (tuple): The (x, y, z) coordinate of the bottom-front-left corner.
        size (tuple): The (width, depth, height) of the cuboid.
        color (str): The face color of the cuboid.
        alpha (float): The transparency of the cuboid.
    """
    x, y, z = position
    dx, dy, dz = size
    
    # Define the 8 vertices of the cuboid
    vertices = [
        (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
        (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)
    ]
    
    # Define the 6 faces using the vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]], # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]]  # Right
    ]
    
    # Create the 3D polygon collection
    poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha, zorder = 1)
    ax.add_collection3d(poly3d)

def plot_scene_with_cuboids(buildings, path, start, end, grid_dims):
    """
    Visualizes the 3D scene using efficient cuboids for obstacles.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot the obstacles (buildings) as cuboids ---
    for b in buildings:
        draw_cuboid(ax, b['pos'], b['size'])

    # --- Plot the path ---
    if path:
        path_x, path_y, path_z = zip(*path)
        ax.plot(path_x, path_y, path_z, color='blue', marker='o', linewidth=3, markersize=8, label='Path', zorder= 10)

    # --- Plot Start and End points ---
    ax.scatter(start[0], start[1], start[2], color='green', s=250, edgecolor='black', depthshade=False, label='Start', zorder=11)
    ax.scatter(end[0], end[1], end[2], color='red', s=250, edgecolor='black', depthshade=False, label='End', zorder=11)

    # --- Customize the plot ---
    ax.set_xlabel('X axis'), ax.set_ylabel('Y axis'), ax.set_zlabel('Z axis')
    ax.set_title('3D Path Planning with Cuboid Obstacles')
    ax.set_xlim(0, grid_dims[0]), ax.set_ylim(0, grid_dims[1]), ax.set_zlim(0, grid_dims[2])
    ax.view_init(elev=30, azim=-65)
    ax.legend()
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    grid_dims = (60, 60, 40)
    
    # --- Define buildings as a list of cuboids {position, size} ---
    # This is much more efficient than creating and iterating through a huge grid.
    buildings = [
        # Downtown District
        {'pos': (5, 5, 0), 'size': (10, 10, 35)},    # Skyscraper 1
        {'pos': (5, 22, 0), 'size': (10, 10, 30)},   # Skyscraper 2
        {'pos': (22, 5, 0), 'size': (10, 10, 28)},   # Skyscraper 3
        # Commercial Block
        {'pos': (22, 22, 0), 'size': (10, 13, 18)},  # Large office building
        {'pos': (20, 42, 0), 'size': (12, 6, 22)},   # Long commercial building
        # Residential Area
        {'pos': (42, 5, 0), 'size': (6, 10, 12)},    # Apartment 1
        {'pos': (42, 20, 0), 'size': (6, 10, 10)},   # Apartment 2
        {'pos': (42, 35, 0), 'size': (6, 10, 14)},   # Apartment 3
        {'pos': (42, 50, 0), 'size': (6, 8, 11)},    # Apartment 4
        # Industrial Zone
        {'pos': (5, 45, 0), 'size': (10, 10, 8)},    # Warehouse 1
        {'pos': (20, 50, 0), 'size': (15, 8, 10)},   # Warehouse 2
        # Additional Structures
        {'pos': (37, 5, 0), 'size': (3, 45, 5)},     # Long low wall/barrier
        {'pos': (55, 10, 0), 'size': (3, 40, 15)},   # City Wall / Elevated rail
    ]

    # --- Define the path data ---
    start_point = (2, 2, 0)
    end_point = (52, 52, 0)
    path_coordinates = [(2, 2, 0), (45, 5, 15), (52, 52, 0)]
    
    # --- PLOT THE RESULT ---
    print("Plotting the 3D environment using efficient cuboids...")
    plot_scene_with_cuboids(buildings, path_coordinates, start_point, end_point, grid_dims)