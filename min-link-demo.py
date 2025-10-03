import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
import math
import time
from scipy.ndimage import binary_dilation
from numba import njit

# ==============================================================================
# 3D HELPER & CORE ALGORITHM FUNCTIONS
# ==============================================================================

def euclidean_dist_3d(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

@njit
def line_of_sight_3d(grid, p1, p2):
    """
    Checks for a clear line of sight between two 3D points using a 3D
    Bresenham's line algorithm.
    """
    x1, y1, z1 = int(p1[0]), int(p1[1]), int(p1[2])
    x2, y2, z2 = int(p2[0]), int(p2[1]), int(p2[2])
    
    if grid[x1, y1, z1] == 1 or grid[x2, y2, z2] == 1:
        return False

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    sx = -1 if dx < 0 else 1
    sy = -1 if dy < 0 else 1
    sz = -1 if dz < 0 else 1
    
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    
    if ax >= ay and ax >= az:
        err_y, err_z = 2 * ay - ax, 2 * az - ax
        for _ in range(ax):
            if grid[x1, y1, z1] == 1: return False
            if err_y > 0: y1 += sy; err_y -= 2 * ax
            if err_z > 0: z1 += sz; err_z -= 2 * ax
            err_y += 2 * ay; err_z += 2 * az; x1 += sx
    elif ay >= ax and ay >= az:
        err_x, err_z = 2 * ax - ay, 2 * az - ay
        for _ in range(ay):
            if grid[x1, y1, z1] == 1: return False
            if err_x > 0: x1 += sx; err_x -= 2 * ay
            if err_z > 0: z1 += sz; err_z -= 2 * ay
            err_x += 2 * ax; err_z += 2 * az; y1 += sy
    else:
        err_x, err_y = 2 * ax - az, 2 * ay - az
        for _ in range(az):
            if grid[x1, y1, z1] == 1: return False
            if err_x > 0: x1 += sx; err_x -= 2 * az
            if err_y > 0: y1 += sy; err_y -= 2 * az
            err_x += 2 * ax; err_y += 2 * ay; z1 += sz
            
    return True

def find_safe_critical_nodes_3d(grid, margin, max_z_level=float('inf'), sampling_step=1):
    """
    Finds "corner" nodes using a vectorized approach, now with an optional
    subsampling step to reduce node density.
    
    Args:
        grid: The 3D numpy array representing the environment.
        margin: The safety margin to apply around obstacles.
        max_z_level: The maximum height to consider for nodes.
        sampling_step (int): The step for subsampling. 1 means no sampling,
                             2 means check every other voxel, etc.
    """
    if margin < 1: margin = 1
    print(f"--- Generating buffered obstacle grid with a margin of {margin} voxel(s)... ---")
    buffered_grid = binary_dilation(grid.astype(bool), iterations=margin)
    
    print(f"--- Finding corner nodes of the free space (Vectorized)... ---")
    
    core = buffered_grid[1:-1, 1:-1, 1:-1]
    
    # --- Create shifted versions of the grid to find edges ---
    x_neg, x_pos = buffered_grid[:-2, 1:-1, 1:-1], buffered_grid[2:, 1:-1, 1:-1]
    y_neg, y_pos = buffered_grid[1:-1, :-2, 1:-1], buffered_grid[1:-1, 2:, 1:-1]
    z_neg, z_pos = buffered_grid[1:-1, 1:-1, :-2], buffered_grid[1:-1, 1:-1, 2:]
    x_edge = (x_neg != x_pos)
    y_edge = (y_neg != y_pos)
    z_edge = (z_neg != z_pos)

    is_corner = (x_edge & y_edge) | (x_edge & z_edge) | (y_edge & z_edge)
    is_free_space = ~core
    
    # Initial map of all corners in free space
    final_corner_map = is_corner & is_free_space

    # --- NEW: Apply the Subsampling Mask ---
    if sampling_step > 1:
        print(f"--- Subsampling corners with a step of {sampling_step}... ---")
        # Create a boolean mask of the same shape as the core grid
        sampling_mask = np.zeros_like(final_corner_map, dtype=bool)
        # Set every Nth point to True
        sampling_mask[::sampling_step, ::sampling_step, ::sampling_step] = True
        # Apply the mask to the corner map
        final_corner_map &= sampling_mask
    # ----------------------------------------
    
    corner_coords = np.argwhere(final_corner_map) + 1
    
    filtered_coords = [tuple(map(int, c)) for c in corner_coords if c[2] <= max_z_level]
    
    print(f"--- Reduced node count to {len(corner_coords)}, filtered to {len(filtered_coords)} by height. ---")
    
    return filtered_coords

def filter_nodes_by_relevance(nodes, start, end, detour_factor=1.5):
    print(f"--- Pruning nodes with detour factor {detour_factor}... ---")
    direct_dist = euclidean_dist_3d(start, end)
    max_len = direct_dist * detour_factor
    pruned_nodes = [n for n in nodes if euclidean_dist_3d(n, start) + euclidean_dist_3d(n, end) <= max_len]
    print(f"--- Pruned node count from {len(nodes)} down to {len(pruned_nodes)}. ---")
    return pruned_nodes

def find_min_link_path_3d(grid, start, end, safety_margin=5, sampling_step=3):
    max_height = max(start[2], end[2]) + 10
    print(f"1. Finding critical nodes (margin={safety_margin}, max_height={max_height})...")
    critical_nodes = find_safe_critical_nodes_3d(grid, safety_margin, max_height, sampling_step)
    relevant_nodes = filter_nodes_by_relevance(critical_nodes, start, end)
    all_nodes = list(set([start, end] + relevant_nodes))
    print(f"2. Building visibility graph for {len(all_nodes)} nodes...")
    visibility_graph = {node: [n for n in all_nodes if n != node and line_of_sight_3d(grid, node, n)] for node in all_nodes}
    print("3. Running A* search for min-link path...")
    pq = [((1, 0), [start])]
    cost_so_far = {start: (1, 0)}
    while pq:
        (links, dist), path = heapq.heappop(pq)
        current = path[-1]
        if current == end:
            print("Path found!")
            return path
        for neighbor in visibility_graph[current]:
            new_cost = (links + 1, dist + euclidean_dist_3d(current, neighbor))
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, path + [neighbor]))
    print("No path found.")
    return None

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def draw_cuboid(ax, position, size, color='lightgray', alpha=0.7):
    """Draws a 3D cuboid on the given axes."""
    x, y, z = position
    dx, dy, dz = size
    verts = [(x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z),
             (x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)]
    faces = [[verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]],
             [verts[0], verts[1], verts[5], verts[4]], [verts[2], verts[3], verts[7], verts[6]],
             [verts[0], verts[3], verts[7], verts[4]], [verts[1], verts[2], verts[6], verts[5]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))

def visualize_results(buildings, start, end, path=None, grid_dims=(60, 60, 40), 
                      figsize=(14, 12), camera_dist=10, tick_spacing=None):
    """
    Visualizes the 3D scene. 
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    print("--- Visualizing buildings... ---")
    for b in buildings:
        draw_cuboid(ax, b['pos'], b['size'])

    if path:
        print("--- Visualizing path... ---")
        path_x, path_y, path_z = zip(*path)
        ax.plot(path_x, path_y, path_z, 'b-o', lw=3, ms=8, label='Path', zorder=10)

    ax.scatter(start[0], start[1], start[2], c='g', s=250, ec='k', label='Start', zorder=11)
    ax.scatter(end[0], end[1], end[2], c='r', s=250, ec='k', label='End', zorder=11)
    
    ax.dist = camera_dist

    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    ax.set_title('3D Min-Link Path Planning')
    ax.set_xlim(0, grid_dims[0]); ax.set_ylim(0, grid_dims[1]); ax.set_zlim(0, grid_dims[2])

    if tick_spacing:
        ax.set_xticks(np.arange(0, grid_dims[0] + 1, tick_spacing))
        ax.set_yticks(np.arange(0, grid_dims[1] + 1, tick_spacing))
        ax.set_zticks(np.arange(0, grid_dims[2] + 1, tick_spacing))
    # ------------------------------------

    ax.view_init(elev=40, azim=-75)
    ax.legend()
    plt.show()
# ==============================================================================
# SCENARIO SETUP FUNCTIONS
# ==============================================================================

def add_hexagonal_tree(grid, buildings_list, center_x, center_y, radius, height):
    """
    Adds a voxelized approximation of a hexagonal/circular tree to the grid.
    Uses a simple circular approximation for grid placement.
    """
    x_min, x_max = max(0, center_x - radius), min(grid.shape[0], center_x + radius + 1)
    y_min, y_max = max(0, center_y - radius), min(grid.shape[1], center_y + radius + 1)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            # Use distance formula to create a circular base
            if euclidean_dist_3d((x, y, 0), (center_x, center_y, 0)) <= radius:
                grid[x, y, 0:height] = 1
    

    buildings_list.append({
        'pos': (center_x - radius, center_y - radius, 0),
        'size': (2 * radius, 2 * radius, height)
    })


def setup_forest_scenario():
    """Scenario 4: A forest of hexagonal pillar trees."""
    print("\n--- SCENARIO 4: FOREST ---")
    grid_dims = (100, 100, 50)
    grid = np.zeros(grid_dims, dtype=int)
    safety_margin = 4
    np.random.seed(42)
    
    trees = [] # Use 'trees' list for the visualizer
    num_trees = 20
    min_dist_between_trees = 2 # To avoid overlap

    print(f"--- Generating a forest with {num_trees} trees... ---")
    
    tree_positions = []
    while len(tree_positions) < num_trees:
        # Randomly pick a potential center for a new tree
        center_x = np.random.randint(10, grid_dims[0] - 10)
        center_y = np.random.randint(10, grid_dims[1] - 10)
        

        is_too_close = False
        for pos in tree_positions:
            if euclidean_dist_3d((center_x, center_y, 0), (pos[0], pos[1], 0)) < min_dist_between_trees:
                is_too_close = True
                break
        
        if not is_too_close:
            tree_positions.append((center_x, center_y))
            radius = np.random.randint(2, 3)
            height = np.random.randint(44, 45)
            add_hexagonal_tree(grid, trees, center_x, center_y, radius, height)

    start_point = (5, 5, 5)
    end_point = (95, 25, 40)
    return grid, trees, start_point, end_point, safety_margin

def setup_small_city_scenario():
    print("\n--- SCENARIO 1: ---")
    grid_dims, safety_margin = (60, 60, 40), 3
    grid, buildings = np.zeros(grid_dims, dtype=int), [
        {'pos': (5, 5, 0),   'size': (10, 10, 35)}, {'pos': (5, 22, 0),  'size': (10, 10, 30)},
        {'pos': (22, 5, 0),  'size': (10, 10, 28)}, {'pos': (22, 22, 0), 'size': (10, 13, 18)},
        {'pos': (20, 42, 0), 'size': (12, 6, 22)},  {'pos': (42, 5, 0),  'size': (6, 10, 12)},
        {'pos': (42, 20, 0), 'size': (6, 10, 10)},  {'pos': (42, 35, 0), 'size': (6, 10, 14)},
        {'pos': (42, 50, 0), 'size': (6, 8, 11)},   {'pos': (5, 45, 0),  'size': (10, 10, 8)},
        {'pos': (20, 50, 0), 'size': (15, 8, 10)},  {'pos': (37, 5, 0),  'size': (3, 45, 5)},
        {'pos': (55, 10, 0), 'size': (3, 40, 15)} ]
    for b in buildings:
        x, y, z = b['pos']; dx, dy, dz = b['size']
        grid[x:x+dx, y:y+dy, z:z+dz] = 1
    return grid, buildings, (18, 18, 0), (50, 48, 0), safety_margin

def setup_large_city_scenario():
    """Scenario 2: A large generated city with wide streets."""
    print("\n--- SCENARIO 2:---")
    grid_dims = (120, 120, 70)
    grid = np.zeros(grid_dims, dtype=int)
    safety_margin = 3
    np.random.seed(42)
    
    # Initialize the list for the visualizer
    buildings = []

    print("--- Generating a large city model... ---")
    
    # --- Generate Downtown ---
    core_buildings = [(30, 50, 30, 50), (30, 50, 70, 90), (70, 90, 30, 50), (70, 90, 70, 90)]
    for x_start, x_end, y_start, y_end in core_buildings:
        height = np.random.randint(50, 70)
        grid[x_start:x_end, y_start:y_end, 0:height] = 1
        buildings.append({'pos': (x_start, y_start, 0), 'size': (x_end - x_start, y_end - y_start, height)})

    # --- Generate Outer Blocks ---
    block_definitions = [(10, 30, 10, 110), (90, 110, 10, 110)]
    building_size, street_width_sub = 7, 8
    step = building_size + street_width_sub

    for x_block_start, x_block_end, y_block_start, y_block_end in block_definitions:
        for x in range(x_block_start, x_block_end, step):
            for y in range(y_block_start, y_block_end, step):
                if np.random.rand() > 0.2:
                    height = np.random.randint(15, 35)
                    grid[x:x+building_size, y:y+building_size, 0:height] = 1
                    buildings.append({'pos': (x, y, 0), 'size': (building_size, building_size, height)})

    start_point = (20, 60, 0)
    end_point = (80, 100, 40)

    # Make sure to return all 5 values
    return grid, buildings, start_point, end_point, safety_margin

def setup_megalopolis_scenario():
    """Scenario 3: A huge city with wide avenues."""
    print("\n--- SCENARIO 3 ---")
    grid_dims = (200, 200, 80)
    grid = np.zeros(grid_dims, dtype=int)
    safety_margin = 10
    np.random.seed(41)

    # Initialize the list for the visualizer
    buildings = []

    print("--- Generating city model... ---")

    # --- Generate Downtown ---
    core_buildings = [(40, 70, 40, 70), (40, 70, 130, 160), (130, 160, 40, 70), (130, 160, 130, 160)]
    for x_start, x_end, y_start, y_end in core_buildings:
        height = np.random.randint(60, 80)
        grid[x_start:x_end, y_start:y_end, 0:height] = 1
        buildings.append({'pos': (x_start, y_start, 0), 'size': (x_end - x_start, y_end - y_start, height)})
    
    # --- Generate Outer Districts ---
    block_definitions = [(10, 40, 10, 190), (160, 190, 10, 190)]
    building_size, street_width_sub = 8, 22
    step = building_size + street_width_sub

    for x_block_start, x_block_end, y_block_start, y_block_end in block_definitions:
        for x in range(x_block_start, x_block_end, step):
            for y in range(y_block_start, y_block_end, step):
                if np.random.rand() > 0.45:
                    height = np.random.randint(20, 40)
                    grid[x:x+building_size, y:y+building_size, 0:height] = 1
                    buildings.append({'pos': (x, y, 0), 'size': (building_size, building_size, height)})

    start_point = (25, 55, 0)
    end_point = (150, 175, 20)
    
    # Make sure to return all 5 values
    return grid, buildings, start_point, end_point, safety_margin

if __name__ == '__main__':
    
    # --- CHOOSE YOUR SCENARIO HERE ---
    scenario_to_run = 4 
    # ----------------------------------

    setup_functions = {
        1: setup_small_city_scenario,
        2: setup_large_city_scenario,
        3: setup_megalopolis_scenario,
        4: setup_forest_scenario
    }
    if scenario_to_run not in setup_functions:
        print(f"ERROR: Scenario {scenario_to_run} is not valid.")
        exit()
    

    grid, buildings, start_point, end_point, safety_margin = setup_functions[scenario_to_run]()
    

    buffered_grid = binary_dilation(grid.astype(bool), iterations=safety_margin)
    if buffered_grid[start_point] or buffered_grid[end_point]:
        print("❌ ERROR: Start or end point is within the safety margin of an obstacle.")

        visualize_results(buildings, start_point, end_point, None, grid.shape)
        exit()
    print("✅ Start and end points are in safe locations.")

    # --- Run Pathfinding ---
    start_time = time.time()

    path = find_min_link_path_3d(grid, start_point, end_point, safety_margin=safety_margin)
    end_time = time.time()

    # --- Print & Visualize Results ---
    if path:
        print(f"\n--- RESULTS ---\nPath found in {end_time - start_time:.4f} seconds.")
        print(f"Number of links: {len(path) - 1}\nPath Coordinates: {path}")
    else:
        print("\n--- RESULTS ---\nNo path was found.")
    
    print("\nLaunching visualization...")

    if scenario_to_run == 3:
        visualize_results(buildings, start_point, end_point, path, grid.shape, 
                          figsize=(20, 18), camera_dist=8, tick_spacing=5)
    elif scenario_to_run == 2:
        visualize_results(buildings, start_point, end_point, path, grid.shape, 
                          figsize=(16, 14), camera_dist=9, tick_spacing=10)
    else:
        visualize_results(buildings, start_point, end_point, path, grid.shape, 
                          tick_spacing=5)