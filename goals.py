"""
Goal configurations for Project 5: Building the Two Towers (TAMP)

We've got 6 blocks to work with: r (red), g (green), b (blue), 
y (yellow), m (magenta), c (cyan)

Each goal is just a Python dict that matches how we set up predicates 
in abstraction.py:
{
  "on": [("top_block", "bottom_block"), ...],     # what's stacked on what
  "ontable": ["block_name", ...],                  # what's sitting on the table
  "clear": ["block_name", ...]                     # what has nothing on top
}

Quick note: We usually don't bother including "handempty" in the goal 
since the gripper should obviously be empty when we're done.

Authors: LA, JHD, JEN
Date: 12/01/2025
"""

# ============================================================
# GOAL 1: Two Towers (RED-GREEN-BLUE + YELLOW-MAGENTA-CYAN)
# ============================================================
GOAL_TWO_TOWERS = {
    "on": [
        ("r", "g"),
        ("g", "b"),
        ("y", "m"),
        ("m", "c")
    ],
    "ontable": ["b", "c"],
    "clear": ["r", "y"]
}


# ============================================================
# GOAL 2: Five-Block Tower (MAGENTA-YELLOW-BLUE-RED-GREEN)
# ============================================================
GOAL_FIVE_TOWER = {
    "on": [
        ("m", "y"),   # magenta on yellow (top to bottom)
        ("y", "b"),   # yellow on blue
        ("b", "r"),   # blue on red
        ("r", "g")    # red on green
    ],
    "ontable": ["g"],   # green is the base on table
    "clear": ["m"],     # magenta is on top (clear)
}


# ============================================================
# GOAL 3: Tallest Tower - Six Blocks
# ============================================================
GOAL_SIX_TOWER = {
    "on": [
        ("c", "m"),   # cyan on magenta (top)
        ("m", "y"),   # magenta on yellow
        ("y", "b"),   # yellow on blue
        ("b", "r"),   # blue on red
        ("r", "g")    # red on green (bottom)
    ],
    "ontable": ["g"],   # green is base
    "clear": ["c"],     # cyan is top
}



# ============================================================
# GOAL 4A-SIMPLE: Two Tower Grid (4 Yellow Blocks) - TEST
# ============================================================
# Simplified version for testing - just 2 two-block towers
GOAL_TOWER_GRID_SIMPLE = {
    "on": [
        ("y1", "y3"),   # Tower 1
        ("y2", "y4"),   # Tower 2
    ],
    "ontable": ["y3", "y4"],
    "clear": ["y1", "y2"],
    # Spatial constraints added dynamically
}


# ============================================================
# GOAL 4A: Tower Grid Configuration (12 Yellow Blocks)
# ============================================================
# Import will happen at runtime to avoid circular dependency
def _get_grid_positions():
    """Lazy import to get grid positions"""
    try:
        from abstraction import calculate_grid_positions
        return calculate_grid_positions(
            center=(0.35, -0.12),  # Moved forward for better reachability (was 0.30)
            spacing=0.045,  # TIGHT spacing for Goal 4A (blocks almost touching!)
            grid_shape=(3, 4)
        )
    except ImportError:
        # Fallback if abstraction not available - also shifted +0.05 in X
        return [(0.3275, -0.0675), (0.3275, -0.0225), (0.3275, 0.0225), (0.3275, 0.0675),
                (0.3725, -0.0675), (0.3725, -0.0225), (0.3725, 0.0225), (0.3725, 0.0675),
                (0.4175, -0.0675), (0.4175, -0.0225), (0.4175, 0.0225), (0.4175, 0.0675)]

# Pattern: X T T X / T X X T / X T T X
# Where X = empty, T = 2-block tower
# Tower positions: indices 1, 2, 4, 7, 9, 10 (out of 0-11)
#
# Goal 4A: Tower Grid Configuration
# Create 6 two-block towers arranged in specific 3×4 grid pattern.
# Uses 12 yellow blocks (y1-y12).
# 
# Pattern visualization:
# Row 1:  Empty  Tower  Tower  Empty
# Row 2:  Tower  Empty  Empty  Tower
# Row 3:  Empty  Tower  Tower  Empty
GOAL_TOWER_GRID = {
    "on": [
        # 6 towers, each 2 blocks high
        ("y1", "y7"),   # Tower at grid position 1
        ("y2", "y8"),   # Tower at grid position 2
        ("y3", "y9"),   # Tower at grid position 4
        ("y4", "y10"),  # Tower at grid position 7
        ("y5", "y11"),  # Tower at grid position 9
        ("y6", "y12"),  # Tower at grid position 10
    ],
    "ontable": ["y7", "y8", "y9", "y10", "y11", "y12"],  # Base blocks
    "clear": ["y1", "y2", "y3", "y4", "y5", "y6"],      # Top blocks
    
    # Spatial constraints (loaded dynamically)
    # Will be populated when goal is retrieved
}

# ============================================================
# GOAL 4B: Adjacent Configuration (3 Red + 3 Green)
# ============================================================
# Goal 4B: Adjacent Configuration
# 
# Create towers in 2×2 grid with mixed heights and colors:
# - Two red towers (one height-2, one height-1)
# - Two green towers (one height-2, one height-1)
# 
# Grid layout (rows × cols):
# [R(2)]  [G(2)]
# [R(1)]  [G(1)]
# 
# Where R/G = color, (N) = height
def _get_2x2_positions():
    """Lazy import to get 2×2 grid positions"""
    try:
        from abstraction import calculate_2x2_grid_positions
        return calculate_2x2_grid_positions(
            center=(0.40, 0.0),  # Move back to clear spawn area
            spacing=0.045  # Reduced from 0.15 to 0.10 (closer together)
        )
    except ImportError:
        # Fallback - also moved back
        return [(0.275, -0.05), (0.275, 0.05), 
                (0.375, -0.05), (0.375, 0.05)]

GOAL_ADJACENT_COLORS = {
    "on": [
        ("r1", "r2"),   # Red tower height 2 (position 0)
        ("g1", "g2"),   # Green tower height 2 (position 1)
        # r3 sits alone at position 2 (height 1)
        # g3 sits alone at position 3 (height 1)
    ],
    "ontable": ["r2", "g2", "r3", "g3"],  # Four base blocks
    "clear": ["r1", "g1", "r3", "g3"],    # Four top blocks (2 are also bases)
    
    # Spatial constraints (loaded dynamically)
}


def get_goal(goal_name):
    """
    Retrieve goal configuration by name
    
    Available goals:
    - "two_towers": Goal 1 - Two 3-block towers (RGB + YMC)
    - "five_tower": Goal 2 - Single 5-block tower
    - "six_tower": Goal 3 - Single 6-block tower (tallest)
    - "tower_grid": Goal 4A - 12 yellow blocks in 3×4 grid pattern
    - "adjacent": Goal 4B - 3 red + 3 green in 2×2 grid with mixed heights
    
    Args:
        goal_name: Name of goal (case-insensitive)
    
    Returns:
        dict: Goal configuration with predicates (and spatial constraints for Goal 4)
    """
    goals = {
        "two_towers": GOAL_TWO_TOWERS,
        "five_tower": GOAL_FIVE_TOWER,
        "six_tower": GOAL_SIX_TOWER,
        "tower_grid": GOAL_TOWER_GRID,
        "tower_grid_simple": GOAL_TOWER_GRID_SIMPLE,  # NEW: Simplified test
        "adjacent": GOAL_ADJACENT_COLORS,
    }
    
    goal = goals.get(goal_name.lower())
    
    if goal is None:
        print(f"[WARN] Goal '{goal_name}' not found. Available goals:")
        for name in goals.keys():
            print(f"  - {name}")
        return None
    
    # Add spatial constraints for Goal 4 if not already present
    if goal_name.lower() == "tower_grid" and "spatial" not in goal:
        positions = _get_grid_positions()
        goal["spatial"] = {
            "y7": positions[1],
            "y8": positions[2],
            "y9": positions[4],
            "y10": positions[7],
            "y11": positions[9],
            "y12": positions[10],
        }
        print("[Goals] Added spatial constraints to tower_grid")
    
    elif goal_name.lower() == "tower_grid_simple" and "spatial" not in goal:
        positions = _get_grid_positions()
        goal["spatial"] = {
            "y3": positions[0],   # First tower base
            "y4": positions[2],   # Second tower base
        }
        print("[Goals] Added spatial constraints to tower_grid_simple")
    
    elif goal_name.lower() == "adjacent" and "spatial" not in goal:
        positions = _get_2x2_positions()
        goal["spatial"] = {
            "r2": positions[0],  # Top-left
            "g2": positions[1],  # Top-right
            "r3": positions[2],  # Bottom-left
            "g3": positions[3],  # Bottom-right
        }
        print("[Goals] Added spatial constraints to adjacent")
    
    return goal


def validate_goal(goal_dict, available_blocks=None):
    """
    Check if a goal configuration actually makes sense (no weird contradictions).
    
    Args:
        goal_dict (dict): The goal predicates we want to check
        available_blocks (set): Which blocks we have to work with (defaults to all 6)
    
    Returns:
        tuple: (is_valid, error_messages) - True/False plus any issues we found
    
    What we're checking for:
        1. No block is stacked on top of two different blocks at once
        2. No block has two different blocks stacked on top of it
        3. Blocks on the table can't also be stacked on another block (duh)
        4. "Clear" blocks can't have stuff on top of them (that's the whole point)
        5. All blocks mentioned actually exist in our available set
    """
    if available_blocks is None:
        available_blocks = {"r", "g", "b", "y", "m", "c"}
    
    errors = []
    all_blocks = set()
    top_blocks = {}
    bottom_blocks = {}
    
    for (top, bottom) in goal_dict.get("on", []):
        all_blocks.add(top)
        all_blocks.add(bottom)
        
        if top in top_blocks:
            errors.append(f"Block '{top}' appears multiple times as top in ON relations")
        else:
            top_blocks[top] = bottom
        
        if bottom in bottom_blocks:
            errors.append(f"Block '{bottom}' appears multiple times as bottom in ON relations")
        else:
            bottom_blocks[bottom] = top
    
    ontable_blocks = set(goal_dict.get("ontable", []))
    all_blocks.update(ontable_blocks)
    
    for block in ontable_blocks:
        if block in top_blocks:
            errors.append(f"Block '{block}' is both ONTABLE and ON another block")
    
    clear_blocks = set(goal_dict.get("clear", []))
    all_blocks.update(clear_blocks)
    
    for block in clear_blocks:
        if block in bottom_blocks:
            errors.append(f"Block '{block}' is both CLEAR and has a block on top")
    
    for block in all_blocks:
        if block not in available_blocks:
            errors.append(f"Block '{block}' not in available blocks: {available_blocks}")
    
    return (len(errors) == 0, errors)


def visualize_goal(goal_dict, title="Goal Configuration"):
    """
    Print a goal configuration in human-readable format.
    
    Args:
        goal_dict (dict): Goal predicate dictionary
        title (str): Title for the visualization
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print('=' * 50)
    
    if goal_dict.get("on"):
        print("\nStacking relations:")
        for (top, bottom) in goal_dict["on"]:
            print(f"  {top.upper()} on {bottom.upper()}")
    
    if goal_dict.get("ontable"):
        print("\nOn table:")
        print(f"  {', '.join(b.upper() for b in goal_dict['ontable'])}")
    
    if goal_dict.get("clear"):
        print("\nClear (top blocks):")
        print(f"  {', '.join(b.upper() for b in goal_dict['clear'])}")
    
    print('=' * 50 + '\n')

 # USAGE EXAMPLES
if __name__ == "__main__":
    goal = get_goal("two_towers")
    visualize_goal(goal, "Goal: Two Towers")
    
    is_valid, errors = validate_goal(GOAL_TWO_TOWERS)
    if is_valid:
        print("CORRECT, Goal is valid")
    else:
        print("WRONG, Goal has errors:")
        for error in errors:
            print(f"  - {error}")
