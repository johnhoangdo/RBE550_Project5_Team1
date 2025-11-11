"""
goals.py
----------------------------------
Goal predicate configurations for Project 5: Building the Two Towers (TAMP).

Available blocks: r (red), g (green), b (blue), y (yellow), m (magenta), c (cyan)

Each goal is represented as a Python dictionary matching the predicate 
structure used in abstraction.py:
{
  "on": [("top_block", "bottom_block"), ...],
  "ontable": ["block_name", ...],
  "clear": ["block_name", ...]
}

Note: The "handempty" predicate is typically omitted from goals since
      the final state should always have the gripper empty.

Author: LA, JHD, JEN
Date: 11/10/2025
"""


# ============================================================
#  GOAL 1: Two Towers (RED-GREEN-BLUE + YELLOW-MAGENTA-CYAN)
# ============================================================
GOAL_TWO_TOWERS = {
    "on": [
        # First tower: RED on GREEN on BLUE
        ("r", "g"),   # red on green
        ("g", "b"),   # green on blue
        
        # Second tower: YELLOW on MAGENTA on CYAN
        ("y", "m"),   # yellow on magenta
        ("m", "c")    # magenta on cyan
    ],
    "ontable": ["b", "c"],  # blue and cyan are on table (tower bases)
    "clear": ["r", "y"],    # red and yellow are on top (clear)
    # handempty is implied in final state
}


# ============================================================
#  GOAL 2: Five-Block Tower (MAGENTA-YELLOW-BLUE-RED-GREEN)
# ============================================================
GOAL_FIVE_TOWER = {
    "on": [
        ("m", "y"),   # magenta on yellow (top)
        ("y", "b"),   # yellow on blue
        ("b", "r"),   # blue on red
        ("r", "g")    # red on green (bottom)
    ],
    "ontable": ["g"],   # green is on table (tower base)
    "clear": ["m"],     # magenta is on top (clear)
    # Note: cyan (c) is not used - should be on table and clear
}

# Alternative: Include cyan explicitly in goal
GOAL_FIVE_TOWER_EXPLICIT = {
    "on": [
        ("m", "y"),   # magenta on yellow
        ("y", "b"),   # yellow on blue
        ("b", "r"),   # blue on red
        ("r", "g")    # red on green
    ],
    "ontable": ["g", "c"],  # green (base) and cyan (unused) on table
    "clear": ["m", "c"],    # magenta (top) and cyan (unused) are clear
}


# ============================================================
#  GOAL 3: Tallest Tower (6 blocks) - BONUS CHALLENGE
# ============================================================
# Start from scene with all blocks on floor (create_scene_6blocks)
# Build: RED-GREEN-BLUE-YELLOW-MAGENTA-CYAN (bottom to top)
GOAL_TALLEST_TOWER = {
    "on": [
        ("c", "m"),   # cyan on magenta (top)
        ("m", "y"),   # magenta on yellow
        ("y", "b"),   # yellow on blue
        ("b", "g"),   # blue on green
        ("g", "r")    # green on red (bottom)
    ],
    "ontable": ["r"],   # red is on table (tower base)
    "clear": ["c"]      # cyan is on top (clear)
}

# Alternative ordering for stability (heavier/wider base)
GOAL_TALLEST_TOWER_STABLE = {
    "on": [
        ("r", "g"),   # red on green (top)
        ("g", "b"),   # green on blue
        ("b", "y"),   # blue on yellow
        ("y", "m"),   # yellow on magenta
        ("m", "c")    # magenta on cyan (bottom)
    ],
    "ontable": ["c"],   # cyan is on table (tower base)
    "clear": ["r"]      # red is on top (clear)
}

"""
# ============================================================
#  GOAL 4: Special Structures (Creative Challenge)
# ============================================================

# Example 1: Simple Arch/Bridge (requires non-standard predicates)
# This is NOT possible with standard blocksworld - kept for reference
# In standard blocksworld, each block can only be on ONE other block
# GOAL_BRIDGE = {
#     "on": [("g", "r"), ("g", "b")],  # INVALID: g cannot be on both r and b
#     "ontable": ["r", "b"],
#     "clear": ["g"]
# }

# Example 2: Two Small Towers (2+2 configuration)
GOAL_TWO_SMALL_TOWERS = {
    "on": [
        # First tower: RED on GREEN
        ("r", "g"),
        
        # Second tower: BLUE on YELLOW
        ("b", "y")
    ],
    "ontable": ["g", "y", "m", "c"],  # bases and unused blocks on table
    "clear": ["r", "b", "m", "c"]     # tops and unused blocks are clear
}

# Example 3: Pyramid Base (three separate 2-block towers)
GOAL_PYRAMID_BASE = {
    "on": [
        ("r", "g"),   # tower 1
        ("b", "y"),   # tower 2
        ("m", "c")    # tower 3
    ],
    "ontable": ["g", "y", "c"],  # three tower bases
    "clear": ["r", "b", "m"]      # three tower tops
}

# Example 4: Staircase (ascending towers: 1, 2, 3 blocks)
GOAL_STAIRCASE = {
    "on": [
        # Tower 1: single block (red)
        # Tower 2: yellow on green
        ("y", "g"),
        # Tower 3: cyan on magenta on blue
        ("c", "m"),
        ("m", "b")
    ],
    "ontable": ["r", "g", "b"],  # three tower bases
    "clear": ["r", "y", "c"]     # three tower tops
}

# Example 5: Line Formation (all blocks on table)
GOAL_LINE_FORMATION = {
    "on": [],  # no stacking - all blocks on table
    "ontable": ["r", "g", "b", "y", "m", "c"],
    "clear": ["r", "g", "b", "y", "m", "c"]
}
"""

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def get_goal(goal_name):
    """
    Retrieve a goal configuration by name.
    
    Args:
        goal_name (str): Name of the goal configuration
        
    Returns:
        dict: Goal predicate dictionary, or None if not found
        
    Example:
        goal = get_goal("two_towers")
        goal = get_goal("five_tower")
    """
    goals = {
        # Main tasks
        "two_towers": GOAL_TWO_TOWERS,
        "five_tower": GOAL_FIVE_TOWER,
        "five_tower_explicit": GOAL_FIVE_TOWER_EXPLICIT,
        "tallest_tower": GOAL_TALLEST_TOWER,
        "tallest_tower_stable": GOAL_TALLEST_TOWER_STABLE,
        
        
        # Creative structures
        #"two_small_towers": GOAL_TWO_SMALL_TOWERS,
        #"pyramid_base": GOAL_PYRAMID_BASE,
        #"staircase": GOAL_STAIRCASE,
        #"line_formation": GOAL_LINE_FORMATION,
    }
    
    goal = goals.get(goal_name.lower())
    if goal is None:
        print(f"[WARN] Goal '{goal_name}' not found. Available goals:")
        for name in goals.keys():
            print(f"  - {name}")
    return goal


def validate_goal(goal_dict, available_blocks=None):
    """
    Check if a goal configuration is valid (no logical contradictions).
    
    Args:
        goal_dict (dict): Goal predicate dictionary
        available_blocks (set): Set of available block names (default: r,g,b,y,m,c)
        
    Returns:
        tuple: (is_valid, error_messages)
        
    Validation checks:
        1. Each block appears at most once as top in ON relations
        2. Each block appears at most once as bottom in ON relations
        3. ONTABLE blocks don't appear as top in ON relations
        4. CLEAR blocks don't appear as bottom in ON relations
        5. All blocks exist in available_blocks set
    """
    if available_blocks is None:
        available_blocks = {"r", "g", "b", "y", "m", "c"}
    
    errors = []
    
    # Extract all blocks mentioned
    all_blocks = set()
    top_blocks = {}  # block -> what it's on top of
    bottom_blocks = {}  # block -> what's on top of it
    
    # Process ON relations
    for (top, bottom) in goal_dict.get("on", []):
        all_blocks.add(top)
        all_blocks.add(bottom)
        
        # Check: each block appears at most once as top
        if top in top_blocks:
            errors.append(f"Block '{top}' appears multiple times as top in ON relations")
        else:
            top_blocks[top] = bottom
        
        # Check: each block appears at most once as bottom
        if bottom in bottom_blocks:
            errors.append(f"Block '{bottom}' appears multiple times as bottom in ON relations")
        else:
            bottom_blocks[bottom] = top
    
    # Process ONTABLE
    ontable_blocks = set(goal_dict.get("ontable", []))
    all_blocks.update(ontable_blocks)
    
    # Check: ONTABLE blocks shouldn't be on top of anything
    for block in ontable_blocks:
        if block in top_blocks:
            errors.append(f"Block '{block}' is both ONTABLE and ON another block")
    
    # Process CLEAR
    clear_blocks = set(goal_dict.get("clear", []))
    all_blocks.update(clear_blocks)
    
    # Check: CLEAR blocks shouldn't have anything on top
    for block in clear_blocks:
        if block in bottom_blocks:
            errors.append(f"Block '{block}' is both CLEAR and has a block on top")
    
    # Check: all blocks exist
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
    
    # ON predicates
    if goal_dict.get("on"):
        print("\nStacking relations:")
        for (top, bottom) in goal_dict["on"]:
            print(f"  {top.upper()} on {bottom.upper()}")
    
    # ONTABLE predicates
    if goal_dict.get("ontable"):
        print("\nOn table:")
        print(f"  {', '.join(b.upper() for b in goal_dict['ontable'])}")
    
    # CLEAR predicates
    if goal_dict.get("clear"):
        print("\nClear (top blocks):")
        print(f"  {', '.join(b.upper() for b in goal_dict['clear'])}")
    
    print('=' * 50 + '\n')


# ============================================================
#  USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    # Example 1: Get a goal by name
    goal = get_goal("two_towers")
    visualize_goal(goal, "Goal: Two Towers")
    
    # Example 2: Validate a goal
    is_valid, errors = validate_goal(GOAL_TWO_TOWERS)
    if is_valid:
        print("✓ Goal is valid")
    else:
        print("✗ Goal has errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Example 3: Check tallest tower
    visualize_goal(GOAL_TALLEST_TOWER, "Goal: Tallest Tower (6 blocks)")
    is_valid, errors = validate_goal(GOAL_TALLEST_TOWER)
    print(f"Valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")