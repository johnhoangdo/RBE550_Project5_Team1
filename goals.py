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
    