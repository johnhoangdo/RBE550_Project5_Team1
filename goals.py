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
Date: 11/10/2025
"""


# GOAL 1: Two Towers (RED-GREEN-BLUE + YELLOW-MAGENTA-CYAN)
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


def get_goal(goal_name):
    goals = {
        "two_towers": GOAL_TWO_TOWERS,   
    }
    
    goal = goals.get(goal_name.lower())
    if goal is None:
        print(f"[WARN] Goal '{goal_name}' not found. Available goals:")
        for name in goals.keys():
            print(f"  - {name}")
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
