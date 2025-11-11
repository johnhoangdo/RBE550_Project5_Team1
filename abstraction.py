"""
abstraction.py
----------------------------------
Symbolic abstraction (a.k.a. "lifting") layer for Project 5.

Purpose:
  • Convert the continuous Genesis simulator state into discrete symbolic predicates
    that can be used by the task planner (Pyperplan / PDDL).
  • Generate a PDDL problem file from those predicates.
  • Provide helper utilities to check goal completion and tower stability.

Key Concepts:
  - LIFTING: Converting continuous state (positions, orientations) to discrete 
    symbolic predicates (ON, CLEAR, ONTABLE, etc.)
  - Predicates are facts about the world that the task planner can reason about
  - Thresholds are tuned for 4cm x 4cm x 4cm blocks in Genesis simulator

Author: LA, JHD, JEN
Date: 11/10/2025
"""

import numpy as np
import os


# ============================================================
# CONSTANTS: Tuned for 4cm blocks in Genesis simulator
# ============================================================

# Block dimensions (Genesis uses meters)
BLOCK_SIZE = 0.04  # 4cm cubic blocks

# Horizontal alignment tolerance for determining if blocks are stacked
# Two blocks are considered aligned if their x,y centers are within this distance
XY_ALIGNMENT_THRESHOLD = 0.02  # 2cm tolerance (50% of block width)

# Vertical separation bounds for ON predicate
# A block is "on" another if z-distance is within this range
Z_MIN_SEPARATION = 0.035  # Minimum 3.5cm (slightly less than block size due to physics settling)
Z_MAX_SEPARATION = 0.045  # Maximum 4.5cm (slightly more than block size for tolerance)

# Table height threshold - blocks below this are considered on the table
TABLE_HEIGHT_THRESHOLD = 0.03  # 3cm - ground plane is at z=0, block centers at z=0.02

# Stability checking thresholds
STABILITY_XY_THRESHOLD = 0.025  # 2.5cm - blocks can drift slightly but still be stable
STABILITY_Z_TOLERANCE = 0.005   # 0.5cm - vertical position variation tolerance


# ============================================================
# 1️⃣  UTILITY FUNCTIONS: Scene State Extraction
# ============================================================

def get_block_positions(blocks_state):
    """
    Extract current 3D positions of all blocks from the Genesis scene.
    
    Args:
        blocks_state (dict): Dictionary mapping block names (str) to Genesis 
                            block entities. Example: {"r": cubeR, "g": cubeG, ...}
                            This dict is returned by create_scene_*() functions.
    
    Returns:
        dict: Mapping of block_name (str) -> position (np.array of shape (3,))
              Example: {"r": array([0.65, 0.0, 0.02]), "g": array([0.65, 0.2, 0.02])}
    
    Notes:
        - Uses Genesis entity's get_pos() method which returns [x, y, z] in meters
        - Positions are in world frame with origin at scene center, z-up
        - Returns empty dict if any errors occur during position extraction
    """
    positions = {}
    for name, block in blocks_state.items():
        try:
            # Genesis entities return position as tensor or array
            pos = block.get_pos()
            positions[name] = np.array(pos, dtype=float)
        except Exception as e:
            print(f"[WARN] Failed to get position for block '{name}': {e}")
            # Continue processing other blocks rather than failing completely
    return positions


def get_gripper_state(robot):
    """
    Determine what object (if any) the robot gripper is holding.
    
    Args:
        robot: Robot entity (Genesis entity or RobotAdapter wrapper)
    
    Returns:
        tuple: (is_holding, object_name)
               - is_holding (bool): True if gripper is holding an object
               - object_name (str or None): Name of held object, or None if empty
    
    Notes:
        - Checks robot.attached_object attribute set by motion primitives
        - For RobotAdapter, this gets forwarded to the underlying robot entity
        - Returns (False, None) if robot has no attached_object attribute
    """
    held_obj = getattr(robot, "attached_object", None)
    if held_obj is not None:
        # Extract name from held object - try multiple possible attributes
        object_name = getattr(held_obj, 'name', None)
        if object_name is None:
            # Fallback: try to find name by searching blocks_state
            # This handles cases where object doesn't have .name attribute
            object_name = str(held_obj)  # Use string representation as fallback
        return True, object_name
    return False, None


# ============================================================
# 2️⃣  PREDICATE COMPUTATION: Lifting to Symbolic State
# ============================================================

def compute_predicates(robot, blocks_state, scene=None,
                       xy_threshold=XY_ALIGNMENT_THRESHOLD,
                       z_min=Z_MIN_SEPARATION,
                       z_max=Z_MAX_SEPARATION):
    """
    Convert continuous simulator state to discrete symbolic predicates for PDDL.
    
    This is the core "lifting" function that bridges geometric and symbolic reasoning.
    It analyzes block positions and relationships to determine which logical facts
    are true in the current scene.
    
    Args:
        robot: Robot entity or RobotAdapter instance
        blocks_state (dict): Map of block_name -> Genesis block entity
        scene: Genesis scene object (optional, reserved for future use)
        xy_threshold (float): Horizontal alignment tolerance in meters
        z_min (float): Minimum vertical separation for ON predicate
        z_max (float): Maximum vertical separation for ON predicate
    
    Returns:
        dict: Symbolic predicates organized by type:
            {
              "on": [(block_a, block_b), ...],      # block_a is on top of block_b
              "ontable": [block_name, ...],         # blocks resting on table
              "clear": [block_name, ...],           # blocks with nothing on top
              "holding": [block_name],              # block in gripper (max 1)
              "handempty": [True] or []             # whether gripper is empty
            }
    
    Predicate Semantics:
        - ON(A, B): Block A is directly on top of block B
          * Requires horizontal alignment within xy_threshold
          * Requires vertical separation in [z_min, z_max] range
          * Used for tower building constraints
        
        - ONTABLE(A): Block A is on the table (no block below it)
          * True if block is not ON any other block
          * Mutually exclusive with ON predicate
        
        - CLEAR(A): No block is on top of A
          * Block can be picked up or stacked upon
          * Computed as blocks NOT appearing as bottom in any ON relation
        
        - HOLDING(A): Gripper is holding block A
          * At most one block can be held at a time
          * Mutually exclusive with HANDEMPTY
        
        - HANDEMPTY(): Gripper is not holding anything
          * Required precondition for pick-up actions
          * Mutually exclusive with HOLDING
    
    Algorithm:
        1. Extract current positions of all blocks from Genesis
        2. Check gripper state to determine HOLDING vs HANDEMPTY
        3. Compare all block pairs to identify ON relationships:
           - Check horizontal alignment (x,y within threshold)
           - Check vertical stacking (z-difference in valid range)
        4. Blocks not ON anything else are ONTABLE
        5. Blocks not serving as base for any ON relation are CLEAR
    
    Example:
        If blocks are arranged as: RED on GREEN, GREEN on table, BLUE on table
        Returns: {
            "on": [("r", "g")],
            "ontable": ["g", "b"],
            "clear": ["r", "b"],
            "holding": [],
            "handempty": [True]
        }
    """
    # Initialize empty predicate lists
    predicates = {
        "on": [],           # Stacking relationships
        "ontable": [],      # Blocks on table surface
        "clear": [],        # Blocks with no block on top
        "holding": [],      # Block currently grasped
        "handempty": []     # Gripper state
    }

    # -------------------------------
    # Step 1: Extract block positions
    # -------------------------------
    block_positions = get_block_positions(blocks_state)
    
    if not block_positions:
        print("[WARN] No block positions found - scene may not be initialized")
        return predicates

    # -------------------------------
    # Step 2: Determine gripper state
    # -------------------------------
    is_holding, held_obj_name = get_gripper_state(robot)
    
    if is_holding:
        # Find the block name in blocks_state that matches held object
        for name, block in blocks_state.items():
            if block == robot.attached_object:
                predicates["holding"].append(name)
                break
    else:
        predicates["handempty"].append(True)

    # -------------------------------
    # Step 3: Compute ON and ONTABLE predicates
    # -------------------------------
    # For each block, check if it's on top of another block
    for name_a, pos_a in block_positions.items():
        # Skip blocks currently being held (they're not on anything)
        if is_holding and name_a in predicates["holding"]:
            continue
            
        is_on_another_block = False
        
        # Check against all other blocks
        for name_b, pos_b in block_positions.items():
            if name_a == name_b:
                continue  # Can't be on itself
            
            # Skip if block_b is being held (can't stack on held block)
            if is_holding and name_b in predicates["holding"]:
                continue
            
            # Horizontal alignment check: centers aligned in x-y plane
            dx = abs(pos_a[0] - pos_b[0])
            dy = abs(pos_a[1] - pos_b[1])
            horizontally_aligned = (dx < xy_threshold and dy < xy_threshold)
            
            # Vertical stacking check: block_a is above block_b by ~block_size
            dz = pos_a[2] - pos_b[2]  # Positive if a is above b
            vertically_stacked = (z_min < dz < z_max)
            
            # ON predicate: both conditions must be satisfied
            if horizontally_aligned and vertically_stacked:
                predicates["on"].append((name_a, name_b))
                is_on_another_block = True
                break  # A block can only be on one other block
        
        # If not on another block, it must be on the table
        if not is_on_another_block:
            predicates["ontable"].append(name_a)

    # -------------------------------
    # Step 4: Compute CLEAR predicate
    # -------------------------------
    # A block is clear if no other block is on top of it
    # Extract all blocks that serve as a base (appear as second element in ON tuples)
    blocks_with_something_on_top = {bottom for (top, bottom) in predicates["on"]}
    
    # All blocks not serving as a base are clear
    for name in block_positions.keys():
        # Skip held blocks - they're not in the scene to be stacked upon
        if is_holding and name in predicates["holding"]:
            continue
        if name not in blocks_with_something_on_top:
            predicates["clear"].append(name)

    return predicates


# ============================================================
# 3️⃣  PDDL FILE GENERATION: Task Planning Interface
# ============================================================

def generate_pddl_problem(predicates, goal_predicates,
                          filename="problem.pddl",
                          domain_name="blocksworld",
                          problem_name="build-towers"):
    """
    Generate a PDDL problem file for use with task planners (e.g., Pyperplan).
    
    PDDL (Planning Domain Definition Language) is the standard format for
    symbolic planning problems. This function takes our computed predicates
    and goal specification and writes them in PDDL syntax.
    
    Args:
        predicates (dict): Current state predicates (from compute_predicates)
        goal_predicates (dict): Desired goal state predicates
        filename (str): Output file path for the PDDL problem file
        domain_name (str): Name of the PDDL domain (must match domain file)
        problem_name (str): Descriptive name for this problem instance
    
    Returns:
        str: Absolute path to the generated PDDL problem file
    
    File Format:
        The generated file follows PDDL syntax:
        ```
        (define (problem <n>)
          (:domain <domain-name>)
          (:objects <block-list>)
          (:init <initial-state-predicates>)
          (:goal (and <goal-predicates>))
        )
        ```
    
    Example:
        Input predicates: {"on": [("r", "g")], "ontable": ["g"], ...}
        Input goal: {"on": [("r", "g"), ("g", "b")], ...}
        
        Output file:
        ```
        (define (problem build-towers)
          (:domain blocksworld)
          (:objects r g b - block)
          (:init
            (on r g)
            (ontable g)
            (clear r)
            (handempty)
          )
          (:goal (and
            (on r g)
            (on g b)
          ))
        )
        ```
    
    Notes:
        - Objects are automatically extracted from predicates (all mentioned blocks)
        - HANDEMPTY is treated as a nullary predicate (no arguments)
        - Goal predicates can be subset of state (partial specification)
    """
    with open(filename, "w") as f:
        # Problem header
        f.write(f"(define (problem {problem_name})\n")
        f.write(f"  (:domain {domain_name})\n\n")

        # (:objects ...) section - extract all unique block names
        # Combine blocks from all predicate types to ensure completeness
        all_blocks = set()
        all_blocks.update(predicates.get("ontable", []))
        all_blocks.update(predicates.get("clear", []))
        all_blocks.update(predicates.get("holding", []))
        for (a, b) in predicates.get("on", []):
            all_blocks.add(a)
            all_blocks.add(b)
        
        # Write objects declaration (sorted for consistency)
        f.write("  (:objects\n")
        f.write("    " + " ".join(sorted(all_blocks)) + " - block\n")
        f.write("  )\n\n")

        # (:init ...) section - current state
        f.write("  (:init\n")
        
        # ON predicates - binary relations
        for (a, b) in predicates.get("on", []):
            f.write(f"    (on {a} {b})\n")
        
        # ONTABLE predicates - unary
        for a in predicates.get("ontable", []):
            f.write(f"    (ontable {a})\n")
        
        # CLEAR predicates - unary
        for a in predicates.get("clear", []):
            f.write(f"    (clear {a})\n")
        
        # HOLDING predicates - unary
        for a in predicates.get("holding", []):
            f.write(f"    (holding {a})\n")
        
        # HANDEMPTY predicate - nullary (no arguments)
        if predicates.get("handempty"):
            f.write("    (handempty)\n")
        
        f.write("  )\n\n")

        # (:goal ...) section - desired state
        f.write("  (:goal (and\n")
        
        # Goal ON predicates
        for (a, b) in goal_predicates.get("on", []):
            f.write(f"    (on {a} {b})\n")
        
        # Goal ONTABLE predicates
        for a in goal_predicates.get("ontable", []):
            f.write(f"    (ontable {a})\n")
        
        # Goal CLEAR predicates
        for a in goal_predicates.get("clear", []):
            f.write(f"    (clear {a})\n")
        
        # Goal HOLDING predicates (rare, but supported)
        for a in goal_predicates.get("holding", []):
            f.write(f"    (holding {a})\n")
        
        # Goal HANDEMPTY (if specified)
        if goal_predicates.get("handempty"):
            f.write("    (handempty)\n")
        
        f.write("  ))\n")
        f.write(")\n")

    abs_path = os.path.abspath(filename)
    print(f"[INFO] PDDL problem file written to {abs_path}")
    return abs_path


# ============================================================
# 4️⃣  GOAL VERIFICATION: Check Task Completion
# ============================================================

def goal_achieved(current_predicates, goal_predicates, strict=True):
    """
    Check if the current state satisfies the goal specification.
    
    Used to determine when the task is complete and the TAMP loop can terminate.
    Can operate in strict mode (exact match) or relaxed mode (subset match).
    
    Args:
        current_predicates (dict): Current symbolic state
        goal_predicates (dict): Target symbolic state specification
        strict (bool): If True, check exact match. If False, check subset match.
    
    Returns:
        bool: True if goal is achieved, False otherwise
    
    Matching Modes:
        - Strict (default): Goal predicates must exactly match current predicates
          Example: If goal has 2 ON relations, current must have exactly those 2
        
        - Relaxed: Goal predicates must be subset of current predicates
          Example: If goal has 2 ON relations, current can have those 2 plus more
    
    Algorithm:
        For each predicate type in goal:
            1. Check if predicate type exists in current state
            2. For each value in goal predicate:
               - Verify it exists in current predicate
            3. If any check fails, return False
        Return True if all checks pass
    
    Notes:
        - Handles both list predicates (ON, ONTABLE) and boolean predicates (HANDEMPTY)
        - Empty goal predicates (None or empty list) are considered satisfied
        - Use strict=False for hierarchical goals where partial completion is acceptable
    
    Example:
        goal = {"on": [("r", "g")], "ontable": ["g"], "clear": ["r"]}
        current = {"on": [("r", "g")], "ontable": ["g", "b"], "clear": ["r", "b"]}
        
        goal_achieved(current, goal, strict=False) → True (goal is subset)
        goal_achieved(current, goal, strict=True) → False (extra predicates present)
    """
    for predicate_type, goal_values in goal_predicates.items():
        # Skip empty or None goal values
        if not goal_values:
            continue
        
        # Check if this predicate type exists in current state
        if predicate_type not in current_predicates:
            return False
        
        current_values = current_predicates[predicate_type]
        
        # For each required goal value, verify it's in current state
        for goal_val in goal_values:
            if goal_val not in current_values:
                return False
        
        # In strict mode, also check no extra predicates exist
        if strict and predicate_type not in ["handempty", "holding"]:
            # For list predicates, lengths should match
            if len(current_values) != len(goal_values):
                return False
    
    return True


# ============================================================
# 5️⃣  STABILITY CHECKING: Physical Verification
# ============================================================

def check_tower_stable(blocks_state, tower_blocks, 
                       xy_threshold=STABILITY_XY_THRESHOLD,
                       z_tolerance=STABILITY_Z_TOLERANCE):
    """
    Verify that a tower of blocks is physically stable (properly aligned).
    
    Checks whether blocks in a tower are sufficiently aligned in the horizontal
    plane. This is crucial for detecting execution failures like blocks slipping
    or towers collapsing.
    
    Args:
        blocks_state (dict): Map of block_name -> Genesis entity
        tower_blocks (list): Block names in tower order, bottom to top
                            Example: ["b", "g", "r"] for blue-green-red tower
        xy_threshold (float): Maximum horizontal misalignment in meters
        z_tolerance (float): Vertical position variation tolerance in meters
    
    Returns:
        bool: True if tower is stable (all blocks aligned), False otherwise
    
    Stability Criteria:
        1. Horizontal Alignment: All blocks' x,y centers within threshold of base
        2. Vertical Spacing: Blocks separated by approximately BLOCK_SIZE
        3. No Missing Blocks: All specified blocks have valid positions
    
    Algorithm:
        1. Extract positions of all blocks in tower
        2. Use bottom block as alignment reference (base_x, base_y)
        3. For each block above base:
           - Check horizontal offset from base
           - If offset > threshold, tower is unstable
        4. Optionally check vertical spacing consistency
    
    Use Cases:
        - After executing STACK action: verify block didn't slip
        - Periodic monitoring: detect gradual tower collapse
        - Before planning: ensure preconditions are still valid
        - Replanning trigger: detected instability requires new plan
    
    Example:
        tower_blocks = ["b", "g", "r"]  # Bottom to top
        positions = {"b": [0.5, 0.0, 0.02], "g": [0.5, 0.0, 0.06], "r": [0.52, 0.0, 0.10]}
        
        check_tower_stable(...) → False  (red block misaligned by 2cm)
    
    Notes:
        - Returns False if any block in tower has no position (error state)
        - More lenient than ON predicate threshold to account for physics settling
        - Can be extended to check verticality (all blocks aligned in column)
    """
    # Extract positions of all tower blocks
    positions = {}
    for name in tower_blocks:
        if name not in blocks_state:
            print(f"[WARN] Block '{name}' not found in blocks_state")
            return False
        
        try:
            pos = blocks_state[name].get_pos()
            positions[name] = np.array(pos, dtype=float)
        except Exception as e:
            print(f"[WARN] Failed to get position for block '{name}': {e}")
            return False
    
    if len(positions) != len(tower_blocks):
        print("[WARN] Not all tower blocks have valid positions")
        return False
    
    # Use base (bottom) block as reference for alignment
    base_name = tower_blocks[0]
    base_x, base_y, base_z = positions[base_name]
    
    # Check each block above the base for horizontal alignment
    for i, name in enumerate(tower_blocks[1:], start=1):
        x, y, z = positions[name]
        
        # Horizontal misalignment
        dx = abs(x - base_x)
        dy = abs(y - base_y)
        
        if dx > xy_threshold or dy > xy_threshold:
            print(f"[WARN] Tower unstable: block '{name}' misaligned by ({dx:.3f}, {dy:.3f})m")
            return False
        
        # Optional: Check vertical spacing is approximately correct
        # Expected z = base_z + i * BLOCK_SIZE
        expected_z = base_z + i * BLOCK_SIZE
        dz = abs(z - expected_z)
        
        if dz > z_tolerance * i:  # Allow more tolerance for higher blocks
            print(f"[WARN] Tower unstable: block '{name}' at wrong height (dz={dz:.3f}m)")
            return False
    
    return True


def get_block_on_top(block_name, predicates):
    """
    Find the block directly on top of a given block.
    
    Useful for tower traversal and dependency analysis in task planning.
    
    Args:
        block_name (str): Name of the base block to check
        predicates (dict): Current symbolic predicates
    
    Returns:
        str or None: Name of block on top, or None if nothing is on top
    
    Example:
        predicates = {"on": [("r", "g"), ("g", "b")], ...}
        get_block_on_top("g", predicates) → "r"
        get_block_on_top("r", predicates) → None
    """
    for (top, bottom) in predicates.get("on", []):
        if bottom == block_name:
            return top
    return None


def get_blocks_below(block_name, predicates):
    """
    Find the block directly below a given block.
    
    Inverse of get_block_on_top, useful for determining support structure.
    
    Args:
        block_name (str): Name of the block to check
        predicates (dict): Current symbolic predicates
    
    Returns:
        str or None: Name of block below, or None if on table
    
    Example:
        predicates = {"on": [("r", "g"), ("g", "b")], ...}
        get_blocks_below("r", predicates) → "g"
        get_blocks_below("b", predicates) → None (on table)
    """
    for (top, bottom) in predicates.get("on", []):
        if top == block_name:
            return bottom
    return None


def get_tower_height(block_name, predicates):
    """
    Count the number of blocks stacked above a given block (inclusive).
    
    Useful for:
    - Determining tower complexity
    - Estimating task difficulty
    - Validating goal configurations
    - Bonus point calculation (tallest tower challenge)
    
    Args:
        block_name (str): Name of block to measure from (base)
        predicates (dict): Current symbolic predicates
    
    Returns:
        int: Height of tower starting from block_name (minimum 1)
    
    Algorithm:
        1. Start with height = 1 (the block itself)
        2. Check if any block is on top of current block
        3. If yes, increment height and repeat from top block
        4. If no, return current height
    
    Example:
        predicates = {"on": [("r", "g"), ("g", "b")], ...}
        get_tower_height("b", predicates) → 3  (b, g, r)
        get_tower_height("g", predicates) → 2  (g, r)
        get_tower_height("r", predicates) → 1  (just r)
    
    Notes:
        - Always returns at least 1 (the block itself)
        - Follows tower upward until reaching a CLEAR block
        - Assumes no cycles in ON relations (validated by predicate computation)
    """
    height = 1
    current = block_name
    
    # Traverse up the tower until no block on top
    while True:
        top_block = get_block_on_top(current, predicates)
        if top_block is None:
            break  # Reached the top
        height += 1
        current = top_block
    
    return height


def get_all_towers(predicates):
    """
    Identify all distinct towers in the current scene.
    
    A tower is a sequence of blocks connected by ON relations, starting from
    a block on the table and extending upward.
    
    Args:
        predicates (dict): Current symbolic predicates
    
    Returns:
        list of lists: Each inner list is a tower (bottom to top)
                      Example: [["b", "g", "r"], ["c", "m", "y"]]
    
    Algorithm:
        1. Find all blocks on table (tower bases)
        2. For each base, traverse upward following ON relations
        3. Collect blocks in sequence until reaching top (CLEAR block)
    
    Use Cases:
        - Visualizing current scene structure
        - Identifying which towers need modification
        - Computing goal distance metrics
        - Generating progress reports
    
    Example:
        predicates = {
            "on": [("r", "g"), ("g", "b"), ("y", "c")],
            "ontable": ["b", "c"],
            ...
        }
        get_all_towers(predicates) → [["b", "g", "r"], ["c", "y"]]
    """
    towers = []
    
    # Start from each block on the table
    for base_block in predicates.get("ontable", []):
        tower = [base_block]
        current = base_block
        
        # Build tower upward
        while True:
            top_block = get_block_on_top(current, predicates)
            if top_block is None:
                break
            tower.append(top_block)
            current = top_block
        
        towers.append(tower)
    
    return towers


# ============================================================
# 6️⃣  DEBUGGING AND VISUALIZATION
# ============================================================

def visualize_predicates(preds, title="Current Symbolic State"):
    """
    Print predicates in human-readable format for debugging and logging.
    
    Useful for:
    - Debugging predicate computation issues
    - Monitoring TAMP execution progress
    - Logging state transitions for replay/analysis
    - Verifying goal specifications
    
    Args:
        preds (dict): Predicate dictionary to visualize
        title (str): Header title for the output
    
    Output Format:
        ```
        === Current Symbolic State ===
        (on r g)
        (on g b)
        (ontable b)
        (clear r)
        (handempty)
        ==============================
        ```
    
    Notes:
        - Formats binary predicates (ON) with both arguments
        - Formats unary predicates (ONTABLE, CLEAR) with single argument
        - Formats nullary predicates (HANDEMPTY) with no arguments
        - Skips empty predicate lists for cleaner output
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print('=' * 50)
    
    # Track if we printed anything
    printed_any = False
    
    # Binary predicates (ON)
    if preds.get("on"):
        for (a, b) in preds["on"]:
            print(f"  (on {a} {b})")
            printed_any = True
    
    # Unary predicates (ONTABLE, CLEAR, HOLDING)
    for pred_type in ["ontable", "clear", "holding"]:
        if preds.get(pred_type):
            for a in preds[pred_type]:
                print(f"  ({pred_type} {a})")
                printed_any = True
    
    # Nullary predicates (HANDEMPTY)
    if preds.get("handempty"):
        print("  (handempty)")
        printed_any = True
    
    if not printed_any:
        print("  (no predicates)")
    
    print('=' * 50 + '\n')


def compare_states(current_preds, goal_preds):
    """
    Compare current state with goal state and report differences.
    
    Helpful for debugging why goals aren't being achieved and understanding
    what actions still need to be taken.
    
    Args:
        current_preds (dict): Current symbolic state
        goal_preds (dict): Target symbolic state
    
    Prints:
        - Predicates present in goal but missing from current (needs to add)
        - Predicates present in current but not in goal (needs to remove)
        - Predicates satisfied (already correct)
    
    Example Output:
        ```
        === State Comparison ===
        ✓ Satisfied: (on r g)
        ✗ Missing: (on g b)
        ⚠ Extra: (ontable g)
        ```
    """
    print("\n=== State Comparison: Current vs Goal ===")
    
    # Check ON predicates
    goal_on = set(goal_preds.get("on", []))
    current_on = set(current_preds.get("on", []))
    
    satisfied_on = goal_on & current_on
    missing_on = goal_on - current_on
    extra_on = current_on - goal_on
    
    if satisfied_on:
        print("\n✓ Satisfied ON predicates:")
        for (a, b) in satisfied_on:
            print(f"    (on {a} {b})")
    
    if missing_on:
        print("\n✗ Missing ON predicates (need to achieve):")
        for (a, b) in missing_on:
            print(f"    (on {a} {b})")
    
    if extra_on:
        print("\n⚠ Extra ON predicates (need to remove):")
        for (a, b) in extra_on:
            print(f"    (on {a} {b})")
    
    # Check ONTABLE predicates
    goal_ontable = set(goal_preds.get("ontable", []))
    current_ontable = set(current_preds.get("ontable", []))
    
    missing_ontable = goal_ontable - current_ontable
    extra_ontable = current_ontable - goal_ontable
    
    if missing_ontable:
        print("\n✗ Missing ONTABLE predicates:")
        for a in missing_ontable:
            print(f"    (ontable {a})")
    
    if extra_ontable:
        print("\n⚠ Extra ONTABLE predicates:")
        for a in extra_ontable:
            print(f"    (ontable {a})")
    
    print("\n" + "=" * 40 + "\n")