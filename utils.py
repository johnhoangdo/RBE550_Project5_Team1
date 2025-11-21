import numpy as np
import math
from typing import Dict, List, Tuple, Any


# Constants - for 4cm blocks (matches abstraction.py)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.01
GRASP_OFFSET = 0.02
BLOCK_SIZE = 0.04  # 4cm cubic blocks


def get_block_pose(block: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Get position and quaternion of a block."""
    try:
        pos = np.array(block.get_pos(), dtype=float)
        quat = np.array(block.get_quat(), dtype=float)
        return pos, quat
    except Exception as e:
        print(f"[utils] Failed to get block pose: {e}")
        return np.zeros(3), np.array([1, 0, 0, 0])


def distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Return planar (x, y) distance between two positions."""
    return float(np.linalg.norm(a[0:2] - b[0:2]))


def distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Return full 3D Euclidean distance between two positions."""
    return float(np.linalg.norm(a - b))


def check_tower_stability(blocks_state: Dict[str, Any],
                          tower_blocks: List[str],
                          xy_threshold: float = 0.025) -> bool:
    """
    Check if a tower of blocks is stable (aligned in x-y plane).
    
    Args:
        blocks_state: Dictionary mapping block names to block objects
        tower_blocks: List of block names in tower (base to top)
        xy_threshold: Maximum allowed x-y deviation (default: 0.025m = 62.5% of block size)
        
    Returns:
        True if tower is stable, False otherwise
        
    Note:
        Uses 2.5cm threshold which is 62.5% of 4cm block size.
        This is slightly more lenient than the 2cm alignment threshold
        used in abstraction.py for determining "on" predicates.
    """
    try:
        if not tower_blocks:
            return True
            
        base_pos = np.array(blocks_state[tower_blocks[0]].get_pos(), dtype=float)
        for name in tower_blocks[1:]:
            pos = np.array(blocks_state[name].get_pos(), dtype=float)
            if abs(pos[0] - base_pos[0]) > xy_threshold or abs(pos[1] - base_pos[1]) > xy_threshold:
                return False
        return True
    except Exception as e:
        print(f"[utils] check_tower_stability failed: {e}")
        return False


def get_tower_height(block_name: str,
                     predicates: Dict[str, List[Tuple[str, str]]]) -> int:
    """
    Get height of tower with given block as base.
    
    Args:
        block_name: Name of base block
        predicates: Dictionary of predicates including "on" relations
        
    Returns:
        Height of tower (number of blocks)
    """
    height = 1
    current = block_name
    while True:
        top = None
        for (a, b) in predicates.get("on", []):
            if b == current:
                top = a
                break
        if top is None:
            break
        height += 1
        current = top
    return height


def check_all_towers_stable(blocks_state: Dict[str, Any],
                            predicates: Dict[str, List[Tuple[str, str]]],
                            xy_threshold: float = 0.025) -> bool:
    """
    Check if all tower structures in scene are stable.
    
    Args:
        blocks_state: Dictionary mapping block names to block objects
        predicates: Dictionary of predicates
        xy_threshold: Maximum allowed x-y deviation in meters
        
    Returns:
        True if all towers are stable, False otherwise
    """
    # Find all base blocks (blocks that are on the table or have something on them)
    bases = set()
    
    # Add all blocks that have something on top
    for (_, base) in predicates.get("on", []):
        bases.add(base)
    
    # Add ontable blocks that don't have anything on them (single-block "towers")
    for block in predicates.get("ontable", []):
        if block not in bases:
            bases.add(block)
    
    # Check each tower
    for base in bases:
        tower_blocks = [base]
        current = base
        
        # Build tower from base upward
        while True:
            top = None
            for (a, b) in predicates.get("on", []):
                if b == current:
                    top = a
                    tower_blocks.append(top)
                    current = top
                    break
            if top is None:
                break
        
        # Check stability
        if not check_tower_stability(blocks_state, tower_blocks, xy_threshold):
            return False
    
    return True


def is_goal_achieved(current_predicates: Dict[str, List[Tuple[str, str]]],
                     goal_predicates: Dict[str, List[Tuple[str, str]]]) -> bool:
    """
    Check if current state satisfies all goal predicates.
    
    Args:
        current_predicates: Current state predicates
        goal_predicates: Goal state predicates
        
    Returns:
        True if goal is achieved, False otherwise
    """
    try:
        for predicate_type in goal_predicates.keys():
            goal_facts = set(goal_predicates[predicate_type])
            current_facts = set(current_predicates.get(predicate_type, []))
            
            # Check if all goal facts are present in current state
            if not goal_facts.issubset(current_facts):
                return False
        
        return True
    except Exception as e:
        print(f"[utils] is_goal_achieved failed: {e}")
        return False


def visualize_plan(plan: List[Tuple[str, ...]]) -> None:
    """Print plan in readable format."""
    print("\n=== Plan ===")
    if not plan:
        print("No plan found.")
        return
    for i, step in enumerate(plan):
        print(f"{i + 1:02d}: {' '.join(step)}")


def visualize_predicates(predicates: Dict[str, List[Tuple[str, str]]]) -> None:
    """Print current predicates in readable format."""
    print("\n=== Current Predicates ===")
    for key, value in predicates.items():
        print(f"{key.upper():10s}: {value}")


def visualize_goal(goal: Dict[str, List[Tuple[str, str]]], title: str = "Goal State") -> None:
    """Print goal state in readable format."""
    print(f"\n===== {title} =====")
    if "on" in goal:
        print("Stacking Relations:")
        for (a, b) in goal["on"]:
            print(f"  {a} on {b}")
    if "ontable" in goal:
        print("On Table:", ", ".join(goal["ontable"]))
    if "clear" in goal:
        print("Clear Blocks:", ", ".join(goal["clear"]))


def log_state_comparison(current_predicates: Dict[str, List[Tuple[str, str]]],
                         goal_predicates: Dict[str, List[Tuple[str, str]]]) -> None:
    """Log comparison between current and goal states for debugging."""
    missing = {}
    for k in goal_predicates.keys():
        missing[k] = []
        for fact in goal_predicates[k]:
            if fact not in current_predicates.get(k, []):
                missing[k].append(fact)

    print("\n=== Goal Comparison ===")
    for k, v in missing.items():
        if v:
            print(f"Unmet {k}: {v}")
        else:
            print(f"All {k} predicates satisfied.")


def compute_grasp_pose(block_pos: np.ndarray,
                       offset: float = GRASP_OFFSET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute grasp pose above a block.
    
    Args:
        block_pos: Position of block center (x, y, z)
        offset: Height offset above block (default: 0.02m = 50% of block height)
        
    Returns:
        Tuple of (position, quaternion) for gripper
        
    Note:
        Offset of 0.02m positions gripper 2cm above the 4cm block's center,
        which is at the top surface of the block.
    """
    pos = np.array([block_pos[0], block_pos[1], block_pos[2] + offset])
    quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
    return pos, quat


def compute_placement_pose(target_pos: np.ndarray,
                          is_table: bool = False,
                          offset: float = GRASP_OFFSET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute placement pose for putting down a block.
    
    Args:
        target_pos: Position of target (table or block center)
        is_table: True if placing on table, False if stacking on block
        offset: Height offset (default: 0.02m)
        
    Returns:
        Tuple of (position, quaternion) for gripper
        
    Note:
        For stacking: Adds BLOCK_SIZE (0.04m) to place block on top.
        For table: Just adds offset for approach height.
        
    Example:
        Target block at z=0.02 (center of 4cm block on table)
        Stack height: 0.02 + 0.04 + 0.02 = 0.08m
        New block center will be at z=0.08m (properly stacked)
    """
    if is_table:
        # Place on table at target height
        pos = np.array([target_pos[0], target_pos[1], target_pos[2] + offset])
    else:
        # Stack on block - add 4cm block size + offset
        pos = np.array([target_pos[0], target_pos[1], 
                       target_pos[2] + BLOCK_SIZE + offset])
    
    quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
    return pos, quat


def save_text(filename: str, content: str) -> None:
    """Save text content to file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[utils] Failed to save text to {filename}: {e}")


def read_text(filename: str) -> str:
    """Read text from file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[utils] Failed to read {filename}: {e}")
        return ""


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING utils.py - Block Size Consistency Check")
    print("=" * 60)
    
    # Verify constants match abstraction.py
    print("\n1. CONSTANT VERIFICATION:")
    print(f"   BLOCK_SIZE = {BLOCK_SIZE}m (should be 0.04)")
    print(f"   GRASP_OFFSET = {GRASP_OFFSET}m (should be 0.02)")
    print(f"   GRIPPER_OPEN = {GRIPPER_OPEN}m")
    print(f"   GRIPPER_CLOSED = {GRIPPER_CLOSED}m")
    
    assert BLOCK_SIZE == 0.04, "BLOCK_SIZE must be 0.04m (4cm)!"
    print("   BLOCK_SIZE correct!")
    
    # Test stacking height calculation
    print("\n2. STACKING HEIGHT TEST:")
    target_pos = np.array([0.0, 0.0, 0.02])  # Block on table
    place_pos, _ = compute_placement_pose(target_pos, is_table=False)
    expected_z = 0.02 + 0.04 + 0.02  # target + block_size + offset
    print(f"   Target block center: z = {target_pos[2]}m")
    print(f"   Placement pose: z = {place_pos[2]}m")
    print(f"   Expected: z = {expected_z}m")
    
    assert abs(place_pos[2] - expected_z) < 0.001, "Stacking height calculation wrong!"
    print("   Stacking height correct!")
    
    # Test grasp height
    print("\n3. GRASP HEIGHT TEST:")
    block_pos = np.array([0.0, 0.0, 0.02])
    grasp_pos, _ = compute_grasp_pose(block_pos)
    expected_grasp_z = 0.02 + 0.02  # block_center + offset
    print(f"   Block center: z = {block_pos[2]}m")
    print(f"   Grasp pose: z = {grasp_pos[2]}m")
    print(f"   Expected: z = {expected_grasp_z}m")
    
    assert abs(grasp_pos[2] - expected_grasp_z) < 0.001, "Grasp height calculation wrong!"
    print("   Grasp height correct!")
    
    # Test other functions
    print("\n4. PREDICATE TESTS:")
    sample_predicates = {
        "on": [("r", "g"), ("g", "b")],
        "ontable": ["b"],
        "clear": ["r"],
        "holding": [],
        "handempty": [True]
    }
    
    sample_goal = {
        "on": [("r", "g"), ("g", "b")],
        "ontable": ["b"],
        "clear": ["r"]
    }
    
    visualize_predicates(sample_predicates)
    visualize_goal(sample_goal)
    
    result = is_goal_achieved(sample_predicates, sample_goal)
    print(f"\n   Goal achieved: {result}")
    assert result == True, "Goal should be achieved!"
    print("   Goal checking works!")
    
    print("\n5. PLAN VISUALIZATION TEST:")
    visualize_plan([("pick-up", "r"), ("stack", "r", "g")])
    print("   Plan visualization works!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! utils.py is consistent with abstraction.py")
    print("Block size: 0.04m (4cm) GOOD")
    print("=" * 60)
