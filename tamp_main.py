"""
tamp_main.py - FIXED VERSION
----------------------------------
Main Task and Motion Planning (TAMP) loop

CRITICAL FIXES APPLIED:
1. compute_predicates(planner_interface) - not robot! (holding detection)
2. Prepend ALL positioning at START - not inline! (no consecutive pick-ups)
3. Added positioning debug messages
4. Added spatial achievement messages

Author: LA, JHD, JEN
Date: 11/21/2025
"""

import sys
import numpy as np
import genesis as gs

from abstraction import (
    compute_predicates, 
    generate_pddl_problem, 
    goal_achieved, 
    goal_achieved_with_spatial,
    visualize_predicates,
    check_tower_stable,
    get_all_towers
)
from goals import GOAL_TWO_TOWERS, GOAL_SIX_TOWER, GOAL_FIVE_TOWER, get_goal
from task_planner import call_planner, parse_plan_output, validate_plan
from planning import PlannerInterface
from scenes import (
    create_scene_6blocks,
    create_scene_12_yellow_blocks,
    create_scene_3red_3green
)


# ============================================================
# POSITION CHECKING FUNCTIONS (for recovery from collisions)
# ============================================================

def check_all_block_positions(blocks_state, spatial_targets, tolerance=0.010):
    """
    Check if any PLACED blocks have been knocked out of their target positions.
    
    CRITICAL: Only checks blocks that are ON the table (not held, not in spawn area)
    
    Args:
        blocks_state: Dict of block_name -> Genesis block entity
        spatial_targets: Dict of block_name -> (x, y) target position
        tolerance: Position tolerance in meters (default 10mm)
    
    Returns:
        List of (block_name, current_pos, target_pos, error) for misplaced blocks
    """
    misplaced_blocks = []
    
    for block_name, target_pos in spatial_targets.items():
        if block_name not in blocks_state:
            continue
        
        block = blocks_state[block_name]
        current_pos = block.get_pos()
        
        # CRITICAL FIX: Only check blocks that are on the table (Z ~ 0.02)
        # Skip blocks that haven't been placed yet (still in spawn area, Z ~ 0.15+)
        # Skip blocks that are being held (Z > 0.10)
        if current_pos[2] > 0.10:  # Block is in spawn area or being held
            continue
        
        # Check XY position only (Z doesn't matter for base blocks)
        dx = abs(current_pos[0] - target_pos[0])
        dy = abs(current_pos[1] - target_pos[1])
        
        if dx > tolerance or dy > tolerance:
            error = (dx * 1000, dy * 1000)  # Convert to mm
            misplaced_blocks.append((block_name, current_pos, target_pos, error))
    
    return misplaced_blocks


def reposition_knocked_blocks(planner, scene, blocks_state, spatial_targets, tolerance=0.010):
    """
    Reposition any blocks that have been knocked out of place.
    
    This is called between major actions to ensure the grid stays intact.
    
    Args:
        planner: PlannerInterface instance
        scene: Genesis scene
        blocks_state: Dict of block_name -> block entity
        spatial_targets: Dict of block_name -> target position
        tolerance: Position tolerance (default 10mm)
    
    Returns:
        int: Number of blocks repositioned
    """
    misplaced_blocks = check_all_block_positions(blocks_state, spatial_targets, tolerance)
    
    if not misplaced_blocks:
        return 0
    
    print("\n" + "="*60)
    print(f"[RECOVERY] Found {len(misplaced_blocks)} misplaced blocks!")
    print("="*60)
    
    for block_name, current_pos, target_pos, error in misplaced_blocks:
        print(f"  {block_name}: ({current_pos[0]:.3f}, {current_pos[1]:.3f}) "
              f"→ ({target_pos[0]:.3f}, {target_pos[1]:.3f}) "
              f"[error: {error[0]:.1f}mm, {error[1]:.1f}mm]")
    
    repositioned_count = 0
    
    for block_name, current_pos, target_pos, error in misplaced_blocks:
        print(f"\n[RECOVERY] Repositioning {block_name}...")
        
        block = blocks_state[block_name]
        
        # Pick up the misplaced block
        if not planner.pick_up(block):
            print(f"  ✗ Failed to pick up {block_name}")
            continue
        
        # Put it down at the correct position
        target_3d = np.array([target_pos[0], target_pos[1], 0.02])
        if planner.put_down(target_3d):
            print(f"  ✓ Successfully repositioned {block_name}")
            repositioned_count += 1
        else:
            print(f"  ✗ Failed to reposition {block_name}")
    
    if repositioned_count > 0:
        print(f"\n[RECOVERY] Repositioned {repositioned_count} blocks")
        print("="*60)
    
    return repositioned_count


def execute_primitive(action_tuple, planner, scene, blocks_state):
    """Execute one symbolic action using PlannerInterface primitives."""
    act = action_tuple[0].lower().replace("_", "-")
    args = list(action_tuple[1:])
    
    print(f"\n[ACTION] Executing: {act} {' '.join(args)}")

    try:
        if act == "pick-up":
            if len(args) < 1:
                print("[ERROR] pick-up requires 1 argument")
                return False
            
            block_name = args[0]
            if block_name not in blocks_state:
                print(f"[ERROR] Block '{block_name}' not found")
                return False
            
            block = blocks_state[block_name]
            success = planner.pick_up(block)
            
            if success:
                print(f" ✓ Successfully picked up {block_name}")
            else:
                print(f" ✗ Failed to pick up {block_name}")
            
            return success

        elif act == "put-down":
            if len(args) < 1:
                print("[ERROR] put-down requires at least 1 argument")
                return False
            
            held_obj = planner.attached_object
            if held_obj is None:
                print("[WARN] Nothing to put down")
                return False
            
            block_name = None
            for name, block in blocks_state.items():
                if block == held_obj:
                    block_name = name
                    break
            
            if len(args) >= 3:
                try:
                    target_x = float(args[1])
                    target_y = float(args[2])
                    target_pos = np.array([target_x, target_y, 0.02])
                    print(f"  Target position: ({target_x:.3f}, {target_y:.3f})")
                except (ValueError, IndexError):
                    pos = held_obj.get_pos()
                    target_pos = np.array([pos[0], pos[1], 0.02])
            else:
                pos = held_obj.get_pos()
                target_pos = np.array([pos[0], pos[1], 0.02])
            
            success = planner.put_down(target_pos)
            
            if success:
                print(f"  ✓ Successfully put down {block_name or 'block'}")
            else:
                print(f"  ✗ Failed to put down {block_name or 'block'}")
            
            return success

        elif act == "stack":
            if len(args) < 2:
                print("[ERROR] stack requires 2 arguments")
                return False
            
            block_a_name = args[0]
            block_b_name = args[1]
            
            if block_b_name not in blocks_state:
                print(f"[ERROR] Target block '{block_b_name}' not found")
                return False
            
            block_b = blocks_state[block_b_name]
            
            held_obj = planner.attached_object
            if held_obj is None:
                print(f"[WARN] Not holding {block_a_name}, trying to pick it up first...")
                if block_a_name in blocks_state:
                    if not planner.pick_up(blocks_state[block_a_name]):
                        print(f"[ERROR] Failed to pick up {block_a_name}")
                        return False
                else:
                    print(f"[ERROR] Block '{block_a_name}' not found")
                    return False
            
            success = planner.stack(block_b)
            
            if success:
                print(f" ✓ Successfully stacked {block_a_name} on {block_b_name}")
            else:
                print(f" ✗ Failed to stack {block_a_name} on {block_b_name}")
            
            return success

        elif act == "unstack":
            if len(args) < 2:
                print("[ERROR] unstack requires 2 arguments")
                return False
            
            block_a_name = args[0]
            block_b_name = args[1]
            
            if block_a_name not in blocks_state:
                print(f"[ERROR] Block '{block_a_name}' not found")
                return False
            
            block_a = blocks_state[block_a_name]
            success = planner.pick_up(block_a)
            
            if success:
                print(f"  ✓ Successfully unstacked {block_a_name} from {block_b_name}")
            else:
                print(f"  ✗ Failed to unstack {block_a_name}")
            
            return success

        else:
            print(f"[ERROR] Unknown action type: {act}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception during {act}: {e}")
        import traceback
        traceback.print_exc()
        return False


def augment_plan_with_positioning(plan, current_state, goal_predicates, blocks_state):
    """
    CRITICAL FIX: Prepend ALL positioning actions at START (not inline!)
    ALSO: Always put down currently held block first!
    
    This prevents consecutive pick-ups which violate blocksworld constraints.
    """
    if "spatial" not in goal_predicates:
        return plan
    
    spatial_targets = goal_predicates["spatial"]
    positioning_actions = []
    blocks_to_position = set()
    
    print("\n[SPATIAL] Augmenting plan with positioning moves...")
    
    # CRITICAL: Check if robot is currently holding a block that needs positioning
    held_block_name = None
    if "holding" in current_state and current_state["holding"]:
        held_block_name = current_state["holding"][0]
    
    # Step 1: Find ALL base blocks that will be stacked on
    for action in plan:
        if action[0] == "stack" and len(action) >= 3:
            bottom_block = action[2]
            if bottom_block in spatial_targets:
                blocks_to_position.add(bottom_block)
    
    # Step 2: If holding a block that needs positioning, put it down FIRST!
    if held_block_name and held_block_name in spatial_targets:
        target_pos = spatial_targets[held_block_name]
        print(f"  [HELD] Putting down {held_block_name} to ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
        positioning_actions.append(("put-down", held_block_name,
                                   str(target_pos[0]), str(target_pos[1])))
        # Remove from blocks_to_position since we're handling it now
        blocks_to_position.discard(held_block_name)
    
    # Step 3: Create positioning for remaining base blocks
    for bottom_block in sorted(blocks_to_position):  # Sort for deterministic order
        target_pos = spatial_targets[bottom_block]
        current_pos = blocks_state[bottom_block].get_pos()
        distance = ((current_pos[0] - target_pos[0])**2 + 
                   (current_pos[1] - target_pos[1])**2)**0.5
        
        if distance > 0.05:
            print(f"  Positioning {bottom_block} to ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
            positioning_actions.append(("pick-up", bottom_block))
            positioning_actions.append(("put-down", bottom_block,
                                       str(target_pos[0]), str(target_pos[1])))
    
    if blocks_to_position or held_block_name:
        all_positioned = blocks_to_position.copy()
        if held_block_name and held_block_name in spatial_targets:
            all_positioned.add(held_block_name)
        print(f"[SPATIAL] Positioned {len(all_positioned)} blocks: {all_positioned}")
    
    # Step 4: Return positioning FIRST, then original plan
    return positioning_actions + plan


def tamp_loop(scene, robot, blocks_state, 
              goal_predicates=GOAL_TWO_TOWERS,
              domain_file="blocksworld_domain.pddl",
              max_iterations=10,
              use_spatial=False,
              planning_timeout=30):
    """
    TAMP loop with critical bug fixes:
    1. Uses planner_interface for compute_predicates (holding detection)
    2. Prepends positioning (no consecutive pick-ups)
    """
    
    planner_interface = PlannerInterface(robot, scene)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations}".center(60))
        print('='*60)

        # CRITICAL FIX #1: Pass planner_interface (not robot) so compute_predicates can see attached_object!
        current_state = compute_predicates(planner_interface, blocks_state, scene)
        
        if use_spatial:
            goal_met = goal_achieved_with_spatial(
                current_state, goal_predicates, blocks_state
            )
        else:
            goal_met = goal_achieved(current_state, goal_predicates)
        
        if goal_met:
            print("\n" + "="*60)
            print("✓ GOAL ACHIEVED!".center(60))
            if use_spatial:
                print("(Including spatial constraints)".center(60))
            print("="*60)
            return True

        goal_pddl = {k: v for k, v in goal_predicates.items() if k != "spatial"}

        problem_file = generate_pddl_problem(
            current_state, goal_pddl, "current_problem.pddl"
        )

        plan = call_planner(domain_file, problem_file,
                           use_pyperplan=True,
                           timeout=planning_timeout)

        if not plan:
            print("\n✗ No plan found. Unable to achieve goal.")
            return False

        plan = parse_plan_output(plan)

        # CRITICAL FIX #2: Prepend ALL positioning at start (not inline!)
        if use_spatial and "spatial" in goal_predicates:
            plan = augment_plan_with_positioning(
                plan, current_state, goal_predicates, blocks_state
            )

        # Execute plan with intermediate position checking
        for i, action in enumerate(plan):
            if not execute_primitive(action, planner_interface, scene, blocks_state):
                print("\n❌ Execution failed, replanning...")
                break
            
            # INTERMEDIATE POSITION CHECK: After every action, check if any blocks got knocked
            # This is especially important for tight grids where collisions can knock blocks
            if use_spatial and "spatial" in goal_predicates:
                # Check every 2 actions (not every single one to save time)
                if i % 2 == 1 or i == len(plan) - 1:
                    misplaced = check_all_block_positions(
                        blocks_state, 
                        goal_predicates["spatial"], 
                        tolerance=0.010
                    )
                    
                    if misplaced:
                        print(f"\n⚠️ [RECOVERY] Detected {len(misplaced)} misplaced blocks after action {i+1}/{len(plan)}")
                        repositioned = reposition_knocked_blocks(
                            planner_interface,
                            scene,
                            blocks_state,
                            goal_predicates["spatial"],
                            tolerance=0.010
                        )
                        
                        if repositioned > 0:
                            print(f"✓ [RECOVERY] Successfully repositioned {repositioned} blocks")
                        else:
                            print(f"⚠️ [RECOVERY] Could not reposition all blocks, continuing...")
        
    print("\n" + "="*60)
    print("✗ FAILED. Goal not achieved.".center(60))
    print("="*60)
    return False


def main():
    print("\n" + "="*60)
    print("PROJECT 5: TASK AND MOTION PLANNING (TAMP)".center(60))
    print("Building Towers with Symbolic Planning".center(60))
    print("="*60)
    
    use_gpu = "gpu" in sys.argv
    use_goal1 = "--goal1" in sys.argv or "--two-towers" in sys.argv
    use_goal2 = "--goal2" in sys.argv or "--five-tower" in sys.argv
    use_goal3 = "--goal3" in sys.argv or "--six-tower" in sys.argv
    use_goal3_extended = "--goal3-ext" in sys.argv
    use_goal4a = "--goal4a" in sys.argv or "--tower-grid" in sys.argv
    use_goal4a_simple = "--goal4a-simple" in sys.argv or "--test" in sys.argv
    use_goal4b = "--goal4b" in sys.argv or "--adjacent" in sys.argv
    
    # Mode selection
    if use_goal1:
        print("\n[MODE] Goal 1: Two 3-Block Towers")
        max_iterations = 10
        planning_timeout = 20
    elif use_goal2:
        print("\n[MODE] Goal 2: Five-Block Tower")
        max_iterations = 12
        planning_timeout = 25
    elif use_goal3:
        print("\n[MODE] Goal 3: Single 6-Block Tall Tower")
        max_iterations = 10
        planning_timeout = 30
    elif use_goal3_extended:
        print("\n[MODE] Goal 3 Extended (10+ blocks)")
        max_iterations = 30
        planning_timeout = 60
    elif use_goal4a or use_goal4b:
        print("\n[MODE] Goal 4 - Spatial Grid Structures")
        max_iterations = 15
        planning_timeout = 90
    else:
        print("\n[MODE] Goal 3 (6-block tower)")
        max_iterations = 10
        planning_timeout = 30
    
    if use_gpu:
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)
    
    # Scene selection
    if use_goal1:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "two_towers"
        use_spatial = False
    elif use_goal2:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "five_tower"
        use_spatial = False
    elif use_goal3:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "six_tower"
        use_spatial = False
    elif use_goal4a:
        scene, franka, blocks_state = create_scene_12_yellow_blocks()
        goal_name_str = "tower_grid"
        use_spatial = True
    elif use_goal4b:
        scene, franka, blocks_state = create_scene_3red_3green()
        goal_name_str = "adjacent"
        use_spatial = True
    else:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "six_tower"
        use_spatial = False
    
    goal = get_goal(goal_name_str)

    success = tamp_loop(
        scene=scene,
        robot=franka,
        blocks_state=blocks_state,
        goal_predicates=goal,
        max_iterations=max_iterations,
        use_spatial=use_spatial,
        planning_timeout=planning_timeout
    )
    
    print("\n" + "="*60)
    if success:
        print("SUCCESS".center(60))
    else:
        print("FAILED".center(60))
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
