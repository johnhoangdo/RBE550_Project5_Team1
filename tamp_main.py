"""
tamp_main.py
----------------------------------
Main Task and Motion Planning (TAMP) loop

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
    goal_achieved_with_spatial,  # NEW for Goal 4
    visualize_predicates,
    check_tower_stable,
    get_all_towers
)
from goals import GOAL_TWO_TOWERS, GOAL_SIX_TOWER, GOAL_FIVE_TOWER, get_goal
from task_planner import call_planner, parse_plan_output, validate_plan
from planning import PlannerInterface
from scenes import (
    create_scene_6blocks,
    create_scene_12_yellow_blocks,  # NEW for Goal 4A
    create_scene_3red_3green        # NEW for Goal 4B
)


def verify_tower_stability(blocks_state, tower_blocks, max_retries=3):
    """
    Check tower stability with retry mechanism for Goal 3
    """
    from abstraction import check_tower_stable
    
    for attempt in range(max_retries):
        # Let tower settle
        for _ in range(100):
            scene.step()
        
        # Check stability
        if check_tower_stable(blocks_state, tower_blocks):
            return True
        
        print(f"  Tower unstable, waiting... (attempt {attempt+1}/{max_retries})")
        
        # Wait longer
        for _ in range(300):
            scene.step()
    
    return False


def execute_primitive(action_tuple, planner, scene, blocks_state):
    """
    Execute one symbolic action (pick-up, put-down, stack, unstack)
    using PlannerInterface primitives.

    Args:
        action_tuple: tuple like ('pick-up','r') or ('stack','r','g')
        planner: PlannerInterface instance (handles OMPL + motion)
        scene: Genesis scene
        blocks_state: dict {name:block_entity}
    
    Returns:
        bool: True if action succeeded, False otherwise
    """
    # Normalize action name (handle both pick-up and pick_up)
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
                print(f"[ERROR] Block '{block_name}' not found in blocks_state")
                return False
            
            block = blocks_state[block_name]
            success = planner.pick_up(block)
            
            if success:
                print(f" Successfully picked up {block_name}")
            else:
                print(f" Failed to pick up {block_name}")
            
            return success

        elif act == "put-down":
            if len(args) < 1:
                print("[ERROR] put-down requires at least 1 argument")
                return False
            
            held_obj = planner.attached_object
            if held_obj is None:
                print("[WARN] Nothing to put down")
                return False
            
            # Find name of held object
            block_name = None
            for name, block in blocks_state.items():
                if block == held_obj:
                    block_name = name
                    break
            
            # Check if target position specified (for Goal 4 spatial)
            if len(args) >= 3:
                # Format: put-down block x y
                try:
                    target_x = float(args[1])
                    target_y = float(args[2])
                    target_pos = np.array([target_x, target_y, 0.02])
                    print(f"  Target position: ({target_x:.3f}, {target_y:.3f})")
                except (ValueError, IndexError):
                    # Fallback to current position
                    pos = held_obj.get_pos()
                    target_pos = np.array([pos[0], pos[1], 0.02])
            else:
                # Place on table at current XY position
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
            
            block_a_name = args[0]  # Block to stack (should be held)
            block_b_name = args[1]  # Block to stack on
            
            if block_b_name not in blocks_state:
                print(f"[ERROR] Target block '{block_b_name}' not found")
                return False
            
            block_b = blocks_state[block_b_name]
            
            # Verify we're holding the right block
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
                print(f" Successfully stacked {block_a_name} on {block_b_name}")
                
                # Check stability
                # Build tower from bottom up
                tower = [block_b_name, block_a_name]
                if not check_tower_stable(blocks_state, tower):
                    print(f"  ⚠ Warning: Tower may be unstable!")
            else:
                print(f" Failed to stack {block_a_name} on {block_b_name}")
            
            return success

        elif act == "unstack":
            if len(args) < 2:
                print("[ERROR] unstack requires 2 arguments")
                return False
            
            block_a_name = args[0]  # Block to unstack
            block_b_name = args[1]  # Block it's on
            
            if block_a_name not in blocks_state:
                print(f"[ERROR] Block '{block_a_name}' not found")
                return False
            
            block_a = blocks_state[block_a_name]
            
            # Unstack is essentially pick_up
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
    For Goal 4: Insert pick-up/put-down actions to move base blocks to target positions
    
    This function analyzes the plan and inserts positioning moves BEFORE any block
    is used as a stacking target, ensuring blocks are at their spatial constraint
    positions before being stacked on.
    
    Args:
        plan: Original symbolic plan from PDDL
        current_state: Current symbolic state
        goal_predicates: Goal with spatial constraints
        blocks_state: Block entities for position checking
    
    Returns:
        Augmented plan with positioning moves
    """
    if "spatial" not in goal_predicates:
        return plan
    
    spatial_targets = goal_predicates["spatial"]
    augmented_plan = []
    blocks_positioned = set()  # Track which blocks we've already positioned
    
    print("\n[SPATIAL] Augmenting plan with positioning moves...")
    
    for action in plan:
        action_name = action[0]
        
        # Check if this action uses a block that needs positioning
        if action_name == "stack" and len(action) >= 3:
            top_block = action[1]
            bottom_block = action[2]
            
            # If bottom block needs to be at specific position and hasn't been moved yet
            if bottom_block in spatial_targets and bottom_block not in blocks_positioned:
                target_pos = spatial_targets[bottom_block]
                
                # Check current position
                current_pos = blocks_state[bottom_block].get_pos()
                distance = ((current_pos[0] - target_pos[0])**2 + 
                           (current_pos[1] - target_pos[1])**2)**0.5
                
                # If block is far from target (>5cm), add positioning moves
                if distance > 0.05:
                    print(f"  Positioning {bottom_block} to {target_pos}")
                    
                    # Add pick-up and put-down to move block
                    augmented_plan.append(("pick-up", bottom_block))
                    augmented_plan.append(("put-down", bottom_block, str(target_pos[0]), str(target_pos[1])))
                    
                    blocks_positioned.add(bottom_block)
        
        elif action_name == "put-down" and len(action) >= 2:
            block = action[1]
            
            # If this block has a spatial target, use it
            if block in spatial_targets and block not in blocks_positioned:
                target_pos = spatial_targets[block]
                print(f"  Redirecting put-down of {block} to {target_pos}")
                
                # Replace with positioned put-down
                augmented_plan.append(("put-down", block, str(target_pos[0]), str(target_pos[1])))
                blocks_positioned.add(block)
                continue  # Don't add original action
        
        # Add original action
        augmented_plan.append(action)
    
    if blocks_positioned:
        print(f"[SPATIAL] Positioned {len(blocks_positioned)} blocks: {blocks_positioned}")
    
    return augmented_plan


def tamp_loop(scene, robot, blocks_state, 
              goal_predicates=GOAL_TWO_TOWERS,
              domain_file="blocksworld_domain.pddl",
              max_iterations=10,
              use_spatial=False,
              planning_timeout=30):  # NEW parameter
    """
    Full TAMP execution loop with replanning.

    Steps:
      1. Symbolic abstraction (compute predicates)
      2. Check if goal achieved (with spatial constraints for Goal 4)
      3. Task planning (PDDL → high-level actions)
      4. Validate plan
      5. Execute plan step-by-step
      6. Check stability and replan if needed

    Args:
        scene: Genesis scene
        robot: Franka robot entity (or RobotAdapter)
        blocks_state: dict {block_name: block_entity}
        goal_predicates: goal definition from goals.py
        domain_file: path to blocksworld_domain.pddl
        max_iterations: maximum planning iterations before giving up
        use_spatial: if True, check spatial constraints (Goal 4)
        planning_timeout: timeout for PDDL planner in seconds
    
    Returns:
        bool: True if goal achieved, False if failed
    """
    print("\n" + "="*60)
    print("STARTING TAMP LOOP".center(60))
    if use_spatial:
        print("(with spatial constraint checking)".center(60))
    print("="*60)
    print(f"Planning timeout: {planning_timeout}s")
    
    # Create planner interface
    planner_interface = PlannerInterface(robot, scene)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations}".center(60))
        print('='*60)

        # =====================================================================
        # STEP 1: SYMBOLIC ABSTRACTION (Lifting)
        # =====================================================================
        print("\n[STEP 1] Symbolic Abstraction...")
        current_state = compute_predicates(robot, blocks_state, scene)
        visualize_predicates(current_state, f"Current State (Iteration {iteration})")
        
        # Display current towers
        towers = get_all_towers(current_state)
        if towers:
            print(f"\nCurrent towers ({len(towers)}):")
            for i, tower in enumerate(towers, 1):
                print(f"  Tower {i}: {' -> '.join(tower)} (height: {len(tower)})")

        # =====================================================================
        # STEP 2: CHECK GOAL (with spatial constraints for Goal 4)
        # =====================================================================
        print("\n[STEP 2] Checking goal...")
        
        if use_spatial:
            # Goal 4: Check both predicates AND spatial constraints
            goal_met = goal_achieved_with_spatial(
                current_state, 
                goal_predicates, 
                blocks_state
            )
        else:
            # Goals 1-3: Check predicates only
            goal_met = goal_achieved(current_state, goal_predicates)
        
        if goal_met:
            print("\n" + "="*60)
            print("✓ GOAL ACHIEVED!".center(60))
            if use_spatial:
                print("(Including spatial constraints)".center(60))
            print("="*60)
            return True

        # =====================================================================
        # STEP 3: TASK PLANNING
        # =====================================================================
        print("\n[STEP 3] Task Planning...")
        problem_file = generate_pddl_problem(
            current_state, 
            goal_predicates, 
            filename="current_problem.pddl",
            problem_name=f"tamp-iteration-{iteration}"
        )
        
        print(f"  Generated problem file: {problem_file}")
        
        # Call planner with timeout (longer for complex goals)
        plan = call_planner(domain_file, problem_file, 
                           use_pyperplan=True, 
                           timeout=planning_timeout)
        
        if not plan:
            print("\n✗ No plan found. Unable to achieve goal.")
            return False

        # Parse and validate plan
        plan = parse_plan_output(plan)
        
        # For Goal 4: Insert positioning moves to satisfy spatial constraints
        if use_spatial and "spatial" in goal_predicates:
            plan = augment_plan_with_positioning(
                plan, current_state, goal_predicates, blocks_state
            )
        
        print(f"\n  Generated plan with {len(plan)} actions:")
        for i, step in enumerate(plan, 1):
            action = step[0]
            args = ' '.join(step[1:])
            print(f"    {i}. {action} {args}")
        
        # Validate plan before execution
        print("\n  Validating plan...")
        is_valid, error = validate_plan(plan, current_state, goal_predicates)
        if not is_valid:
            print(f" Warning: Plan validation failed: {error}")
            print("  Proceeding anyway (validation may be overly strict)")

        # =====================================================================
        # STEP 4: EXECUTE PLAN
        # =====================================================================
        print("\n[STEP 4] Executing Plan...")
        
        execution_failed = False
        for i, action in enumerate(plan, 1):
            print(f"\n--- Action {i}/{len(plan)} ---")
            
            success = execute_primitive(action, planner_interface, scene, blocks_state)
            
            if not success:
                print(f"\n  Action {action} failed. Will replan...")
                execution_failed = True
                break
            
            # Let physics settle after each action
            print("  Settling physics...")
            for _ in range(300):
                scene.step()
        
        if execution_failed:
            print("\n Execution failed, replanning...")
            continue  # Go to next iteration

        # =====================================================================
        # STEP 5: VERIFY EXECUTION
        # =====================================================================
        print("\n[STEP 5] Verifying execution...")
        
        # Let physics fully settle
def main():
    """
    Main entry point with support for Goals 3, 3 Extended, and 4
    
    Command-line arguments:
        gpu              - Use GPU backend instead of CPU
        --goal3-ext      - Goal 3 Extended mode (10+ blocks)
        --goal4a         - Goal 4A: Tower Grid (12 yellow blocks)
        --goal4b         - Goal 4B: Adjacent (3 red + 3 green)
    """
    print("\n" + "="*60)
    print("PROJECT 5: TASK AND MOTION PLANNING (TAMP)".center(60))
    print("Building Towers with Symbolic Planning".center(60))
    print("="*60)
    
    # Parse arguments
    use_gpu = "gpu" in sys.argv
    use_goal3_extended = "--goal3-ext" in sys.argv
    use_goal4a = "--goal4a" in sys.argv or "--tower-grid" in sys.argv
    use_goal4a_simple = "--goal4a-simple" in sys.argv or "--test" in sys.argv
    use_goal4b = "--goal4b" in sys.argv or "--adjacent" in sys.argv
    
    # Configure mode
    if use_goal3_extended:
        print("\n[MODE] Goal 3 Extended (10+ blocks) - Extreme Difficulty")
        import abstraction, planning
        abstraction.set_goal_mode('goal3_extended')
        planning.set_planning_mode('goal3_extended')
        max_iterations = 30
        planning_timeout = 60  # Longer timeout for complex goals
    elif use_goal4a or use_goal4b:
        print("\n[MODE] Goal 4 - Spatial Grid Structures")
        max_iterations = 15
        planning_timeout = 90  # Much longer timeout for 12-block planning
    else:
        print("\n[MODE] Goal 3 (6-block tower)")
        max_iterations = 10
        planning_timeout = 30  # Default timeout
    
    # Initialize Genesis
    print("\n[INIT] Initializing Genesis simulator...")
    if use_gpu:
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)
    
    # Create scene
    print("\n[INIT] Creating scene...")
    if use_goal4a:
        scene, franka, blocks_state = create_scene_12_yellow_blocks()
        goal_name_str = "tower_grid"
        goal_description = "Goal 4A: Tower Grid (12 yellow)"
        use_spatial = True
    elif use_goal4a_simple:
        # Use regular 6-block scene but only use 4 blocks
        scene, franka, blocks_state = create_scene_6blocks()
        # Rename blocks to y1-y4
        blocks_renamed = {
            "y1": blocks_state["y"],
            "y2": blocks_state["m"],
            "y3": blocks_state["c"],
            "y4": blocks_state["r"],
        }
        blocks_state = blocks_renamed
        goal_name_str = "tower_grid_simple"
        goal_description = "Goal 4A-Simple: 2 Towers (4 blocks TEST)"
        use_spatial = True
    elif use_goal4b:
        scene, franka, blocks_state = create_scene_3red_3green()
        goal_name_str = "adjacent"
        goal_description = "Goal 4B: Adjacent (3R+3G)"
        use_spatial = True
    else:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "six_tower"
        goal_description = "Goal 3: Six-Block Tower"
        use_spatial = False
    
    print(f"  Created {len(blocks_state)} blocks: {', '.join(sorted(blocks_state.keys()))}")
    
    # Set control gains
    print("\n[INIT] Setting robot control gains...")
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
    )
    
    # Select goal
    print("\n[INIT] Selecting goal...")
    goal = get_goal(goal_name_str)
    print(f"  Goal: {goal_description}")
    visualize_predicates(goal, "Goal Configuration")
    
    if use_spatial and "spatial" in goal:
        print(f"\n  Spatial constraints: {len(goal['spatial'])} blocks")
    
    # Run TAMP
    success = tamp_loop(
        scene=scene,
        robot=franka,
        blocks_state=blocks_state,
        goal_predicates=goal,
        domain_file="blocksworld_domain.pddl",
        max_iterations=max_iterations,
        use_spatial=use_spatial,
        planning_timeout=planning_timeout
    )
    
    # Report
    print("\n" + "="*60)
    if success:
        print(f"✓ SUCCESS! {goal_description}".center(60))
    else:
        print("✗ FAILED. Goal not achieved.".center(60))
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
