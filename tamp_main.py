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
    visualize_predicates,
    check_tower_stable,
    get_all_towers
)
from goals import GOAL_TWO_TOWERS, get_goal
from task_planner import call_planner, parse_plan_output, validate_plan
from planning import PlannerInterface
from scenes import create_scene_6blocks


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
                print("[ERROR] put-down requires 1 argument")
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
            
            # Place on table at reasonable location
            # Use current position but on table height
            pos = held_obj.get_pos()
            target_pos = np.array([pos[0], pos[1], 0.02])
            
            success = planner.put_down(target_pos)
            
            if success:
                print(f" Successfully put down {block_name or 'block'}")
            else:
                print(f" Failed to put down {block_name or 'block'}")
            
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


def tamp_loop(scene, robot, blocks_state, 
              goal_predicates=GOAL_TWO_TOWERS,
              domain_file="blocksworld_domain.pddl",
              max_iterations=10):
    """
    Full TAMP execution loop with replanning.

    Steps:
      1. Symbolic abstraction (compute predicates)
      2. Check if goal achieved
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
    
    Returns:
        bool: True if goal achieved, False if failed
    """
    print("\n" + "="*60)
    print("STARTING TAMP LOOP".center(60))
    print("="*60)
    
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
        # STEP 2: CHECK GOAL
        # =====================================================================
        print("\n[STEP 2] Checking goal...")
        if goal_achieved(current_state, goal_predicates):
            print("\n" + "="*60)
            print("✓ GOAL ACHIEVED!".center(60))
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
        
        # Call planner with timeout
        plan = call_planner(domain_file, problem_file, 
                           use_pyperplan=True, 
                           timeout=30)
        
        if not plan:
            print("\n✗ No plan found. Unable to achieve goal.")
            return False

        # Parse and validate plan
        plan = parse_plan_output(plan)
        
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
        for _ in range(300):
            scene.step()
        
        # Check final state
        final_state = compute_predicates(robot, blocks_state, scene)
        
        if goal_achieved(final_state, goal_predicates):
            print("\n" + "="*60)
            print("GOAL ACHIEVED AFTER EXECUTION!".center(60))
            print("="*60)
            visualize_predicates(final_state, "Final State")
            return True
        else:
            print("\n⚠ Goal not achieved after execution. Replanning...")

    # Max iterations reached
    print("\n" + "="*60)
    print("MAXIMUM ITERATIONS REACHED".center(60))
    print("="*60)
    print(f"Failed to achieve goal after {max_iterations} iterations")
    return False


def main():
    """Main entry point with proper Genesis initialization."""
    
    print("\n" + "="*60)
    print("PROJECT 5: BUILDING THE TWO TOWERS".center(60))
    print("Task and Motion Planning (TAMP)".center(60))
    print("="*60)
    
    # =========================================================================
    # STEP 1: INITIALIZE GENESIS
    # =========================================================================
    print("\n[INIT] Initializing Genesis simulator...")
    
    # Check for GPU flag
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        print("  Using GPU backend")
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        print("  Using CPU backend")
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)
    
    # =========================================================================
    # STEP 2: CREATE SCENE
    # =========================================================================
    print("\n[INIT] Creating scene...")
    scene, franka, blocks_state = create_scene_6blocks()
    print(f"  Scene created with {len(blocks_state)} blocks")
    print(f"  Blocks: {', '.join(sorted(blocks_state.keys()))}")
    
    # =========================================================================
    # STEP 3: SET CONTROL GAINS (CRITICAL!)
    # =========================================================================
    print("\n[INIT] Setting robot control gains...")
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    print("  ✓ Control gains configured")
    
    # =========================================================================
    # STEP 4: SELECT GOAL
    # =========================================================================
    print("\n[INIT] Selecting goal...")
    
    # You can change the goal here
    goal = get_goal("five_tower")  # Import get_goal from goals
    goal_name = "Five-Block Tower (MYBRG)"
    
    print(f"  Goal: {goal_name}")
    visualize_predicates(goal, "Goal Configuration")
    
    # =========================================================================
    # STEP 5: RUN TAMP LOOP
    # =========================================================================
    success = tamp_loop(
        scene=scene,
        robot=franka,
        blocks_state=blocks_state,
        goal_predicates=goal,
        domain_file="blocksworld_domain.pddl",
        max_iterations=20
    )
    
    # =========================================================================
    # STEP 6: FINAL REPORT
    # =========================================================================
    print("\n" + "="*60)
    if success:
        print("SUCCESS! Goal achieved!".center(60))
    else:
        print("FAILED. Goal not achieved.".center(60))
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
