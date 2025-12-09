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
    """
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
                print(f" Successfully stacked {block_a_name} on {block_b_name}")
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
    if "spatial" not in goal_predicates:
        return plan
    
    spatial_targets = goal_predicates["spatial"]
    positioning_actions = []
    
    print("\n[SPATIAL] Augmenting plan with positioning moves...")
    
    blocks_to_position = set()
    for action in plan:
        if action[0] == "stack" and len(action) >= 3:
            bottom_block = action[2]
            if bottom_block in spatial_targets:
                blocks_to_position.add(bottom_block)
    
    for bottom_block in blocks_to_position:
        target_pos = spatial_targets[bottom_block]
        current_pos = blocks_state[bottom_block].get_pos()
        distance = ((current_pos[0] - target_pos[0])**2 + 
                   (current_pos[1] - target_pos[1])**2)**0.5
        
        if distance > 0.05:
            positioning_actions.append(("pick-up", bottom_block))
            positioning_actions.append(("put-down", bottom_block,
                                       str(target_pos[0]), str(target_pos[1])))
    
    return positioning_actions + plan


def tamp_loop(scene, robot, blocks_state, 
              goal_predicates=GOAL_TWO_TOWERS,
              domain_file="blocksworld_domain.pddl",
              max_iterations=10,
              use_spatial=False,
              planning_timeout=30):

    planner_interface = PlannerInterface(robot, scene)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        current_state = compute_predicates(robot, blocks_state, scene)

        if use_spatial:
            goal_met = goal_achieved_with_spatial(
                current_state, goal_predicates, blocks_state
            )
        else:
            goal_met = goal_achieved(current_state, goal_predicates)
        
        if goal_met:
            print("✓ GOAL ACHIEVED!")
            return True

        goal_pddl = {k: v for k, v in goal_predicates.items() if k != "spatial"}

        problem_file = generate_pddl_problem(
            current_state, goal_pddl, "current_problem.pddl"
        )

        plan = call_planner(domain_file, problem_file,
                           use_pyperplan=True,
                           timeout=planning_timeout)

        if not plan:
            return False

        plan = parse_plan_output(plan)

        if use_spatial and "spatial" in goal_predicates:
            plan = augment_plan_with_positioning(
                plan, current_state, goal_predicates, blocks_state
            )

        for action in plan:
            if not execute_primitive(action, planner_interface, scene, blocks_state):
                break
        
    return False


def main():
    print("\n" + "="*60)
    print("PROJECT 5: TASK AND MOTION PLANNING (TAMP)".center(60))
    print("Building Towers with Symbolic Planning".center(60))
    print("="*60)
    
    use_gpu = "gpu" in sys.argv
    use_goal1 = "--goal1" in sys.argv or "--two-towers" in sys.argv   # ✅ NEW
    use_goal2 = "--goal2" in sys.argv or "--five-tower" in sys.argv   # ✅ NEW
    use_goal3 = "--goal3" in sys.argv or "--six-tower" in sys.argv   # ✅ NEW
    use_goal3_extended = "--goal3-ext" in sys.argv
    use_goal4a = "--goal4a" in sys.argv or "--tower-grid" in sys.argv
    use_goal4a_simple = "--goal4a-simple" in sys.argv or "--test" in sys.argv
    use_goal4b = "--goal4b" in sys.argv or "--adjacent" in sys.argv
    
    # ✅ MODE SELECTION
    if use_goal1:
        print("\n[MODE] Goal 1: Two 3-Block Towers")
        max_iterations = 10
        planning_timeout = 20
    elif use_goal2:   # ✅ NEW
        print("\n[MODE] Goal 2: Five-Block Tower")
        max_iterations = 12
        planning_timeout = 25
    elif use_goal3:   # ✅ NEW
        print("\n[MODE] Goal 3: Single 6-Block Tall Tower")
        max_iterations = 10
        planning_timeout = 30
    elif use_goal3_extended:
        print("\n[MODE] Goal 3 Extended (10+ blocks)")
        max_iterations = 30
        planning_timeout = 60
    elif use_goal4a or use_goal4b:
        print("\n[MODE] Goal 4 - Spatial Grid")
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
    
    # ✅ SCENE SELECTION
    if use_goal1:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "two_towers"
        goal_description = "Goal 1: Two 3-Block Towers"
        use_spatial = False
    elif use_goal2:   # ✅ NEW
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "five_tower"
        goal_description = "Goal 2: Five-Block Tower"
        use_spatial = False
    elif use_goal3:   # ✅ NEW
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "six_tower"
        goal_description = "Goal 3: Single 6-Block Tall Tower"
        use_spatial = False
    elif use_goal4a:
        scene, franka, blocks_state = create_scene_12_yellow_blocks()
        goal_name_str = "tower_grid"
        goal_description = "Goal 4A"
        use_spatial = True
    elif use_goal4b:
        scene, franka, blocks_state = create_scene_3red_3green()
        goal_name_str = "adjacent"
        goal_description = "Goal 4B"
        use_spatial = True
    else:
        scene, franka, blocks_state = create_scene_6blocks()
        goal_name_str = "six_tower"
        goal_description = "Goal 3"
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
    
    print("SUCCESS" if success else "FAILED")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
