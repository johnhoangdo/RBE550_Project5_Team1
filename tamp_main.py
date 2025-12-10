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
# for CLI flags and utils
import time

# make sure these modules are imported somewhere near the top
import abstraction
import task_planner
import planning
import goals

# if you put scenes in a module named scenes.py:
from scenes import create_scene_10blocks   # add this (create_scene_10blocks() must exist in scenes.py)


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

# ----------------------------- Tallest Tower Helper -----------------------------
def attempt_tallest_tower(franka, scene, blocks_state, starting_order, extras, min_blocks=8):
    """
    Incrementally try to build tallest tower using starting_order (bottom->top)
    and extras (list) to add on top. Tries orientation schedules (0 or 45 deg).
    Returns (best_height, best_goal)
    """
    import time
    from abstraction import compute_predicates, generate_pddl_problem  # adapt names if different
    from task_planner import call_planner, parse_plan_output, create_pddl_problem_file
    # PlannerInterface is your motion primitive wrapper. If you named it differently, import the correct class.
    try:
        from planning import PlannerInterface
        planner_iface = PlannerInterface(franka, scene)
    except Exception:
        planner_iface = None
        print("[TALLEST] Warning: PlannerInterface not available; attempting to call planning.execute_primitive directly")

    # build all candidate orders (starting + 0..len(extras) extras)
    candidates = []
    for k in range(0, len(extras) + 1):
        order = list(starting_order) + extras[:k]
        if len(order) < min_blocks:
            continue
        candidates.append(order)

    best_height = 0
    best_goal = None

    for candidate in candidates:
        # build goal dict bottom->top
        on_rel = []
        for i in range(1, len(candidate)):
            on_rel.append((candidate[i], candidate[i-1]))
        goal = {"on": on_rel, "ontable": [candidate[0]], "clear": [candidate[-1]]}

        print(f"[TALLEST] Trying candidate height {len(candidate)}: {candidate}")

        # 1) Get symbolic predicates from abstraction (adapt compute_predicates signature if needed)
        try:
            preds = abstraction.compute_predicates()
        except Exception:
            # if your compute_predicates needs args, call with (franka, blocks_state, scene)
            try:
                preds = abstraction.compute_predicates(franka, blocks_state, scene)
            except Exception as e:
                print("[TALLEST] Failed to compute predicates:", e)
                continue

        # 2) Create a PDDL problem file for this goal (adapt function name if different)
        try:
            # Some projects call it create_pddl_problem_file; others generate_pddl_problem
            try:
                problem_file = task_planner.create_pddl_problem_file(preds, goal)
            except Exception:
                problem_file = task_planner.generate_pddl_problem(preds, goal, filename="tallest_problem.pddl")
        except Exception as e:
            print("[TALLEST] Failed to generate PDDL problem:", e)
            continue

        # 3) Call symbolic planner
        try:
            plan_result = task_planner.call_planner(problem_file, domain_file="blocksworld_domain.pddl", timeout=30)
        except Exception:
            try:
                plan_result = task_planner.call_planner(problem_file, timeout=30)
            except Exception as e:
                print("[TALLEST] call_planner failed:", e)
                plan_result = None

        if not plan_result:
            print(f"[TALLEST] planner returned no plan for {len(candidate)} blocks")
            continue

        # 4) Parse plan output into action list
        try:
            actions = task_planner.parse_plan_output(plan_result)
        except Exception:
            # If parser returns raw text, split lines into simple tokens
            if isinstance(plan_result, str):
                actions = [line.strip() for line in plan_result.splitlines() if line.strip()]
            else:
                actions = plan_result

        # 5) Define orientation schedules to try
        schedules = [
            {b: 0.0 for b in candidate},
            {b: (45.0 if i % 2 == 0 else 0.0) for i, b in enumerate(candidate)},
            {b: 45.0 for b in candidate}
        ]

        success_for_candidate = False

        for schedule in schedules:
            print(f"[TALLEST] Trying orientation schedule: {schedule}")
            # Execute actions using planner_iface where possible, else call planning.execute_primitive()
            all_ok = True
            for act in actions:
                # Normalize action representation to a tuple or string tokens
                # Typical string form: "STACK R G" or "pick-up r"
                if isinstance(act, str):
                    tokens = act.replace("(", " ").replace(")", " ").replace(",", " ").split()
                    tokens = [t.lower() for t in tokens if t.strip()]
                    if not tokens:
                        continue
                    op = tokens[0]
                    args = tokens[1:]
                elif isinstance(act, (list, tuple)):
                    op = str(act[0]).lower()
                    args = [str(a).lower() for a in act[1:]]
                else:
                    print("[TALLEST] Unknown action format:", act)
                    all_ok = False
                    break

                orientation = None
                # for stack operations, args often are [top, bottom] -> orientation applies to top (placement)
                if op in ("stack", "stack-on", "stackon") and len(args) >= 2:
                    top_block = args[0]
                    orientation = schedule.get(top_block, None)

                # for put-down / place operations: orientation on the held block (args[0] or none)
                if op in ("put-down", "putdown", "put_down", "place") and len(args) >= 1:
                    held = args[0]
                    orientation = schedule.get(held, None)

                # Execute using PlannerInterface if available
                ok = False
                try:
                    if planner_iface is not None:
                        # Try mapping ops to planner_iface methods
                        if op in ("pick-up", "pickup", "pick"):
                            ok = planner_iface.pick_up(blocks_state[args[0]])
                        elif op in ("put-down", "putdown", "put_down", "place"):
                            # pass orientation hint if method supports it
                            if hasattr(planner_iface.put_down, "__call__"):
                                try:
                                    ok = planner_iface.put_down(orientation_deg=orientation)
                                except TypeError:
                                    ok = planner_iface.put_down()
                            else:
                                ok = True
                        elif op in ("stack", "stack-on", "stackon"):
                            # planner_iface.stack(target_entity, orientation_deg=orientation)
                            bottom = args[1] if len(args) > 1 else None
                            target_entity = blocks_state.get(bottom)
                            if hasattr(planner_iface.stack, "__call__"):
                                try:
                                    ok = planner_iface.stack(target_entity, orientation_deg=orientation)
                                except TypeError:
                                    ok = planner_iface.stack(target_entity)
                            else:
                                ok = True
                        elif op in ("unstack",):
                            ok = planner_iface.pick_up(blocks_state[args[0]])
                        else:
                            # fallback: try calling a generic execute_primitive
                            ok = planning.execute_primitive(" ".join([op] + args), orientation_deg=orientation)
                    else:
                        # No PlannerInterface — call generic execute_primitive
                        ok = planning.execute_primitive(" ".join([op] + args), orientation_deg=orientation)
                except Exception as e:
                    print("[TALLEST] Exception while executing:", e)
                    ok = False

                if not ok:
                    all_ok = False
                    print("[TALLEST] Action failed:", op, args)
                    break

            if all_ok:
                print(f"[TALLEST] Candidate of size {len(candidate)} succeeded with schedule {schedule}")
                best_height = len(candidate)
                best_goal = goal
                success_for_candidate = True
                break

        if not success_for_candidate:
            print(f"[TALLEST] Candidate of size {len(candidate)} failed")
            # Optional: stop trying larger sizes if this failed
            break

    print(f"[TALLEST] Best height built: {best_height}")
    return best_height, best_goal
# -------------------------- end tallest helper --------------------------



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
    # ---------- Goal 3 Extended / Tallest Tower Mode ----------
    if use_goal3_extended or "--tallest" in sys.argv:
        print("\n[MODE] Goal 3 Extended (Tallest Tower attempt — 8+ blocks)")
        # Create the 10-block demo scene (factory must be in scenes.py)
        scene, franka, blocks_state = create_scene_10blocks()

        # define a sensible starting tower (bottom -> top) and extras pool
        starting = ["r", "g", "b", "y", "m"]   # example base 5-block stack
        extras = ["c", "o", "p", "q", "s"]     # extras to reach up to 10 blocks

        # run the tallest-tower routine (enforces min 8 by default)
        best_h, best_goal = attempt_tallest_tower(franka, scene, blocks_state, starting, extras, min_blocks=8)
        print("[MODE] Tallest tower attempt finished. Best height:", best_h)
        print("[MODE] Best goal:", best_goal)

        # exit after attempt (or comment-out to continue into interactive tamp loop)
        sys.exit(0 if best_h >= 8 else 1)



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
    elif use_goal3_extended:
        print("\n[MODE] Goal 3 Extended (Tallest Tower attempt — 8+ blocks)")
        # Create the 10-block demo scene (factory must be in scenes.py)
        scene, franka, blocks_state = create_scene_10blocks()

        # define a sensible starting tower (bottom -> top) and extras pool
        starting = ["r", "g", "b", "y", "m"]   # example base 5-block stack
        extras = ["c", "o", "p", "q", "s"]     # extras to reach up to 10 blocks

        # run the tallest-tower routine (enforces min 8 by default)
        best_h, best_goal = attempt_tallest_tower(franka, scene, blocks_state, starting, extras, min_blocks=8)
        print("[MODE] Tallest tower attempt finished. Best height:", best_h)
        print("[MODE] Best goal:", best_goal)

        # exit after attempt (or remove sys.exit(...) to continue into interactive tamp loop)
        sys.exit(0 if best_h >= 8 else 1)

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
