# task_planner.py
"""
task_planner.py
-----------------------------
Task planner wrapper for Project 5 (TAMP).
- Primary strategy: use pyperplan if available.
- Fallback: a built-in simple STRIPS-like blocksworld planner.

IMPROVEMENTS OVER ORIGINAL:
  • Better pyperplan integration (supports multiple versions)
  • Input validation (checks files exist)
  • Timeout protection for fallback planner
  • Better logging and error messages
  • Plan validation utility
  • Action name normalization

API:
    create_pddl_problem_file(current_state, goal_state, filename="problem.pddl")
    call_planner(domain_file, problem_file, timeout=30)
    parse_plan_output(plan)
    validate_plan(plan, initial_state, goal_state)

Expectations for predicate dicts (same shape used in abstraction.py):
    {
      "on": [("r","g"), ...],
      "ontable": ["b","c", ...],
      "clear": ["r","b", ...],
      "holding": ["x"] or [],
      "handempty": [True] or []
    }

Return plan format:
    list of tuples, e.g. [('pick-up','r'), ('stack','r','g'), ('put-down','x')]

Author: LA, JHD, JEN
Date: 11/10/2025
"""

import os
import subprocess
import shlex
import tempfile
import time
from typing import List, Tuple, Dict, Optional
import copy


# =============================================================================
# CONFIGURATION
# =============================================================================

VERBOSE = True  # Set to False to reduce logging
FALLBACK_TIMEOUT = 30  # seconds - max time for fallback planner
MAX_FALLBACK_PLAN_LENGTH = 100  # max actions in fallback plan


# =============================================================================
# 1) CREATE PDDL PROBLEM FILE
# =============================================================================

def create_pddl_problem_file(current_state: Dict, goal_state: Dict, 
                             filename: str = "problem.pddl",
                             domain_name: str = "blocksworld",
                             problem_name: str = "tamp-problem") -> str:
    """
    Write a PDDL problem file given current symbolic state and a goal specification.

    Args:
        current_state: dict with keys 'on', 'ontable', 'clear', 'holding', 'handempty'
        goal_state: dict same shape but specifying goal predicates
        filename: path to write problem file
        domain_name: domain name used in PDDL (:domain ...)
        problem_name: problem instance name

    Returns:
        filename written

    Raises:
        ValueError: if states are invalid or missing required keys
    """
    # Validate inputs
    if not isinstance(current_state, dict):
        raise ValueError("current_state must be a dictionary")
    if not isinstance(goal_state, dict):
        raise ValueError("goal_state must be a dictionary")
    
    # Collect object names from both current and goal states
    objs = set()
    
    # From current state
    for (a, b) in current_state.get("on", []):
        objs.add(a)
        objs.add(b)
    for a in current_state.get("ontable", []):
        objs.add(a)
    for a in current_state.get("clear", []):
        objs.add(a)
    for a in current_state.get("holding", []):
        objs.add(a)
    
    # From goal state
    for (a, b) in goal_state.get("on", []):
        objs.add(a)
        objs.add(b)
    for a in goal_state.get("ontable", []):
        objs.add(a)
    for a in goal_state.get("clear", []):
        objs.add(a)
    
    if not objs:
        raise ValueError("No objects found in current_state or goal_state")

    # Write PDDL problem file
    with open(filename, "w") as f:
        f.write(f"(define (problem {problem_name})\n")
        f.write(f"  (:domain {domain_name})\n\n")
        
        # Objects declaration
        f.write("  (:objects\n")
        f.write("    " + " ".join(sorted(objs)) + " - block\n")
        f.write("  )\n\n")
        
        # Initial state
        f.write("  (:init\n")
        for (a, b) in current_state.get("on", []):
            f.write(f"    (on {a} {b})\n")
        for a in current_state.get("ontable", []):
            f.write(f"    (ontable {a})\n")
        for a in current_state.get("clear", []):
            f.write(f"    (clear {a})\n")
        for a in current_state.get("holding", []):
            f.write(f"    (holding {a})\n")
        # handempty is a boolean flag
        if current_state.get("handempty"):
            f.write("    (handempty)\n")
        f.write("  )\n\n")

        # Goal state
        f.write("  (:goal (and\n")
        for (a, b) in goal_state.get("on", []):
            f.write(f"    (on {a} {b})\n")
        for a in goal_state.get("ontable", []):
            f.write(f"    (ontable {a})\n")
        for a in goal_state.get("clear", []):
            f.write(f"    (clear {a})\n")
        f.write("  ))\n")
        f.write(")\n")

    if VERBOSE:
        print(f"[task_planner] PDDL problem file written to {os.path.abspath(filename)}")
    
    return filename


# =============================================================================
# 2) CALL EXTERNAL PLANNER (PYPERPLAN) OR FALLBACK
# =============================================================================

def call_planner(domain_file: str, problem_file: str, 
                 use_pyperplan: bool = True,
                 timeout: int = 30) -> Optional[List[Tuple]]:
    """
    Call the task planner and return a list of actions (tuples).
    
    Strategy:
        1. Try pyperplan library API (fastest, most reliable)
        2. Try pyperplan CLI (if library fails)
        3. Use lightweight fallback planner (if pyperplan unavailable)

    Args:
        domain_file: path to PDDL domain file
        problem_file: path to PDDL problem file
        use_pyperplan: whether to attempt pyperplan (True = try it)
        timeout: maximum time for fallback planner in seconds

    Returns:
        list of action tuples, e.g. [('pick-up','r'), ('stack','r','g'), ...]
        or None if no plan found

    Raises:
        FileNotFoundError: if domain or problem file doesn't exist
    """
    # Validate input files exist
    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain file not found: {domain_file}")
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    plan = None
    
    # -------------------------------------------------------------------------
    # Strategy 1: Try pyperplan library API
    # -------------------------------------------------------------------------
    if use_pyperplan:
        try:
            if VERBOSE:
                print("[task_planner] Attempting to use pyperplan library...")
            
            # Try importing and using pyperplan directly
            from pyperplan import planner as pyplanner
            from pyperplan.pddl.parser import Parser
            
            # Parse domain and problem
            parser = Parser(domain_file, problem_file)
            domain = parser.parse_domain()
            problem = parser.parse_problem(domain)
            
            # Run planner (pyperplan uses FF-like search by default)
            solution = pyplanner.search(problem)
            
            if solution:
                plan = _normalize_pyperplan_output(solution)
                if VERBOSE:
                    print(f"[task_planner] ✓ Pyperplan found plan with {len(plan)} actions")
                return plan
            else:
                if VERBOSE:
                    print("[task_planner] Pyperplan returned no solution")
        
        except ImportError:
            if VERBOSE:
                print("[task_planner] Pyperplan not installed, trying CLI...")
        except Exception as e:
            if VERBOSE:
                print(f"[task_planner] Pyperplan library failed: {e}")
    
    # -------------------------------------------------------------------------
    # Strategy 2: Try pyperplan CLI
    # -------------------------------------------------------------------------
    if use_pyperplan and plan is None:
        try:
            if VERBOSE:
                print("[task_planner] Attempting pyperplan CLI...")
            
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".plan") as tmpf:
                tmp_plan_file = tmpf.name
            
            # Try different CLI invocations (pyperplan versions differ)
            cli_commands = [
                # Standard pyperplan CLI
                f"pyperplan {shlex.quote(domain_file)} {shlex.quote(problem_file)}",
                # With explicit output flag
                f"pyperplan {shlex.quote(domain_file)} {shlex.quote(problem_file)} -o {shlex.quote(tmp_plan_file)}",
                # Python module invocation
                f"python -m pyperplan {shlex.quote(domain_file)} {shlex.quote(problem_file)}",
            ]
            
            for cmd in cli_commands:
                try:
                    result = subprocess.run(
                        shlex.split(cmd), 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        timeout=timeout
                    )
                    
                    # Try to read plan from temp file
                    plan = _read_plan_file(tmp_plan_file)
                    if plan:
                        if VERBOSE:
                            print(f"[task_planner] ✓ Pyperplan CLI found plan with {len(plan)} actions")
                        os.remove(tmp_plan_file)
                        return plan
                    
                    # Also try parsing stdout
                    stdout = result.stdout.decode('utf-8')
                    if stdout and '(' in stdout:
                        plan = _parse_plan_from_text(stdout)
                        if plan:
                            if VERBOSE:
                                print(f"[task_planner] ✓ Parsed plan from stdout with {len(plan)} actions")
                            os.remove(tmp_plan_file)
                            return plan
                
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    continue  # Try next command
            
            # Clean up temp file
            if os.path.exists(tmp_plan_file):
                os.remove(tmp_plan_file)
        
        except Exception as e:
            if VERBOSE:
                print(f"[task_planner] Pyperplan CLI failed: {e}")
    
    # -------------------------------------------------------------------------
    # Strategy 3: Use fallback planner
    # -------------------------------------------------------------------------
    if plan is None:
        if VERBOSE:
            print("[task_planner] Using fallback blocksworld planner...")
        
        try:
            current_state = _parse_problem_file_init(problem_file)
            goal_state = _parse_problem_file_goal(problem_file)
            plan = _fallback_blocksworld_planner(current_state, goal_state, timeout=timeout)
            
            if plan:
                if VERBOSE:
                    print(f"[task_planner] ✓ Fallback planner found plan with {len(plan)} actions")
            else:
                if VERBOSE:
                    print("[task_planner] ✗ Fallback planner found no solution")
        
        except Exception as e:
            if VERBOSE:
                print(f"[task_planner] ✗ Fallback planner failed: {e}")
            plan = None

    return plan


# =============================================================================
# 3) PARSE/NORMALIZE PLAN OUTPUTS
# =============================================================================

def parse_plan_output(plan: List) -> List[Tuple]:
    """
    Accept different plan formats and return uniform list of action tuples.

    Examples:
        input: ["(pick-up r)", "(stack r g)"]  
        output: [('pick-up','r'), ('stack','r','g')]
        
        input: [('pick-up','r'), ('stack','r','g')] 
        output: [('pick-up','r'), ('stack','r','g')]

    Args:
        plan: Plan in various formats (list of strings, tuples, etc.)

    Returns:
        Normalized list of action tuples
    """
    if plan is None:
        return []
    
    normalized = []
    for step in plan:
        if isinstance(step, tuple) or isinstance(step, list):
            # Already in tuple/list format
            normalized.append(tuple(step))
        elif isinstance(step, str):
            # Parse string format: "(pick-up r)" or "pick-up r"
            s = step.strip()
            # Remove surrounding parentheses if present
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1].strip()
            # Split on whitespace and commas
            parts = s.replace(",", " ").split()
            if not parts:
                continue
            action = parts[0].lower()
            args = [p.strip() for p in parts[1:]]
            normalized.append(tuple([action] + args))
        else:
            # Unknown format - try converting to string
            s = str(step)
            normalized.extend(parse_plan_output([s]))
    
    return normalized


def normalize_action_names(plan: List[Tuple], 
                           to_format: str = "hyphen") -> List[Tuple]:
    """
    Normalize action names to consistent format.
    
    Args:
        plan: List of action tuples
        to_format: "hyphen" for pick-up style, "underscore" for pick_up style
    
    Returns:
        Plan with normalized action names
    """
    normalized = []
    for action_tuple in plan:
        if not action_tuple:
            continue
        action = action_tuple[0]
        args = action_tuple[1:]
        
        if to_format == "hyphen":
            action = action.replace("_", "-")
        elif to_format == "underscore":
            action = action.replace("-", "_")
        
        normalized.append(tuple([action] + list(args)))
    
    return normalized


# =============================================================================
# 4) PLAN VALIDATION
# =============================================================================

def validate_plan(plan: List[Tuple], 
                  initial_state: Dict, 
                  goal_state: Dict) -> Tuple[bool, str]:
    """
    Validate that a plan is well-formed and achieves the goal.
    
    Performs simulation to check:
    1. All actions are applicable (preconditions satisfied)
    2. Final state satisfies goal predicates
    3. No invalid states reached during execution
    
    Args:
        plan: List of action tuples
        initial_state: Initial predicate dictionary
        goal_state: Goal predicate dictionary
    
    Returns:
        (is_valid, error_message)
        - is_valid: True if plan is valid
        - error_message: Empty string if valid, error description if invalid
    """
    if not plan:
        return False, "Plan is empty"
    
    # Simulate plan execution
    state = copy.deepcopy(initial_state)
    
    for i, action_tuple in enumerate(plan):
        if not action_tuple:
            return False, f"Empty action at step {i}"
        
        action = action_tuple[0]
        args = action_tuple[1:]
        
        # Apply action and check if it's valid
        new_state, error = _apply_action(state, action, args)
        if error:
            return False, f"Step {i} ({action} {' '.join(args)}): {error}"
        
        state = new_state
    
    # Check if final state satisfies goal
    for (a, b) in goal_state.get("on", []):
        if (a, b) not in state.get("on", []):
            return False, f"Goal not achieved: missing (on {a} {b})"
    
    for a in goal_state.get("ontable", []):
        if a not in state.get("ontable", []):
            return False, f"Goal not achieved: missing (ontable {a})"
    
    for a in goal_state.get("clear", []):
        if a not in state.get("clear", []):
            return False, f"Goal not achieved: missing (clear {a})"
    
    return True, ""


def _apply_action(state: Dict, action: str, args: List[str]) -> Tuple[Dict, str]:
    """
    Apply an action to a state and return new state or error.
    
    Returns:
        (new_state, error_message)
        - new_state: Updated state dict (or original if error)
        - error_message: Empty if success, error description if failed
    """
    new_state = copy.deepcopy(state)
    
    action = action.lower().replace("_", "-")  # Normalize
    
    if action == "pick-up":
        if len(args) < 1:
            return state, "pick-up requires 1 argument"
        x = args[0]
        
        # Check preconditions
        if not new_state.get("handempty"):
            return state, "hand not empty"
        if x not in new_state.get("clear", []):
            return state, f"{x} not clear"
        if x not in new_state.get("ontable", []):
            return state, f"{x} not on table"
        
        # Apply effects
        new_state["holding"] = [x]
        new_state["handempty"] = []
        new_state["ontable"].remove(x)
        new_state["clear"].remove(x)
        
        return new_state, ""
    
    elif action == "put-down":
        if len(args) < 1:
            return state, "put-down requires 1 argument"
        x = args[0]
        
        # Check preconditions
        if x not in new_state.get("holding", []):
            return state, f"not holding {x}"
        
        # Apply effects
        new_state["holding"] = []
        new_state["handempty"] = [True]
        new_state["ontable"].append(x)
        new_state["clear"].append(x)
        
        return new_state, ""
    
    elif action == "stack":
        if len(args) < 2:
            return state, "stack requires 2 arguments"
        x, y = args[0], args[1]
        
        # Check preconditions
        if x not in new_state.get("holding", []):
            return state, f"not holding {x}"
        if y not in new_state.get("clear", []):
            return state, f"{y} not clear"
        
        # Apply effects
        new_state["holding"] = []
        new_state["handempty"] = [True]
        new_state["on"].append((x, y))
        new_state["clear"].append(x)
        new_state["clear"].remove(y)
        
        return new_state, ""
    
    elif action == "unstack":
        if len(args) < 2:
            return state, "unstack requires 2 arguments"
        x, y = args[0], args[1]
        
        # Check preconditions
        if not new_state.get("handempty"):
            return state, "hand not empty"
        if (x, y) not in new_state.get("on", []):
            return state, f"{x} not on {y}"
        if x not in new_state.get("clear", []):
            return state, f"{x} not clear"
        
        # Apply effects
        new_state["holding"] = [x]
        new_state["handempty"] = []
        new_state["on"].remove((x, y))
        new_state["clear"].remove(x)
        new_state["clear"].append(y)
        
        return new_state, ""
    
    else:
        return state, f"unknown action: {action}"


# =============================================================================
# HELPER: READ PLAN FILE GENERATED BY PYPERPLAN CLI
# =============================================================================

def _read_plan_file(path: str) -> List[Tuple]:
    """Read plan from file generated by pyperplan CLI."""
    if not os.path.exists(path):
        return []
    
    steps = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith(";") or ln.startswith("#"):
                continue
            # Parse action line
            steps.append(ln)
    
    return parse_plan_output(steps)


def _parse_plan_from_text(text: str) -> List[Tuple]:
    """Parse plan from text output (stdout)."""
    lines = text.split('\n')
    plan_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            continue
        if '(' in line and ')' in line:
            plan_lines.append(line)
    
    return parse_plan_output(plan_lines)


def _normalize_pyperplan_output(plan_obj) -> List[Tuple]:
    """
    Convert pyperplan's internal plan representation to our tuple format.
    
    Pyperplan returns different formats depending on version:
    - List of Action objects
    - List of tuples
    - List of strings
    """
    if plan_obj is None:
        return []
    
    normalized = []
    
    for step in plan_obj:
        # Try to extract action name and arguments
        if hasattr(step, 'name') and hasattr(step, 'sig'):
            # Action object format
            action = step.name.lower()
            args = [str(arg) for arg in step.sig]
            normalized.append(tuple([action] + args))
        elif isinstance(step, tuple) or isinstance(step, list):
            # Already tuple format
            normalized.append(tuple(step))
        elif isinstance(step, str):
            # String format
            normalized.extend(parse_plan_output([step]))
        else:
            # Try str() conversion
            normalized.extend(parse_plan_output([str(step)]))
    
    return normalized


# =============================================================================
# FALLBACK PLANNER: SIMPLE BLOCKSWORLD SOLVER
# =============================================================================

def _fallback_blocksworld_planner(init: Dict, goal: Dict, 
                                  timeout: int = 30) -> Optional[List[Tuple]]:
    """
    Simple goal-regression planner for blocksworld.
    
    Uses basic means-ends analysis:
    1. Process ON goals first (bottom-up tower building)
    2. Process ONTABLE goals
    3. Process CLEAR goals
    
    Args:
        init: Initial state dictionary
        goal: Goal state dictionary
        timeout: Maximum planning time in seconds
    
    Returns:
        List of action tuples, or None if no plan found
    """
    start_time = time.time()
    state = copy.deepcopy(init)
    plan = []
    
    def is_timeout():
        return (time.time() - start_time) > timeout
    
    def is_on(x, y):
        return (x, y) in state.get("on", [])
    
    def is_ontable(x):
        return x in state.get("ontable", [])
    
    def is_clear(x):
        return x in state.get("clear", [])
    
    def holding_obj():
        h = state.get("holding", [])
        return h[0] if h else None
    
    def sim_apply(action_tuple):
        """Simulate action without modifying state."""
        action = action_tuple[0]
        args = action_tuple[1:]
        _, error = _apply_action(state, action, args)
        if error:
            return False
        # Actually apply if no error
        new_state, _ = _apply_action(state, action, args)
        state.clear()
        state.update(new_state)
        return True
    
    def achieve(predicate):
        """Recursively achieve a goal predicate."""
        if is_timeout():
            return False
        if len(plan) > MAX_FALLBACK_PLAN_LENGTH:
            return False
        
        typ = predicate[0]
        
        if typ == "on":
            x, y = predicate[1], predicate[2]
            if is_on(x, y):
                return True
            
            # Need to: hold x, then stack x onto y
            # First ensure x is ready to be picked up
            if not is_clear(x):
                # Make x clear
                if not achieve(("clear", x)):
                    return False
            
            # Ensure y is clear
            if not is_clear(y):
                if not achieve(("clear", y)):
                    return False
            
            # Pick up x if not holding
            if holding_obj() != x:
                if state.get("handempty"):
                    # Try pick-up from table
                    if is_ontable(x):
                        if not sim_apply(("pick-up", x)):
                            return False
                        plan.append(("pick-up", x))
                    else:
                        # x is on something, unstack it
                        below = None
                        for (a, b) in state.get("on", []):
                            if a == x:
                                below = b
                                break
                        if below is None:
                            return False
                        if not sim_apply(("unstack", x, below)):
                            return False
                        plan.append(("unstack", x, below))
                else:
                    # Put down what we're holding
                    cur = holding_obj()
                    if cur:
                        if not sim_apply(("put-down", cur)):
                            return False
                        plan.append(("put-down", cur))
                    # Now pick up x
                    if is_ontable(x):
                        if not sim_apply(("pick-up", x)):
                            return False
                        plan.append(("pick-up", x))
                    else:
                        below = None
                        for (a, b) in state.get("on", []):
                            if a == x:
                                below = b
                                break
                        if below is None:
                            return False
                        if not sim_apply(("unstack", x, below)):
                            return False
                        plan.append(("unstack", x, below))
            
            # Now stack x onto y
            if not sim_apply(("stack", x, y)):
                return False
            plan.append(("stack", x, y))
            return True
        
        elif typ == "ontable":
            x = predicate[1]
            if is_ontable(x):
                return True
            # Unstack x and put it down
            below = None
            for (a, b) in state.get("on", []):
                if a == x:
                    below = b
                    break
            if below is None:
                return False
            if not sim_apply(("unstack", x, below)):
                return False
            plan.append(("unstack", x, below))
            if not sim_apply(("put-down", x)):
                return False
            plan.append(("put-down", x))
            return True
        
        elif typ == "clear":
            x = predicate[1]
            if is_clear(x):
                return True
            # Remove blocks on top of x
            for (a, b) in state.get("on", []):
                if b == x:
                    # Unstack a from x
                    if not achieve(("ontable", a)):
                        return False
            return True
        
        else:
            return False
    
    # Process goals in order: ON, ONTABLE, CLEAR
    goal_list = []
    for (a, b) in goal.get("on", []):
        goal_list.append(("on", a, b))
    for a in goal.get("ontable", []):
        goal_list.append(("ontable", a))
    for a in goal.get("clear", []):
        goal_list.append(("clear", a))
    
    for g in goal_list:
        if is_timeout():
            if VERBOSE:
                print(f"[task_planner] Fallback planner timed out after {timeout}s")
            return None
        if not achieve(g):
            return None
    
    return plan


# =============================================================================
# HELPER FUNCTIONS: PARSE PROBLEM FILE (FOR FALLBACK)
# =============================================================================

def _parse_problem_file_init(problem_file: str) -> Dict:
    """
    Parse PDDL problem file to extract initial state.
    Simple/tolerant parser for standard blocksworld format.
    """
    init = {"on": [], "ontable": [], "clear": [], "holding": [], "handempty": []}
    
    with open(problem_file, "r") as f:
        content = f.read()
    
    if "(:init" not in content:
        return init
    
    # Extract init block
    init_block = content.split("(:init", 1)[1].split(")", 1)[0]
    lines = init_block.replace("\n", " ").split("(")
    
    for token in lines:
        tok = token.strip().replace(")", "").strip()
        if not tok:
            continue
        parts = tok.split()
        if not parts:
            continue
        
        pred = parts[0].lower()
        args = parts[1:]
        
        if pred == "on" and len(args) >= 2:
            init["on"].append((args[0], args[1]))
        elif pred == "ontable" and len(args) >= 1:
            init["ontable"].append(args[0])
        elif pred == "clear" and len(args) >= 1:
            init["clear"].append(args[0])
        elif pred == "holding" and len(args) >= 1:
            init["holding"].append(args[0])
        elif pred == "handempty":
            init["handempty"] = [True]
    
    return init


def _parse_problem_file_goal(problem_file: str) -> Dict:
    """
    Parse PDDL problem file to extract goal state.
    """
    goal = {"on": [], "ontable": [], "clear": []}
    
    with open(problem_file, "r") as f:
        content = f.read()
    
    if "(:goal" not in content:
        return goal
    
    # Extract goal block
    goal_block = content.split("(:goal", 1)[1]
    if "(and" in goal_block:
        goal_block = goal_block.split("(and", 1)[1]
    goal_block = goal_block.split(")", 1)[0]
    
    lines = goal_block.replace("\n", " ").split("(")
    
    for token in lines:
        tok = token.strip().replace(")", "").strip()
        if not tok:
            continue
        parts = tok.split()
        if not parts:
            continue
        
        pred = parts[0].lower()
        args = parts[1:]
        
        if pred == "on" and len(args) >= 2:
            goal["on"].append((args[0], args[1]))
        elif pred == "ontable" and len(args) >= 1:
            goal["ontable"].append(args[0])
        elif pred == "clear" and len(args) >= 1:
            goal["clear"].append(args[0])
    
    return goal


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Task Planner Demo")
    print("=" * 60)
    
    # Simple demo: stack r on g on b
    initial = {
        "on": [],
        "ontable": ["r", "g", "b"],
        "clear": ["r", "g", "b"],
        "holding": [],
        "handempty": [True]
    }
    
    goal = {
        "on": [("r", "g"), ("g", "b")],
        "ontable": ["b"],
        "clear": ["r"]
    }
    
    # Create problem file
    tmp_problem = "demo_problem.pddl"
    create_pddl_problem_file(initial, goal, tmp_problem)
    print(f"\n✓ Created problem file: {tmp_problem}")
    
    # Try planning (will use fallback since domain file may not exist)
    print("\n" + "=" * 60)
    print("Running planner...")
    print("=" * 60)
    
    plan = call_planner("blocksworld_domain.pddl", tmp_problem, 
                       use_pyperplan=False)  # Use fallback for demo
    
    if plan:
        print(f"\n✓ Plan found with {len(plan)} actions:")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action[0]} {' '.join(action[1:])}")
        
        # Validate plan
        print("\n" + "=" * 60)
        print("Validating plan...")
        print("=" * 60)
        is_valid, error = validate_plan(plan, initial, goal)
        if is_valid:
            print("✓ Plan is valid!")
        else:
            print(f"✗ Plan validation failed: {error}")
    else:
        print("\n✗ No plan found")
    
    # Clean up
    if os.path.exists(tmp_problem):
        os.remove(tmp_problem)