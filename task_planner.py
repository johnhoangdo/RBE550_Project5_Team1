# task_planner.py
"""
task_planner.py
-----------------------------
Task planner wrapper for Project 5 (TAMP).
- Primary strategy: use pyperplan if available

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
    Write a PDDL problem file given current symbolic state and a goal specification

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
# 2) CALL EXTERNAL PLANNER (PYPERPLAN)
# =============================================================================

def call_planner(domain_file: str, problem_file: str, 
                 use_pyperplan: bool = True,
                 timeout: int = 30) -> Optional[List[Tuple]]:
    """
    Call the task planner and return a list of actions (tuples)
    
    Strategy:
        1. Try pyperplan library API (fastest, most reliable)
        2. Try pyperplan CLI
        3. Check for .soln file from CLI
        4. Use lightweight fallback planner (if pyperplan unavailable)

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
            
            # Try importing pyperplan
            import pyperplan
            from pyperplan.pddl.parser import Parser
            
            # Parse domain and problem
            parser = Parser(domain_file, problem_file)
            domain = parser.parse_domain()
            problem = parser.parse_problem(domain)
            
            from pyperplan import search as pyperplan_search
            solution = pyperplan_search.breadth_first_search(problem)
                      
            if solution:
                plan = _normalize_pyperplan_output(solution)
                if VERBOSE:
                    print(f"[task_planner] Pyperplan found plan with {len(plan)} actions")
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
            
            # Try standard CLI command
            cmd = f"pyperplan {domain_file} {problem_file}"
            
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(os.path.abspath(problem_file)) or "."
                )
                
                if VERBOSE:
                    print(f"[task_planner] Pyperplan CLI exit code: {result.returncode}")
                    if result.stdout:
                        print(f"[task_planner] stdout: {result.stdout[:200]}")
                    if result.stderr:
                        print(f"[task_planner] stderr: {result.stderr[:200]}")
                
                # Check for solution file (pyperplan creates problem_file.soln)
                soln_file = problem_file + ".soln"
                if os.path.exists(soln_file):
                    if VERBOSE:
                        print(f"[task_planner] Found solution file: {soln_file}")
                    plan = _read_plan_file(soln_file)
                    if plan:
                        if VERBOSE:
                            print(f"[task_planner] Read plan from {soln_file} with {len(plan)} actions")
                        return plan
                
                # Also check for plan in current directory
                problem_basename = os.path.basename(problem_file)
                alt_soln_file = problem_basename + ".soln"
                if os.path.exists(alt_soln_file) and alt_soln_file != soln_file:
                    if VERBOSE:
                        print(f"[task_planner] Found alternative solution file: {alt_soln_file}")
                    plan = _read_plan_file(alt_soln_file)
                    if plan:
                        if VERBOSE:
                            print(f"[task_planner] Read plan from {alt_soln_file} with {len(plan)} actions")
                        return plan
            
            except subprocess.TimeoutExpired:
                if VERBOSE:
                    print(f"[task_planner] Pyperplan CLI timed out after {timeout}s")
            except FileNotFoundError:
                if VERBOSE:
                    print("[task_planner] Pyperplan CLI not found in PATH")
            except Exception as e:
                if VERBOSE:
                    print(f"[task_planner] Pyperplan CLI error: {e}")
        
        except Exception as e:
            if VERBOSE:
                print(f"[task_planner] Pyperplan CLI failed: {e}")
    
    # -------------------------------------------------------------------------
    # Check for .soln file one more time (in case it was created but not detected above)
    # -------------------------------------------------------------------------
    if plan is None:
        soln_file = problem_file + ".soln"
        if os.path.exists(soln_file):
            if VERBOSE:
                print(f"[task_planner] Found existing solution file: {soln_file}")
            plan = _read_plan_file(soln_file)
            if plan:
                if VERBOSE:
                    print(f"[task_planner] Read plan from {soln_file} with {len(plan)} actions")
                return plan
              
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
