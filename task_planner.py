"""
Task planner wrapper for Project 5 (TAMP)

Uses pyperplan if available, otherwise tries fallback options.

API:
    create_pddl_problem_file() - Write PDDL problem from predicates
    call_planner() - Run pyperplan to get plan
    parse_plan_output() - Clean up plan format
    validate_plan() - Check if plan actually works

Expected predicate format (same as abstraction.py):
    {
      "on": [("r","g"), ...],
      "ontable": ["b","c", ...],
      "clear": ["r","b", ...],
      "holding": ["x"] or [],
      "handempty": [True] or []
    }

Plan format: list of tuples like [('pick-up','r'), ('stack','r','g'), ...]

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


VERBOSE = True
FALLBACK_TIMEOUT = 30
MAX_FALLBACK_PLAN_LENGTH = 100


def create_pddl_problem_file(current_state: Dict, goal_state: Dict, 
                             filename: str = "problem.pddl",
                             domain_name: str = "blocksworld",
                             problem_name: str = "tamp-problem") -> str:
    """
    Write a PDDL problem file from current state and goal.
    
    Returns the filename written.
    """
    if not isinstance(current_state, dict):
        raise ValueError("current_state must be a dictionary")
    if not isinstance(goal_state, dict):
        raise ValueError("goal_state must be a dictionary")
    
    # Collect all block names from both states
    objs = set()
    
    for (a, b) in current_state.get("on", []):
        objs.add(a)
        objs.add(b)
    for a in current_state.get("ontable", []):
        objs.add(a)
    for a in current_state.get("clear", []):
        objs.add(a)
    for a in current_state.get("holding", []):
        objs.add(a)
    
    for (a, b) in goal_state.get("on", []):
        objs.add(a)
        objs.add(b)
    for a in goal_state.get("ontable", []):
        objs.add(a)
    for a in goal_state.get("clear", []):
        objs.add(a)
    
    if not objs:
        raise ValueError("No objects found in current_state or goal_state")

    # Write the file
    with open(filename, "w") as f:
        f.write(f"(define (problem {problem_name})\n")
        f.write(f"  (:domain {domain_name})\n\n")
        
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


def call_planner(domain_file: str, problem_file: str, 
                 use_pyperplan: bool = True,
                 timeout: int = 30) -> Optional[List[Tuple]]:
    """
    Call task planner and return list of actions.
    
    Strategy:
        1. Try pyperplan library API (fastest)
        2. Try pyperplan CLI
        3. Check for .soln file
    
    Returns plan as list of tuples, or None if no plan found.
    """
    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain file not found: {domain_file}")
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    plan = None
    
    # Try pyperplan library first
    if use_pyperplan:
        try:
            if VERBOSE:
                print("[task_planner] Attempting to use pyperplan library...")
            
            import pyperplan
            from pyperplan.pddl.parser import Parser
            
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
    
    # Try pyperplan CLI
    if use_pyperplan and plan is None:
        try:
            if VERBOSE:
                print("[task_planner] Attempting pyperplan CLI...")
            
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
                
                # Check for solution file
                soln_file = problem_file + ".soln"
                if os.path.exists(soln_file):
                    if VERBOSE:
                        print(f"[task_planner] Found solution file: {soln_file}")
                    plan = _read_plan_file(soln_file)
                    if plan:
                        if VERBOSE:
                            print(f"[task_planner] Read plan from {soln_file} with {len(plan)} actions")
                        return plan
                
                # Also check current directory
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
    
    # Check for .soln file one more time
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


def parse_plan_output(plan: List) -> List[Tuple]:
    """
    Convert various plan formats to uniform tuple format.
    
    Examples:
        ["(pick-up r)", "(stack r g)"] -> [('pick-up','r'), ('stack','r','g')]
        [('pick-up','r'), ('stack','r','g')] -> [('pick-up','r'), ('stack','r','g')]
    """
    if plan is None:
        return []
    
    normalized = []
    for step in plan:
        if isinstance(step, tuple) or isinstance(step, list):
            normalized.append(tuple(step))
        elif isinstance(step, str):
            s = step.strip()
            # Remove parens if present
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
            s = str(step)
            normalized.extend(parse_plan_output([s]))
    
    return normalized


def normalize_action_names(plan: List[Tuple], 
                           to_format: str = "hyphen") -> List[Tuple]:
    """
    Normalize action names to consistent format.
    to_format can be "hyphen" (pick-up) or "underscore" (pick_up)
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


def validate_plan(plan: List[Tuple], 
                  initial_state: Dict, 
                  goal_state: Dict) -> Tuple[bool, str]:
    """
    Check if plan is valid by simulating execution.
    
    Returns (is_valid, error_message)
    """
    if not plan:
        return False, "Plan is empty"
    
    # Simulate execution
    state = copy.deepcopy(initial_state)
    
    for i, action_tuple in enumerate(plan):
        if not action_tuple:
            return False, f"Empty action at step {i}"
        
        action = action_tuple[0]
        args = action_tuple[1:]
        
        new_state, error = _apply_action(state, action, args)
        if error:
            return False, f"Step {i} ({action} {' '.join(args)}): {error}"
        
        state = new_state
    
    # Check if we hit the goal
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
    Apply action to state, return (new_state, error_message).
    Error message is empty if successful.
    """
    new_state = copy.deepcopy(state)
    action = action.lower().replace("_", "-")
    
    if action == "pick-up":
        if len(args) < 1:
            return state, "pick-up requires 1 argument"
        x = args[0]
        
        if not new_state.get("handempty"):
            return state, "hand not empty"
        if x not in new_state.get("clear", []):
            return state, f"{x} not clear"
        if x not in new_state.get("ontable", []):
            return state, f"{x} not on table"
        
        new_state["holding"] = [x]
        new_state["handempty"] = []
        new_state["ontable"].remove(x)
        new_state["clear"].remove(x)
        
        return new_state, ""
    
    elif action == "put-down":
        if len(args) < 1:
            return state, "put-down requires 1 argument"
        x = args[0]
        
        if x not in new_state.get("holding", []):
            return state, f"not holding {x}"
        
        new_state["holding"] = []
        new_state["handempty"] = [True]
        new_state["ontable"].append(x)
        new_state["clear"].append(x)
        
        return new_state, ""
    
    elif action == "stack":
        if len(args) < 2:
            return state, "stack requires 2 arguments"
        x, y = args[0], args[1]
        
        if x not in new_state.get("holding", []):
            return state, f"not holding {x}"
        if y not in new_state.get("clear", []):
            return state, f"{y} not clear"
        
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
        
        if not new_state.get("handempty"):
            return state, "hand not empty"
        if (x, y) not in new_state.get("on", []):
            return state, f"{x} not on {y}"
        if x not in new_state.get("clear", []):
            return state, f"{x} not clear"
        
        new_state["holding"] = [x]
        new_state["handempty"] = []
        new_state["on"].remove((x, y))
        new_state["clear"].remove(x)
        new_state["clear"].append(y)
        
        return new_state, ""
    
    else:
        return state, f"unknown action: {action}"


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
            steps.append(ln)
    
    return parse_plan_output(steps)


def _parse_plan_from_text(text: str) -> List[Tuple]:
    """Parse plan from text output."""
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
    Convert pyperplan's plan format to our tuple format.
    Pyperplan returns different formats depending on version.
    """
    if plan_obj is None:
        return []
    
    normalized = []
    
    for step in plan_obj:
        if hasattr(step, 'name') and hasattr(step, 'sig'):
            # Action object format
            action = step.name.lower()
            args = [str(arg) for arg in step.sig]
            normalized.append(tuple([action] + args))
        elif isinstance(step, tuple) or isinstance(step, list):
            normalized.append(tuple(step))
        elif isinstance(step, str):
            normalized.extend(parse_plan_output([step]))
        else:
            normalized.extend(parse_plan_output([str(step)]))
    
    return normalized
