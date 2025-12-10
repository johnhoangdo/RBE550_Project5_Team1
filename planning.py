import genesis as gs
import numpy as np
import torch
from typing import Any

from genesis.utils.misc import tensor_to_array
from robot_adapter import RobotAdapter


class PlanningConfig:
    """
    Configuration parameters for motion planning primitives
    
    Different goals require different planning parameters:
    - Default (Goals 1-2): Fast execution, lower precision
    - Goal 3 (6 blocks): Slower, higher precision for tall towers
    - Goal 3 Extended (10+ blocks): Very slow, very high precision
    
    Attributes:
        pre_grasp_height: Height above block for pre-grasp approach (meters)
        pre_place_height: Height above target for pre-place approach (meters)
        descent_waypoints: Number of waypoints for vertical descent
        settling_time: Physics settling steps after placement
    """
    
    # Default configuration (Goals 1-2)
    DEFAULT_PRE_GRASP_HEIGHT = 0.15
    DEFAULT_PRE_PLACE_HEIGHT = 0.15
    DEFAULT_DESCENT_WAYPOINTS = 50
    DEFAULT_SETTLING_TIME = 100
    
    # Goal 3 configuration (6-block tower)
    GOAL3_PRE_GRASP_HEIGHT = 0.25      # Higher approach for tall towers
    GOAL3_PRE_PLACE_HEIGHT = 0.30      # Much higher placement approach
    GOAL3_DESCENT_WAYPOINTS = 100      # Slower, more controlled descent
    GOAL3_SETTLING_TIME = 300          # Longer physics settling
    
    # Goal 3 Extended configuration (10+ block towers)
    GOAL3_EXT_PRE_GRASP_HEIGHT = 0.35  # Very high approach
    GOAL3_EXT_PRE_PLACE_HEIGHT = 0.40  # Very high placement approach
    GOAL3_EXT_DESCENT_WAYPOINTS = 150  # Very slow descent
    GOAL3_EXT_SETTLING_TIME = 500      # Very long settling
    
    def __init__(self, mode='goal3'):
        """
        Initialize planning configuration
        
        Args:
            mode: Configuration mode
                - 'default': Goals 1-2 (fast, low precision)
                - 'goal3': Goal 3 - 6 blocks (slow, high precision) [DEFAULT]
                - 'goal3_extended': 10+ blocks (very slow, very high precision)
        
        Notes:
            - Default mode is 'goal3' to match abstraction.py defaults
            - Switch to 'default' for Goals 1-2 if needed
            - Switch to 'goal3_extended' for 10+ block towers
        """
        if mode == 'goal3_extended':
            self.pre_grasp_height = self.GOAL3_EXT_PRE_GRASP_HEIGHT
            self.pre_place_height = self.GOAL3_EXT_PRE_PLACE_HEIGHT
            self.descent_waypoints = self.GOAL3_EXT_DESCENT_WAYPOINTS
            self.settling_time = self.GOAL3_EXT_SETTLING_TIME
            self.mode = 'goal3_extended'
            print('\n' + '='*60)
            print('[Planning] Goal 3 Extended mode (10+ blocks)'.center(60))
            print('='*60)
            print(f'  Pre-grasp height:  {self.pre_grasp_height:.2f}m')
            print(f'  Pre-place height:  {self.pre_place_height:.2f}m')
            print(f'  Descent waypoints: {self.descent_waypoints}')
            print(f'  Settling time:     {self.settling_time} steps')
            print('='*60 + '\n')
            
        elif mode == 'default':
            self.pre_grasp_height = self.DEFAULT_PRE_GRASP_HEIGHT
            self.pre_place_height = self.DEFAULT_PRE_PLACE_HEIGHT
            self.descent_waypoints = self.DEFAULT_DESCENT_WAYPOINTS
            self.settling_time = self.DEFAULT_SETTLING_TIME
            self.mode = 'default'
            print('\n' + '='*60)
            print('[Planning] Default mode (Goals 1-2)'.center(60))
            print('='*60)
            print(f'  Pre-grasp height:  {self.pre_grasp_height:.2f}m')
            print(f'  Pre-place height:  {self.pre_place_height:.2f}m')
            print(f'  Descent waypoints: {self.descent_waypoints}')
            print(f'  Settling time:     {self.settling_time} steps')
            print('='*60 + '\n')
            
        else:  # 'goal3' - default mode
            self.pre_grasp_height = self.GOAL3_PRE_GRASP_HEIGHT
            self.pre_place_height = self.GOAL3_PRE_PLACE_HEIGHT
            self.descent_waypoints = self.GOAL3_DESCENT_WAYPOINTS
            self.settling_time = self.GOAL3_SETTLING_TIME
            self.mode = 'goal3'
            print('\n' + '='*60)
            print('[Planning] Goal 3 mode (6 blocks)'.center(60))
            print('='*60)
            print(f'  Pre-grasp height:  {self.pre_grasp_height:.2f}m')
            print(f'  Pre-place height:  {self.pre_place_height:.2f}m')
            print(f'  Descent waypoints: {self.descent_waypoints}')
            print(f'  Settling time:     {self.settling_time} steps')
            print('='*60 + '\n')


# Global configuration instance (defaults to Goal 3)
planning_config = PlanningConfig(mode='goal3')


def set_planning_mode(mode='goal3'):
    """
    Convenience function to switch planning configuration
    
    Args:
        mode: 'default', 'goal3', or 'goal3_extended'
    
    Usage:
        import planning
        planning.set_planning_mode('goal3_extended')
    """
    global planning_config
    planning_config = PlanningConfig(mode=mode)


def get_planning_config():
    """
    Get current planning configuration
    
    Returns:
        PlanningConfig: Current configuration instance
    """
    return planning_config



def _ensure_adapter(robot: Any, scene: Any) -> RobotAdapter:
    """Wrap raw genesis robot in RobotAdapter if needed.

    This keeps PlannerInterface backward-compatible with callers that may
    still pass the raw genesis entity.
    """
    if isinstance(robot, RobotAdapter):
        return robot
    return RobotAdapter(robot, scene)


class PlannerInterface:
    def __init__(self, robot: Any, scene: Any):
        # ensure we have a RobotAdapter so the rest of the code can rely on a
        # stable interface (but attribute access is forwarded to the raw robot)
        self.robot = _ensure_adapter(robot, scene)
        self.scene = scene
        self.attached_object = None

    def diagnose_bounds_violation(self, si, state):
        # print the bounds the current state is violating
        violated_bounds = []
        for i_q in range(self.robot.n_qs):
            val = state[i_q]
            low = si.getStateSpace().getBounds().low[i_q]
            high = si.getStateSpace().getBounds().high[i_q]
            if val < low or val > high:
                violated_bounds.append((i_q, val, low, high))
        gs.logger.warning(f"State violates bounds on joints: {violated_bounds}")

    def diagnose_valid_violation(self, state):
        # set robot to the candidate start and check collisions / joint violations
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        print(self.robot.get_qpos())
        collision_pairs = self.robot.detect_collision()
        if collision_pairs.any() and len(collision_pairs) > 0:
            bad_links = set()
            for a, b in collision_pairs:
                # print(self.scene.rigid_solver.geoms[a])
                print(self.scene.rigid_solver.geoms[a].link.name)
                # print(self.scene.rigid_solver.geoms[b])
                print(self.scene.rigid_solver.geoms[b].link.name)
                bad_links.add(self.scene.rigid_solver.geoms[a].link.name)
                bad_links.add(self.scene.rigid_solver.geoms[b].link.name)
            gs.logger.warning(f"State causes collisions between links: {sorted(bad_links)}")

    def plan_path(
            self,
            qpos_goal,
            qpos_start=None,
            timeout=5.0,
            smooth_path=True,
            num_waypoints=100,
            attached_object=None,
            planner="RRTConnect",
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            Goal state
        qpos_start : None | array_like, optional
            Start state. If None, the current state of the rigid entity will be used. Defaults to None
        timeout : float, optional
            Max time (in seconds) allowed for the motion planning algorithm to find a solution. Default: 5.0s
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            Whether to ignore joint limits during motion planning. Defaults to False.
        planner : str, optional
            The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.

        Returns
        -------
        waypoints : list
            A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
        """

        ########## validate ##########
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            ou.setLogLevel(ou.LOG_ERROR)
        except:
            gs.raise_exception(
                    "Failed to import OMPL. Did you install? (For installation instructions, see https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning)"
            )

        supported_planners = [
            "PRM",
            "RRT",
            "RRTConnect",
            "RRTstar",
            "EST",
            "FMT",
            "BITstar",
            "ABITstar",
        ]
        if planner not in supported_planners:
            gs.raise_exception(f"Planner {planner} is not supported. Supported planners: {supported_planners}.")

        if self.robot._solver.n_envs > 0:
            gs.raise_exception("Motion planning is not supported for batched envs (yet).")

        if self.robot.n_qs != self.robot.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        qpos_cur = self.robot.get_qpos()

        if qpos_start is None:
            qpos_start = self.robot.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.robot.n_qs,) or qpos_goal.shape != (self.robot.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        ######### process joint limit ##########

        # ensure we use numpy float64 for bounds
        q_limit_lower = np.asarray(self.robot.q_limit[0], dtype=float)
        q_limit_upper = np.asarray(self.robot.q_limit[1], dtype=float)

        # Clip start/goal joint vectors to be strictly within bounds (with
        # a tiny epsilon) to avoid floating-point rounding causing an OMPL
        # "out of bounds" / start-tree initialization failure. We log if
        # clipping occurred so it's visible during debugging.
        epsilon = 1e-9
        clipped = False
        qpos_start = np.asarray(qpos_start, dtype=float)
        qpos_goal = np.asarray(qpos_goal, dtype=float)
        qpos_start_clipped = np.clip(qpos_start, q_limit_lower + epsilon, q_limit_upper - epsilon)
        qpos_goal_clipped = np.clip(qpos_goal, q_limit_lower + epsilon, q_limit_upper - epsilon)
        if not np.allclose(qpos_start_clipped, qpos_start):
            gs.logger.warning("Clipping qpos_start to satisfy joint limits (tiny epsilon applied)")
            clipped = True
        if not np.allclose(qpos_goal_clipped, qpos_goal):
            gs.logger.warning("Clipping qpos_goal to satisfy joint limits (tiny epsilon applied)")
            clipped = True
        # Use clipped arrays from here on
        qpos_start = qpos_start_clipped
        qpos_goal = qpos_goal_clipped

        ######### setup OMPL ##########
        space = ob.RealVectorStateSpace(self.robot.n_qs)
        bounds = ob.RealVectorBounds(self.robot.n_qs)

        for i_q in range(self.robot.n_qs):
            # pass native Python float (double) to OMPL to match C++ signature
            bounds.setLow(i_q, float(q_limit_lower[i_q]))
            bounds.setHigh(i_q, float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)

        self.attached_object = attached_object
        
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_ompl_state_valid))
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.robot.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        # Diagnostic: check start/goal satisfy bounds and are valid according to the state validity checker
        si = ss.getSpaceInformation()
        start_in_bounds = bool(si.satisfiesBounds(state_start.get()))
        if not start_in_bounds:
            gs.logger.warning(f"OMPL start state out of bounds")
            self.diagnose_bounds_violation(si, state_start.get())

        goal_in_bounds = bool(si.satisfiesBounds(state_goal.get()))
        if not goal_in_bounds:
            gs.logger.warning(f"OMPL goal state out of bounds")
            self.diagnose_bounds_violation(si, state_goal)

        start_valid = bool(si.isValid(state_start.get()))
        if not start_valid:
            gs.logger.warning(f"OMPL start state invalid")
            self.diagnose_valid_violation(state_start)

        goal_valid = bool(si.isValid(state_goal.get()))
        if not goal_valid:
            gs.logger.warning(f"OMPL goal state invalid")
            self.diagnose_valid_violation(state_goal)

        # set start/goal in OMPL
        ss.setStartAndGoalStates(state_start, state_goal)
        ss.setup()

        ######### solve ##########
        solved = ss.solve(timeout)
        waypoints = []
        if solved:
            gs.logger.info("Path solution found successfully.")
            path = ss.getSolutionPath()
            if smooth_path:
                ss.simplifySolution()

            path.interpolate(num_waypoints)
            print("Number of waypoints in path:", path.getStateCount())
            waypoints = self._ompl_states_to_tensor_list(path.getStates())
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")

        ########## restore original state #########
        self.robot.set_qpos(qpos_cur)

        return waypoints

    def _is_ompl_state_valid(self, state):
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        collision_pairs = self.robot.detect_collision()

        if not len(collision_pairs):
            return True

        if not self.attached_object:
            return False

        return self.collision_with_attached_object(collision_pairs)

    def collision_with_attached_object(self, collision_pairs):
        finger_names = {"left_finger", "right_finger", "hand"}
        for a, b in collision_pairs:
            name_a = self.scene.rigid_solver.geoms[a].link.name
            name_b = self.scene.rigid_solver.geoms[b].link.name
            if (name_a in finger_names and b == self.attached_object.idx) or \
                 (name_b in finger_names and a == self.attached_object.idx):
                continue
            return False
        return True

    def _ompl_states_to_tensor_list(self, states):
        tensor_list = []
        for state in states:
            tensor_list.append(self._ompl_state_to_tensor(state))
        return tensor_list

    def _ompl_state_to_tensor(self, state):
        tensor = torch.empty(self.robot.n_qs, dtype=gs.tc_float, device=gs.device)
        for i in range(self.robot.n_qs):
            tensor[i] = state[i]
        return tensor

    # =========================================================================
    # MOTION PRIMITIVES
    # =========================================================================
    
    def plan_to_position(self, target_pos, gripper_open=True):
        """
        Plan IK to reach a specific 3D position with gripper pointing down.
        
        Args:
            target_pos: Target [x, y, z] position for end effector
            gripper_open: Whether gripper should be open (True) or closed (False)
        
        Returns:
            qpos configuration if successful, None if failed
        """
        try:
            qpos = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=target_pos,
                quat=np.array([0, 1, 0, 0])  # Pointing down
            )
            
            if qpos is not None:
                # Set gripper state
                qpos[-2:] = 0.04 if gripper_open else 0.005
            
            return qpos
            
        except Exception as e:
            gs.logger.warning(f"IK failed for position {target_pos}: {e}")
            return None

    def pick_up(self, block, pre_grasp_height=0.25, grasp_offset=0.10):
        """
        Pick up a block from the table or from on top of another block
        
        Args:
            block: Genesis block entity to pick up
            pre_grasp_height: Height above block TOP for approach (meters)
            grasp_offset: Distance above block TOP for grasping (meters)
                        SMALLER = grab closer to top (lower down)
                        LARGER = grab higher above top
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            block_pos = block.get_pos()
            gs.logger.info(f"Attempting pick-up at position ({block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f})")
            
            BLOCK_HEIGHT = 0.04  # 4cm blocks
            
            # 1. Plan to pre-grasp pose ABOVE TOP of block
            block_top_z = block_pos[2] + BLOCK_HEIGHT/2
            pre_grasp_pos = np.array([
                block_pos[0], 
                block_pos[1], 
                block_top_z + pre_grasp_height  # High above block
            ])
            
            qpos_pregrasp = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=pre_grasp_pos,
                quat=np.array([0, 1, 0, 0])  # Pointing down
            )
            
            if qpos_pregrasp is None:
                gs.logger.warning("IK failed for pre-grasp pose")
                return False
            
            # Set gripper to open
            qpos_pregrasp[-2:] = 0.04
            
            # Plan collision-free path to pre-grasp
            path = self.plan_path(qpos_goal=qpos_pregrasp, timeout=10.0, num_waypoints=300)
            if not path:
                gs.logger.warning("Failed to plan path to pre-grasp")
                return False
            
            # Execute path to pre-grasp
            gs.logger.info("Moving to pre-grasp...")
            for waypoint in path:
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # 2. Move straight down to grasp pose (just above block top)
            grasp_pos = np.array([
                block_pos[0], 
                block_pos[1], 
                block_top_z + grasp_offset  # Just above top of block
            ])
            
            qpos_grasp = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=grasp_pos,
                quat=np.array([0, 1, 0, 0])
            )
            
            if qpos_grasp is None:
                gs.logger.warning("IK failed for grasp pose")
                return False
            
            qpos_grasp[-2:] = 0.04  # Keep gripper open
            
            # Straight line interpolation down (SLOWER for control)
            gs.logger.info("Lowering to grasp...")
            num_steps = 150  # Increased from 100
            for i in range(num_steps + 1):
                alpha = i / num_steps
                waypoint = (1-alpha) * qpos_pregrasp + alpha * qpos_grasp
                waypoint[-2:] = 0.04  # Keep gripper open
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # 3. Close gripper with tighter grip
            gs.logger.info("Closing gripper...")
            qpos_grasp[-2:] = 0.005  # Tighter grip (was 0.01)
            for _ in range(100):  # More time to ensure firm grasp (was 50)
                self.robot.control_dofs_position(qpos_grasp)
                self.scene.step()
            
            # Verify block is grasped by checking position (single check)
            block_pos_after = block.get_pos()
            if block_pos_after[2] < block_pos[2] - 0.01:  # Block fell
                gs.logger.warning("Block may have fallen during grasp")
            
            # Optional: Log grasp alignment for debugging (commented out by default)
            # gripper_pos = self.robot.get_eef_pose()[:3]
            # xy_error = gripper_pos[:2] - block_pos_after[:2]
            # gs.logger.info(f"Grasp offset: ({xy_error[0]*1000:.1f}mm, {xy_error[1]*1000:.1f}mm)")
            
            # 4. Attach object for collision checking
            self.attached_object = block
            gs.logger.info(f"Attached block for collision checking")
            
            # 5. Lift straight up to pre-grasp height (SLOW to prevent dropping)
            gs.logger.info("Lifting...")
            num_steps = 200  # Increased from 100 for slower lift
            for i in range(num_steps + 1):
                alpha = i / num_steps
                waypoint = (1-alpha) * qpos_grasp + alpha * qpos_pregrasp
                waypoint[-2:] = 0.005  # Keep gripper tightly closed
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            gs.logger.info("Pick-up completed successfully")
            return True
            
        except Exception as e:
            gs.logger.error(f"Pick-up failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def put_down(self, target_pos, pre_place_height=0.30, place_offset=0.12):
        """
        Place the currently held object at target position
        
        Args:
            target_pos: np.array [x, y, z] - target CENTER position for block
            pre_place_height: Height above target for approach (meters)
            place_offset: Additional height above target CENTER for gripper (meters)
                        HIGHER value = release block HIGHER (less slamming)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.attached_object is None:
                gs.logger.warning("No object attached to put down")
                return False
            
            gs.logger.info(f"Attempting put-down at target position {target_pos}")
            
            BLOCK_HEIGHT = 0.04  # 4cm blocks
            
            # Calculate where gripper should be
            # target_pos[2] is where block CENTER should end up
            # Gripper should be ABOVE block center by half block height + offset
            target_gripper_z = target_pos[2] + BLOCK_HEIGHT/2 + place_offset
            
            # 1. Plan to pre-place pose above target
            pre_place_pos = np.array([
                target_pos[0], 
                target_pos[1], 
                target_gripper_z + pre_place_height
            ])
            
            qpos_preplace = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=pre_place_pos,
                quat=np.array([0, 1, 0, 0])
            )
            
            if qpos_preplace is None:
                gs.logger.warning("IK failed for pre-place pose")
                return False
            
            qpos_preplace[-2:] = 0.01  # Gripper closed (holding object)
            
            # Plan path WITH attached object for collision checking
            # Use RRTstar for stacking to find straighter, more optimal paths
            path = self.plan_path(
                qpos_goal=qpos_preplace,
                attached_object=self.attached_object,
                timeout=10.0,
                num_waypoints=300,
                planner="RRTstar"  # More optimal paths for tight grids
            )
            
            if not path:
                gs.logger.warning("Failed to plan path to pre-place")
                return False
            
            # Execute path
            gs.logger.info("Moving to pre-place...")
            for waypoint in path:
                waypoint[-2:] = 0.005  # Keep gripper TIGHTLY closed (was 0.01)
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # 2. Lower GENTLY to place position
            place_pos = np.array([
                target_pos[0], 
                target_pos[1], 
                target_gripper_z  # Gripper position for release
            ])
            
            qpos_place = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=place_pos,
                quat=np.array([0, 1, 0, 0])
            )
            
            if qpos_place is None:
                gs.logger.warning("IK failed for place pose")
                return False
            
            qpos_place[-2:] = 0.005  # Keep gripper TIGHTLY closed
            
            # SLOW, GENTLE lowering
            gs.logger.info("Lowering GENTLY to place...")
            num_steps = 150  # Slower = gentler
            for i in range(num_steps + 1):
                alpha = i / num_steps
                waypoint = (1-alpha) * qpos_preplace + alpha * qpos_place
                waypoint[-2:] = 0.005  # Keep gripper TIGHTLY closed
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # CRITICAL: Wait for motion to fully stop before releasing!
            gs.logger.info("Holding position (motion stabilization)...")
            for _ in range(100):  # Hold steady for 100 steps
                self.robot.control_dofs_position(qpos_place)
                qpos_place[-2:] = 0.005  # Keep closed!
                self.scene.step()
            
            # 3. Open gripper SLOWLY (now that motion has stopped)
            gs.logger.info("Opening gripper...")
            for step in range(100):  # Slow opening over 100 steps
                alpha = step / 100
                grip_width = 0.005 + alpha * (0.04 - 0.005)  # 0.005 → 0.04
                qpos_place[-2:] = grip_width
                self.robot.control_dofs_position(qpos_place)
                self.scene.step()
            
            # 4. Detach object AFTER settling (moved from before)
            # Save reference before detaching
            placed_block = self.attached_object
            
            # 5. Let physics settle FIRST (CRITICAL - before retraction!)
            gs.logger.info("="*60)
            gs.logger.info(f"PUT-DOWN VERIFICATION - Target: ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
            gs.logger.info("Settling physics (200 steps for tight spacing)...")
            
            for _ in range(200):  # Reduced from 500 for faster execution
                self.scene.step()
            
            # 6. VERIFY POSITION (before detaching/retracting!)
            if placed_block:
                final_pos = placed_block.get_pos()
                dx = abs(final_pos[0] - target_pos[0])
                dy = abs(final_pos[1] - target_pos[1])
                dz = abs(final_pos[2] - target_pos[2])
                
                # For Goal 4A with spacing=0.045m (4.5cm), tolerance = 0.010m (10mm)
                TIGHT_TOLERANCE = 0.010  # 10mm for Goal 4A (relaxed from 5mm)
                tolerance = TIGHT_TOLERANCE
                
                gs.logger.info(f"Final position: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
                gs.logger.info(f"Position error: dx={dx*1000:.2f}mm, dy={dy*1000:.2f}mm, dz={dz*1000:.2f}mm")
                gs.logger.info(f"Tolerance: {tolerance*1000:.1f}mm")
                
                if dx > tolerance or dy > tolerance:
                    gs.logger.error(f"❌ POSITION ERROR EXCEEDS TOLERANCE!")
                    gs.logger.error(f"   Target:  ({target_pos[0]:.4f}, {target_pos[1]:.4f})")
                    gs.logger.error(f"   Actual:  ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
                    gs.logger.error(f"   Error:   ({dx*1000:.2f}mm, {dy*1000:.2f}mm)")
                    gs.logger.error(f"   Limit:   {tolerance*1000:.1f}mm")
                    gs.logger.info("="*60)
                    
                    # IMMEDIATE REPOSITIONING - Don't retract, pick up and try again!
                    gs.logger.warning("Attempting immediate repositioning...")
                    
                    # CRITICAL FIX: Re-center gripper above block's CURRENT position!
                    # Block may have shifted, so can't use old qpos_place
                    current_block_pos = placed_block.get_pos()
                    gs.logger.info(f"Re-centering gripper above block at ({current_block_pos[0]:.4f}, {current_block_pos[1]:.4f})")
                    
                    # Calculate IK for position directly above block's current location
                    # Use same gripper height as original placement: center + half_block + offset
                    BLOCK_HEIGHT = 0.04
                    regrasp_height = current_block_pos[2] + BLOCK_HEIGHT/2 + place_offset
                    
                    regrasp_target = np.array([
                        current_block_pos[0],
                        current_block_pos[1], 
                        regrasp_height  # Same height as original placement
                    ])
                    
                    # Plan path to re-center above block
                    qpos_recenter = self.plan_to_position(regrasp_target, gripper_open=True)
                    if qpos_recenter is None:
                        gs.logger.error("Failed to plan re-centering motion!")
                        # Still try with old position as fallback
                        qpos_recenter = qpos_place
                    else:
                        # CRITICAL: Move safely by lifting, shifting, lowering (not direct path!)
                        gs.logger.info("Safe re-centering motion (lift → shift → lower)...")
                        
                        # Step 1: Lift straight up FIRST to clear any obstacles
                        gs.logger.info("  Step 1/3: Lifting to safe height...")
                        if hasattr(qpos_preplace, 'clone'):
                            qpos_safe_height = qpos_preplace.clone()
                        else:
                            qpos_safe_height = qpos_preplace.copy()
                        qpos_safe_height[-2:] = 0.04  # Keep open during lift
                        
                        for i in range(50):
                            alpha = i / 50
                            waypoint = (1-alpha) * qpos_place + alpha * qpos_safe_height
                            waypoint[-2:] = 0.04  # Keep open
                            self.robot.control_dofs_position(waypoint)
                            self.scene.step()
                        
                        # Step 2: Plan path to new XY position at safe height
                        gs.logger.info("  Step 2/3: Moving to new XY position at safe height...")
                        # Calculate position at safe height above new target
                        current_block_pos = placed_block.get_pos()
                        BLOCK_HEIGHT = 0.04
                        safe_height_above_new = current_block_pos[2] + BLOCK_HEIGHT/2 + place_offset + pre_place_height
                        
                        new_xy_safe_pos = np.array([
                            current_block_pos[0],
                            current_block_pos[1],
                            safe_height_above_new
                        ])
                        
                        qpos_new_xy_safe = self.robot.inverse_kinematics(
                            link=self.robot.get_link("hand"),
                            pos=new_xy_safe_pos,
                            quat=np.array([0, 1, 0, 0])
                        )
                        
                        if qpos_new_xy_safe is not None:
                            qpos_new_xy_safe[-2:] = 0.04
                            # Use motion planning for horizontal move
                            path_horizontal = self.plan_path(
                                qpos_goal=qpos_new_xy_safe,
                                timeout=8.0,  # Increased from 3.0 for better paths
                                num_waypoints=200
                            )
                            
                            if path_horizontal:
                                for waypoint in path_horizontal:
                                    waypoint[-2:] = 0.04  # Keep open
                                    self.robot.control_dofs_position(waypoint)
                                    self.scene.step()
                            else:
                                # Fallback: direct interpolation if planning fails
                                for i in range(50):
                                    alpha = i / 50
                                    waypoint = (1-alpha) * qpos_safe_height + alpha * qpos_new_xy_safe
                                    waypoint[-2:] = 0.04
                                    self.robot.control_dofs_position(waypoint)
                                    self.scene.step()
                        
                        # Step 3: Lower straight down to re-grasp position
                        gs.logger.info("  Step 3/3: Lowering to re-grasp position...")
                        # Re-calculate qpos_recenter at lower height
                        qpos_recenter = self.plan_to_position(regrasp_target, gripper_open=True)
                        if qpos_recenter is None:
                            qpos_recenter = qpos_place  # Fallback
                        
                        qpos_recenter[-2:] = 0.04
                        
                        # Get current position (should be at safe height)
                        current_qpos = self.robot.get_dofs_position()
                        
                        for i in range(50):
                            alpha = i / 50
                            waypoint = (1-alpha) * current_qpos + alpha * qpos_recenter
                            waypoint[-2:] = 0.04  # Keep open
                            self.robot.control_dofs_position(waypoint)
                            self.scene.step()
                        
                        qpos_place = qpos_recenter  # Update placement position
                    
                    # Now close gripper to re-grasp (centered above block)
                    gs.logger.info("Closing gripper to re-grasp...")
                    if hasattr(qpos_place, 'clone'):
                        qpos_grasp = qpos_place.clone()
                    else:
                        qpos_grasp = qpos_place.copy()
                    qpos_grasp[-2:] = 0.005
                    
                    for _ in range(50):
                        self.robot.control_dofs_position(qpos_grasp)
                        self.scene.step()
                    
                    # Re-attach for repositioning
                    self.attached_object = placed_block
                    
                    # Lift slightly - use .clone() for tensors, .copy() for numpy
                    gs.logger.info("Lifting for repositioning...")
                    if hasattr(qpos_preplace, 'clone'):  # PyTorch tensor
                        qpos_lift = qpos_preplace.clone()
                    else:  # Numpy array
                        qpos_lift = qpos_preplace.copy()
                    qpos_lift[-2:] = 0.005
                    
                    for i in range(100):
                        alpha = i / 100
                        waypoint = (1-alpha) * qpos_place + alpha * qpos_lift
                        waypoint[-2:] = 0.005
                        self.robot.control_dofs_position(waypoint)
                        self.scene.step()
                    
                    # Try placing again with more precision
                    gs.logger.info("Attempting corrected placement...")
                    for i in range(150):
                        alpha = i / 150
                        waypoint = (1-alpha) * qpos_lift + alpha * qpos_place
                        waypoint[-2:] = 0.005
                        self.robot.control_dofs_position(waypoint)
                        self.scene.step()
                    
                    # Open gripper
                    qpos_place[-2:] = 0.04
                    for _ in range(50):
                        self.robot.control_dofs_position(qpos_place)
                        self.scene.step()
                    
                    # Settle again
                    for _ in range(500):
                        self.scene.step()
                    
                    # Check again
                    final_pos2 = placed_block.get_pos()
                    dx2 = abs(final_pos2[0] - target_pos[0])
                    dy2 = abs(final_pos2[1] - target_pos[1])
                    
                    gs.logger.info(f"After repositioning: error=({dx2*1000:.2f}mm, {dy2*1000:.2f}mm)")
                    
                    if dx2 > tolerance or dy2 > tolerance:
                        gs.logger.error("Still exceeds tolerance after repositioning!")
                        
                        # CRITICAL: Pick up the misplaced block so replanner can try again!
                        # Don't leave it in the wrong position!
                        gs.logger.warning("Picking up misplaced block to allow repositioning on replan...")
                        
                        # Close gripper to grasp the block
                        if hasattr(qpos_place, 'clone'):
                            qpos_grasp = qpos_place.clone()
                        else:
                            qpos_grasp = qpos_place.copy()
                        qpos_grasp[-2:] = 0.005  # Close gripper
                        
                        for _ in range(50):
                            self.robot.control_dofs_position(qpos_grasp)
                            self.scene.step()
                        
                        # Re-attach the block (we're holding it now)
                        self.attached_object = placed_block
                        gs.logger.info("Block re-grasped - will be held for replanning")
                        
                        # Lift to safe height
                        for i in range(num_steps + 1):
                            alpha = i / num_steps
                            waypoint = (1-alpha) * qpos_grasp + alpha * qpos_preplace
                            waypoint[-2:] = 0.005  # Keep closed
                            self.robot.control_dofs_position(waypoint)
                            self.scene.step()
                        
                        # Return False - robot is now holding the block
                        # Replanner will see hand NOT empty and can try put-down again
                        gs.logger.info("Holding misplaced block - ready for replan")
                        return False
                    else:
                        gs.logger.info("✓ Repositioning successful!")
                else:
                    gs.logger.info(f"✓ Position within tolerance ({tolerance*1000:.1f}mm)")
                    gs.logger.info("="*60)
            
            # 7. NOW detach and retract (only if position is good)
            self.attached_object = None
            gs.logger.info("Detached object")
            
            # 8. Retract straight up SLOWLY (directly upward)
            gs.logger.info("Retracting straight up...")
            for i in range(num_steps + 1):
                alpha = i / num_steps
                waypoint = (1-alpha) * qpos_place + alpha * qpos_preplace
                waypoint[-2:] = 0.04  # Keep gripper open
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            gs.logger.info("Put-down completed successfully")
            return True
            
        except Exception as e:
            gs.logger.error(f"Put-down failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stack(self, target_block, stack_height=0.04):
        """
        Stack the currently held block on top of target block
        
        Args:
            target_block: Genesis block entity to stack on
            stack_height: Height of one block (meters) - default 0.04 for 4cm blocks
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.attached_object is None:
                gs.logger.warning("No object attached to stack")
                return False
            
            # Get target block position (center of target block)
            target_pos = target_block.get_pos()
            gs.logger.info(f"Stacking on block at {target_pos}")
            
            # Calculate where new block's CENTER should be
            # target_pos[2] is center of lower block
            # New block center = lower block center + one full block height
            stack_pos = np.array([
                target_pos[0], 
                target_pos[1], 
                target_pos[2] + stack_height  # One block height above center
            ])
            
            # Use put_down with HIGHER offset to prevent slamming
            return self.put_down(stack_pos, place_offset=0.08)
            
        except Exception as e:  
            gs.logger.error(f"Stack failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def push(self, block, push_direction, push_distance=0.02, push_height=0.03):
        """
        Push a block in a specified direction for place-then-push strategy.
        
        Used in Goal 4B to achieve tight spacing (5-10mm) after initial placement.
        
        Args:
            block: Genesis block entity to push
            push_direction: np.array([dx, dy]) - normalized direction vector
            push_distance: How far to push (meters) - default 2cm
            push_height: Height above block center for pushing (meters) - default 3cm
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get block's current position
            block_pos = block.get_pos()
            gs.logger.info(f"Pushing block at ({block_pos[0]:.3f}, {block_pos[1]:.3f}) by {push_distance*1000:.1f}mm")
            
            BLOCK_HEIGHT = 0.04  # 4cm blocks
            
            # Normalize push direction
            direction = np.array(push_direction[:2])  # Only X, Y
            direction = direction / np.linalg.norm(direction)
            
            # Calculate starting position: Behind the block, above center
            offset_distance = 0.03  # 3cm behind block
            start_pos = np.array([
                block_pos[0] - direction[0] * offset_distance,
                block_pos[1] - direction[1] * offset_distance,
                block_pos[2] + push_height  # Above block center
            ])
            
            # Calculate end position: Push distance forward
            end_pos = np.array([
                start_pos[0] + direction[0] * (offset_distance + push_distance),
                start_pos[1] + direction[1] * (offset_distance + push_distance),
                start_pos[2]  # Same height
            ])
            
            # Step 1: Move to start position (behind block)
            gs.logger.info("Moving to push start position...")
            qpos_start = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=start_pos,
                quat=np.array([0, 1, 0, 0])
            )
            
            if qpos_start is None:
                gs.logger.warning("IK failed for push start position")
                return False
            
            qpos_start[-2:] = 0.04  # Gripper open
            
            # Plan path to start position
            path_to_start = self.plan_path(qpos_goal=qpos_start, timeout=5.0, num_waypoints=200)
            if not path_to_start:
                gs.logger.warning("Failed to plan path to push start")
                return False
            
            for waypoint in path_to_start:
                waypoint[-2:] = 0.04
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # Step 2: Calculate end position IK
            qpos_end = self.robot.inverse_kinematics(
                link=self.robot.get_link("hand"),
                pos=end_pos,
                quat=np.array([0, 1, 0, 0])
            )
            
            if qpos_end is None:
                gs.logger.warning("IK failed for push end position")
                return False
            
            qpos_end[-2:] = 0.04  # Keep gripper open
            
            # Step 3: Execute push motion (slow, controlled)
            gs.logger.info("Executing push...")
            num_steps = 100  # Slow push
            for i in range(num_steps + 1):
                alpha = i / num_steps
                waypoint = (1-alpha) * qpos_start + alpha * qpos_end
                waypoint[-2:] = 0.04
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            # Step 4: Retract
            gs.logger.info("Retracting after push...")
            for i in range(50):
                alpha = i / 50
                waypoint = (1-alpha) * qpos_end + alpha * qpos_start
                waypoint[-2:] = 0.04
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            gs.logger.info("Push completed successfully")
            return True
            
        except Exception as e:
            gs.logger.error(f"Push failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unstack(self, block, below_block):
        """
        Remove a block from on top of another block
        
        This is essentially the same as pick_up(), since we're picking up
        a block that happens to be on top of another.
        
        Args:
            block: Genesis block entity to unstack (top block)
            below_block: Genesis block entity that block is on (not used directly)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            gs.logger.info(f"Unstacking block from another block")
            
            # Unstack is just pick_up with the top block
            return self.pick_up(block)
            
        except Exception as e:
            gs.logger.error(f"Unstack failed with exception: {e}")
            return False

    def move_to_home(self):
        """
        Move robot to a home position
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            gs.logger.info("Moving to home position...")
            
            # Home configuration (taken from from demo.py)
            home_qpos = np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.04, 0.04])
            
            # Clear any attached object
            self.attached_object = None
            
            # Plan path to home
            path = self.plan_path(qpos_goal=home_qpos, timeout=5.0, num_waypoints=200)
            if not path:
                gs.logger.warning("Failed to plan path to home")
                return False
            
            # Execute path
            for waypoint in path:
                self.robot.control_dofs_position(waypoint)
                self.scene.step()
            
            gs.logger.info("Reached home position")
            return True
            
        except Exception as e:
            gs.logger.error(f"Move to home failed: {e}")
            return False
