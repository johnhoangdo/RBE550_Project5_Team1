"""
Scene factory helpers for Project 5.

Each function creates a scene and returns (scene, franka, blocks_state)
so you can jump right into testing.
"""
from typing import Any, Dict, Tuple
import random
import time
random.seed(time.time())

import numpy as np
import genesis as gs
from robot_adapter import RobotAdapter


def _build_base_scene(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.5)) -> gs.Scene:
    """Basic scene setup with viewer settings."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=8),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=True,
    )
    return scene

def _elevate_robot_base(franka: Any) -> None:
    """Raise robot base slightly to avoid collision issues on startup."""
    base_pos = np.asarray(franka.get_pos(), dtype=float)
    new_pos = base_pos.copy()
    new_pos[2] += 0.01
    franka.set_pos(new_pos) 

def _rand_xy(base, noise=0.05):
    """Add random noise to x/y position (keeps z the same)."""
    dx = random.uniform(-noise, noise)
    dy = random.uniform(-noise, noise)
    return (base[0] + dx, base[1] + dy, base[2])

def create_scene_6blocks() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create the default 6-block scene with blocks spread out on the table.
    
    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene()
    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Position blocks with slight randomization (up to 5cm noise in x/y)
    posR = _rand_xy((0.65, 0.0, 0.02))
    posG = _rand_xy((0.65, 0.2, 0.02))
    posB = _rand_xy((0.65, 0.4, 0.02))
    posY = _rand_xy((0.45, 0.0, 0.02))
    posM = _rand_xy((0.45, 0.2, 0.02))
    posC = _rand_xy((0.45, 0.4, 0.02))

    # Create colored blocks
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )
    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    # Add robot
    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # Build the scene (sets up physics and visuals)
    scene.build()

    # Set initial robot joint positions (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # Lift robot slightly to prevent initial collision weirdness
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state

def create_scene_12_yellow_blocks() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create scene with 12 identical yellow blocks for Tower configuration (Goal 4A).
    
    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene()
    plane = scene.add_entity(gs.morphs.Plane())
    
    blocks = {}
    # Create 12 yellow blocks with distinct names
    positions = [
        (0.70, -0.3, 0.02), (0.70, -0.1, 0.02), (0.70, 0.1, 0.02), (0.70, 0.3, 0.02),
        (0.55, -0.3, 0.02), (0.55, -0.1, 0.02), (0.55, 0.1, 0.02), (0.55, 0.3, 0.02),
        (0.40, -0.3, 0.02), (0.40, -0.1, 0.02), (0.40, 0.1, 0.02), (0.40, 0.3, 0.02)
    ]

# ---------- add near the top, after imports ----------
# ---------- end helper ----------

# ---------- add new scene factory ----------
def create_scene_10blocks(settle_steps: int = 200, allow_random_yaw: bool = False) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create a demo scene with 10 uniquely-colored blocks.
    Ensures blocks spawn upright on the table and are simulated for `settle_steps`.
    If allow_random_yaw=True some blocks may be spawned with 45deg yaw (for stability testing).
    """
    scene = _build_base_scene()

    plane = scene.add_entity(gs.morphs.Plane())

    # block half-height (meters)
    half_h = 0.02  # for box size (0.04,0.04,0.04)
    safe_z = half_h + 0.002  # slightly above table to avoid initial interpenetration

    pos = {
        "r": _rand_xy((0.85, -0.12, safe_z)),
        "g": _rand_xy((0.85,  0.12, safe_z)),
        "b": _rand_xy((0.65, -0.12, safe_z)),
        "y": _rand_xy((0.65,  0.12, safe_z)),
        "m": _rand_xy((0.45, -0.12, safe_z)),
        "c": _rand_xy((0.45,  0.12, safe_z)),
        "o": _rand_xy((0.25, -0.06, safe_z)),
        "p": _rand_xy((0.25,  0.18, safe_z)),
        "q": _rand_xy((0.35, -0.26, safe_z)),
        "s": _rand_xy((0.35,  0.26, safe_z)),
    }

    COLORS = {
        "r": (1.0, 0.0, 0.0),
        "g": (0.0, 1.0, 0.0),
        "b": (0.0, 0.0, 1.0),
        "y": (1.0, 1.0, 0.0),
        "m": (1.0, 0.0, 1.0),
        "c": (0.0, 1.0, 1.0),
        "o": (1.0, 0.5, 0.0),
        "p": (0.0, 0.6, 0.6),
        "q": (0.6, 0.2, 0.6),
        "s": (1.0, 0.7, 0.8),
    }

    # identity quaternion = no rotation (x,y,z,w)
    quat_identity = (0.0, 0.0, 0.0, 1.0)
    quat_45 = yaw_to_quat(45.0)

    def add_box(name, use_yaw45=False):
        quat = quat_45 if (allow_random_yaw and use_yaw45) else quat_identity
        # ensure z is exactly half height (safe_z used above)
        px, py, pz = pos[name]
        # use safe_z (small lift) to avoid interpenetration; physics settle will drop it
        entity = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(px, py, safe_z), quat=quat),
            surface=gs.options.surfaces.Plastic(color=COLORS[name]),
        )
        # If Genesis version doesn't set quat on construction, try setting afterward:
        try:
            entity.set_quat(quat)
        except Exception:
            pass
        return entity

    # decide which blocks, if any, get 45deg at spawn (only when allow_random_yaw True)
    yaw_map = {}
    if allow_random_yaw:
        # e.g., alternate yaw for every other block (or random.choice)
        for i, name in enumerate(["r","g","b","y","m","c","o","p","q","s"]):
            yaw_map[name] = (i % 2 == 0)  # even-indexed get 45 deg
    else:
        for name in ["r","g","b","y","m","c","o","p","q","s"]:
            yaw_map[name] = False

    blocks_state = {}
    for name in ["r","g","b","y","m","c","o","p","q","s"]:
        blocks_state[name] = add_box(name, use_yaw45=yaw_map[name])

    # add robot
    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # build and settle
    scene.build()

    # put robot in initial pose AFTER building so it doesn't collide while blocks settle
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))
    _elevate_robot_base(franka)

    # run physics steps to allow blocks to fall/settle onto the plane
    # If your Genesis API has scene.step() or scene.simulate(), use that. Many versions use scene.step().
    try:
        for _ in range(settle_steps):
            scene.step()
    except Exception:
        # fallback: some APIs use scene.simulate(dt, nsteps) or scene.advance()
        try:
            scene.simulate(settle_steps)
        except Exception:
            # if no explicit stepping, sleep briefly to allow viewer to render
            time.sleep(0.1)

    # final safety: ensure all blocks have z >= half height (if some are below, lift them slightly)
    for name, ent in blocks_state.items():
        try:
            p = ent.get_pos()
            if p[2] < half_h - 1e-4:
                # lift block to half height and let it settle again
                ent.set_pos((p[0], p[1], half_h + 0.001))
        except Exception:
            pass

    # one more short settle
    try:
        for _ in range(50):
            scene.step()
    except Exception:
        pass

    return scene, franka, blocks_state


def create_scene_3red_3green() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create scene with 3 red + 3 green blocks for Adjacent configuration
    
    Returns:
        scene, franka_adapter, blocks_state
    
    Blocks named: r1, r2, r3, g1, g2, g3
    """
    scene = _build_base_scene()
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Initial scattered positions
    positions_red = [
        (0.65, -0.20, 0.02),
        (0.65, 0.00, 0.02),
        (0.65, 0.20, 0.02)
    ]
    
    positions_green = [
        (0.50, -0.20, 0.02),
        (0.50, 0.00, 0.02),
        (0.50, 0.20, 0.02)
    ]
    
    blocks = {}
    
    # Create red blocks
    for i, pos in enumerate(positions_red):
        pos_noisy = _rand_xy(pos, noise=0.03)
        cube = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos_noisy),
            surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0))  # Red
        )
        blocks[f"r{i+1}"] = cube
    
    # Create green blocks
    for i, pos in enumerate(positions_green):
        pos_noisy = _rand_xy(pos, noise=0.03)
        cube = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos_noisy),
            surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0))  # Green
        )
        blocks[f"g{i+1}"] = cube
    
    # Add robot
    franka_raw = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
    )
    franka = RobotAdapter(franka_raw, scene)
    
    scene.build()
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))
    _elevate_robot_base(franka)
    
    print(f"[Scene] Created 3 red + 3 green blocks: r1-r3, g1-g3")
    
    return scene, franka, blocks

def create_scene_stacked() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """
    Create a scene with all 6 blocks pre-stacked in a tower.
    Useful for testing unstack operations.
    
    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.6, 0.0, 0.2))

    plane = scene.add_entity(gs.morphs.Plane())

    # Stack all blocks in one tower with some position randomization
    startx, starty, _ = _rand_xy((0.45, 0.0, 0.02), noise=0.2) 
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.02)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.06)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.10)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.14)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.18)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )
    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.22)),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)
    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state
