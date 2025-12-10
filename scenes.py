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
def create_scene_10blocks() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create a 10-block demo scene with blocks closer together for tall tower building.
    Blocks are spread in a compact workspace but not too far apart.
    Returns scene, franka_adapter, blocks_state.
    """
    scene = _build_base_scene()
    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # MORE SPREAD OUT LAYOUT - Blocks further apart but still reachable
    # Workspace: x=[0.30-0.70], y=[-0.25, +0.25]
    # Each block has more space around it to avoid collisions
    
    # Row 1 (back row, x=0.70)
    posR = _rand_xy((0.70, -0.15, 0.02), noise=0.02)
    posG = _rand_xy((0.70,  0.00, 0.02), noise=0.02)
    posB = _rand_xy((0.70,  0.15, 0.02), noise=0.02)
    
    # Row 2 (middle-back, x=0.60)
    posY = _rand_xy((0.60, -0.25, 0.02), noise=0.02)
    posM = _rand_xy((0.60,  0.00, 0.02), noise=0.02)
    posC = _rand_xy((0.60,  0.25, 0.02), noise=0.02)
    
    # Row 3 (middle-front, x=0.50)
    posO = _rand_xy((0.50, -0.15, 0.02), noise=0.02)
    posP = _rand_xy((0.50,  0.15, 0.02), noise=0.02)
    
    # Row 4 (front row, x=0.40)
    posQ = _rand_xy((0.40, -0.08, 0.02), noise=0.02)
    posS = _rand_xy((0.40,  0.08, 0.02), noise=0.02)

    # Create colored blocks (no explicit quat/orientation)
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 1.0)),
    )
    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 1.0)),
    )
    cubeO = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.5, 0.0)),  # Orange
    )
    cubeP = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posP),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.6, 0.6)),  # Teal
    )
    cubeQ = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posQ),
        surface=gs.options.surfaces.Plastic(color=(0.6, 0.2, 0.6)),  # Purple
    )
    cubeS = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posS),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.7, 0.8)),  # Pink
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

    blocks_state: Dict[str, Any] = {
        "r": cubeR, "g": cubeG, "b": cubeB,
        "y": cubeY, "m": cubeM, "c": cubeC,
        "o": cubeO, "p": cubeP, "q": cubeQ, "s": cubeS
    }

    print("[Scene] Created 10-block scene (spread out layout)")
    print(f"[Scene] Workspace center: x=0.50m, y=0.0m")
    print(f"[Scene] Spread: x=[0.40-0.70], y=[-0.25,+0.25]")
    print(f"[Scene] Min block spacing: ~10-15cm")

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
