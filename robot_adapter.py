"""RobotAdapter: a thin adapter around a Genesis robot entity.

Purpose:
- Normalize and centralize robot API access so higher-level modules can
  call a stable interface. By default the adapter forwards unknown
  attributes to the underlying genesis robot, so existing code that
  expects the raw robot still works when passed an adapter.

Usage:
    adapter = RobotAdapter(raw_robot, scene)
    adapter.get_qpos()  # forwarded
    adapter.set_qpos(q) # forwarded

This is intentionally minimal: it implements __getattr__ forwarding and
adds an explicit .robot property to access the raw object when needed.
"""
from typing import Any


class RobotAdapter:
    def __init__(self, robot: Any, scene: Any = None):
        """Wrap a genesis robot entity.

        Args:
            robot: the raw genesis robot entity (e.g., returned from scene.add_entity)
            scene: optional scene reference (some callers use scene alongside robot)
        """
        self.robot = robot
        self.scene = scene

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the underlying robot.

        This makes the adapter nearly transparent by default so existing
        code can keep using the usual robot API.
        """
        return getattr(self.robot, name)

    # Optional: convenience explicit aliases (delegation examples). Keep these
    # so callers can rely on these names being present even if we later
    # enrich/transform arguments.
    def get_pos(self):
        return self.robot.get_pos()

    def set_pos(self, pos):
        return self.robot.set_pos(pos)

    def get_qpos(self):
        return self.robot.get_qpos()

    def set_qpos(self, qpos):
        return self.robot.set_qpos(qpos)

    def control_dofs_position(self, *args, **kwargs):
        return self.robot.control_dofs_position(*args, **kwargs)

    def control_dofs_force(self, *args, **kwargs):
        return self.robot.control_dofs_force(*args, **kwargs)

    def get_link(self, *args, **kwargs):
        return self.robot.get_link(*args, **kwargs)

    def inverse_kinematics(self, *args, **kwargs):
        return self.robot.inverse_kinematics(*args, **kwargs)

    def detect_collision(self, *args, **kwargs):
        return self.robot.detect_collision(*args, **kwargs)

    # expose the raw object if callers need direct access
    @property
    def raw(self):
        return self.robot
