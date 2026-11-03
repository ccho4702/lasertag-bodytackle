"""Per-robot I/O profiles: how labels are read from a session, what the policy outputs, and how
that output is turned into a command the robot API accepts.

Paper (arXiv 2603.11980) roles: Team I "ally" = shooter, Team II "enemy" = escapes.
    chaser : Misty  (shooter)  — command = (linear %, angular %)          positive angular = left
    evader : Cozmo  (escapes)  — command = (left, right) wheel speed mm/s

The policy always predicts a body velocity a = [v, ω] (paper: "velocity command"):
    chaser : v = linear_velocity_percent, ω = angular_velocity_percent          (units: %)
    evader : v = (L + R) / 2,              ω = (R - L) / 2                       (units: mm/s;
             ω > 0 = counter-clockwise = left, same sign convention as Misty)
`Robot.to_command(a)` maps it back to what the robot API takes (inverse for Cozmo: L = v - ω,
R = v + ω), clipped to the robot limits.

Heatmap channel names are relative to the robot itself: 'enemy' = the opponent, 'ally' = a
teammate. The YOLO checkpoint of each robot uses exactly these class names.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _first(*paths):
    """first existing path (release layout ../checkpoints/yolo/ first, then the training-server layout)"""
    return next((p for p in paths if os.path.exists(p)), paths[0])


@dataclass(frozen=True)
class Robot:
    name: str
    platform: str
    role: str                                   # 'shooter' (paper Team I) | 'evader' (Team II)
    units: str
    action_names: tuple                         # names of the policy's 2 outputs [v, ω]
    levels: tuple                               # discrete command values per dim seen in the demos
    commands: dict                              # keyboard command name -> (v, ω)
    cmd_limit: tuple                            # |v|, |ω| clip at deployment (robot units)
    yolo: str                                   # detector checkpoint (own-camera view)
    yolo_imgsz: int
    fps: float                                  # camera / label rate of the demos
    hfov_deg: float                             # horizontal FOV of the image as stored (assumed)
    width_m: dict = field(default_factory=dict)  # physical width per YOLO class (for D = f·W/w)
    task: str = ''                              # VLA instruction

    @property
    def act_scale(self):
        """Per-dim normalisation (max |level|) — the network works in [-1, 1]."""
        return np.array([max(abs(x) for x in lv) for lv in self.levels], np.float32)

    # ── labels ──────────────────────────────────────────────────────────────
    def load_actions(self, session_dir, source='requested'):
        """(N, 2) float32 [v, ω] per row of labels.csv, robot units.

        source='requested' : the command the operator issued at this frame (paper: the action the
                             expert chose for this observation). 'executed': the command the robot
                             had acknowledged (lags the operator by the HTTP round trip).
        """
        df = pd.read_csv(os.path.join(session_dir, 'labels.csv'))
        if self.platform == 'misty':
            if source == 'requested' and 'requested_command' in df:
                return np.array([self.commands[c] for c in df['requested_command']], np.float32)
            return df[['linear_velocity_percent', 'angular_velocity_percent']].to_numpy(np.float32)
        # cozmo: the CSV already stores the PC-issued wheel command at image receipt
        L = df['left_wheel_mmps'].to_numpy(np.float32)
        R = df['right_wheel_mmps'].to_numpy(np.float32)
        return np.stack([(L + R) / 2, (R - L) / 2], 1)

    # ── deployment ──────────────────────────────────────────────────────────
    def snap(self, a):
        """Nearest demonstrated command level per dim."""
        a = np.asarray(a, np.float32)
        return np.array([min(lv, key=lambda x: abs(x - a[i])) for i, lv in enumerate(self.levels)],
                        np.float32)

    def to_command(self, a, snap=False):
        """Policy output [v, ω] (robot units) → kwargs for the robot API."""
        a = self.snap(a) if snap else np.asarray(a, np.float32)
        v, w = (float(np.clip(a[i], -self.cmd_limit[i], self.cmd_limit[i])) for i in range(2))
        if self.platform == 'misty':
            # Misty REST /api/drive  {"LinearVelocity": %, "AngularVelocity": %}
            return {'LinearVelocity': v, 'AngularVelocity': w}
        # Cozmo SDK robot.drive_wheels(l_wheel_speed, r_wheel_speed)  [mm/s]
        return {'l_wheel_speed': v - w, 'r_wheel_speed': v + w}


ROBOTS = {
    'chaser': Robot(
        name='chaser', platform='misty', role='shooter', units='%',
        action_names=('linear_velocity_percent', 'angular_velocity_percent'),
        levels=((-25.0, 0.0, 25.0), (-25.0, 0.0, 30.0)),
        commands={'stop': (0, 0), 'forward': (25, 0), 'backward': (-25, 0), 'left': (0, 30),
                  'right': (0, -25), 'forward_left': (25, 30), 'forward_right': (25, -25),
                  'backward_left': (-25, 30), 'backward_right': (-25, -25)},
        cmd_limit=(100.0, 100.0),
        yolo=_first(os.path.join(HERE, '..', 'checkpoints', 'yolo', 'misty_yolo26s_v1.pt'),
                    os.path.join(HERE, '..', 'yolo_extracted', 'release_v1', 'checkpoint', 'misty_yolo26s_v1.pt')),
        yolo_imgsz=640, fps=9.8,
        hfov_deg=55.0,                              # ASSUMED (portrait stream; measure & update)
        width_m={'enemy': 0.055, 'ally': 0.25, 'obstacle': 0.30},   # enemy = Cozmo, ally = Misty
        task='Drive into the enemy robot and bump it with your body while avoiding obstacles.'),
    'evader': Robot(
        name='evader', platform='cozmo', role='evader', units='mm/s',
        action_names=('linear_mmps', 'angular_mmps'),
        levels=((-90.0, 0.0, 90.0), (-60.0, 0.0, 60.0)),
        commands={'stop': (0, 0), 'forward': (90, 0), 'backward': (-90, 0), 'left': (0, 60),
                  'right': (0, -60), 'forward_left': (90, 60), 'forward_right': (90, -60),
                  'backward_left': (-90, 60), 'backward_right': (-90, -60)},
        cmd_limit=(220.0, 220.0),
        yolo=_first(os.path.join(HERE, '..', 'checkpoints', 'yolo', 'cozmo_yolo_v1.pt'),
                    os.path.join(HERE, 'runs', 'detect', 'runs_det', 'cozmo_v1', 'weights', 'best.pt')),
        yolo_imgsz=320, fps=8.4,
        hfov_deg=58.0,                              # Cozmo camera (SDK fov_x ≈ 58°)
        width_m={'enemy': 0.25, 'ally': 0.055, 'obstacle': 0.30},   # enemy = Misty, ally = Cozmo
        task='Run away from the chasing robot so that it cannot touch you; use obstacles to hide when you can.'),
}


def get(name):
    if isinstance(name, Robot):
        return name
    if name not in ROBOTS:
        raise ValueError(f'unknown robot {name!r}; choose from {list(ROBOTS)}')
    return ROBOTS[name]
