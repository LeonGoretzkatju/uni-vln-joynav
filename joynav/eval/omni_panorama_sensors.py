"""Register left/right RGB camera sensors for OmniNav-style 3-view evaluation.

Importing this module adds two extra RGB sensors to Habitat's ``sim_sensors``
config group so ``configs/omni_r2r.yaml`` can give the Omni evaluator REAL
left / front / right current observations, matching OmniNav's
``add_frame([observations['left'], observations['right'], observations['front']])``
in ``referenceomni/OmniNav/infer_r2r_rxr/agent/waypoint_agent.py``.

Why a custom config is needed: the stock ``rgb_sensor`` node
(``HabitatSimRGBSensorConfig``) has no ``uuid`` field, and the
``HabitatSimRGBSensor`` class hard-defaults its uuid to ``"rgb"`` — so three
plain ``rgb_sensor`` entries would all collide on uuid ``"rgb"``. Habitat reads
``Sensor.uuid`` from ``config.uuid`` when present (see
``habitat/core/simulator.py``: ``if "uuid" in self.config: self.uuid = self.config.uuid``),
so a subclass that declares a distinct ``uuid`` yields a distinct observation key.

This mirrors how habitat-lab itself defines ``head_rgb_sensor`` / ``third_rgb_sensor``.

NOTE on train/eval consistency: only evaluate with these sensors (i.e. with
``configs/omni_r2r.yaml`` + ``omni_use_panorama=True``) when the checkpoint was
ALSO trained on real left/front/right current views. The default JD-VLN training
data is front-only (the generator replicates the front frame into all three
current slots), so for a front-only checkpoint use ``configs/vln_r2r.yaml``
(front replicated) to keep the train/eval input distribution aligned.
"""

from dataclasses import dataclass

from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig
from hydra.core.config_store import ConfigStore


@dataclass
class OmniLeftRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "rgb_left"


@dataclass
class OmniRightRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "rgb_right"


_cs = ConfigStore.instance()
_cs.store(
    group="habitat/simulator/sim_sensors",
    name="rgb_left_sensor",
    node=OmniLeftRGBSensorConfig,
)
_cs.store(
    group="habitat/simulator/sim_sensors",
    name="rgb_right_sensor",
    node=OmniRightRGBSensorConfig,
)
