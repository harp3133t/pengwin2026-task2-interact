"""Deployment-contract tests for the Task 2 v3.5 always-expert candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _entrypoint_module():
    path = ROOT / "inference" / "inference.py"
    spec = importlib.util.spec_from_file_location("task2_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dockerfile_selects_all_v35_experts_at_t075():
    dockerfile = (ROOT / "Dockerfile").read_text()
    expected = (
        "PENGWIN_DS539_TRAINER=PengwinTrainerSTUNetBaseAnatomyV301",
        "PENGWIN_DS539_FOLD=0",
        "PENGWIN_DS538_TRAINER_SACRUM="
        "PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal",
        "PENGWIN_DS538_TRAINER_HIP="
        "PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal",
        "PENGWIN_DS538_TRAINER_FEMUR="
        "PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal",
        "PENGWIN_DS538_FOLD=0",
        "PENGWIN_DS538_OUT_CH=13",
        "PENGWIN_AFFINITY_DECODE=1",
        "PENGWIN_AGGLO_T=0.75",
        "PENGWIN_CLICK_INJECT=0",
    )
    for item in expected:
        assert item in dockerfile


def test_pipeline_maps_both_hips_to_the_shared_expert():
    pipeline = (ROOT / "inference" / "task1_pipeline.py").read_text()
    assert '"Sacrum": os.environ.get("PENGWIN_DS538_TRAINER_SACRUM", "")' in pipeline
    assert '"LeftHip": os.environ.get("PENGWIN_DS538_TRAINER_HIP", "")' in pipeline
    assert '"RightHip": os.environ.get("PENGWIN_DS538_TRAINER_HIP", "")' in pipeline
    assert '"Femur": os.environ.get("PENGWIN_DS538_TRAINER_FEMUR", "")' in pipeline
    assert "desired_ds538_trainer = DS538_EXPERT_TRAINERS.get(" in pipeline
    assert "torch.cuda.empty_cache()" in pipeline


def test_v35_checkpoint_discovery_names_are_vendored():
    core = (ROOT / "code_task1" / "core.py").read_text()
    for name in (
        "PengwinTrainerSTUNetBaseAffinityV308DeployedVal",
        "PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal",
        "PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal",
        "PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal",
    ):
        assert f"class {name}(" in core


def test_click_routing_still_forces_exact_anatomies(tmp_path):
    module = _entrypoint_module()
    clicks = tmp_path / "peripelvic-fragment-clicks.json"
    clicks.write_text(
        """{
          "name": "test",
          "points": [
            {"name": "Sacrum Point 1", "point": [1, 2, 3]},
            {"name": "Left Hipbone Point 1", "point": [4, 5, 6]},
            {"name": "Right Hipbone Point 1", "point": [7, 8, 9]}
          ]
        }"""
    )
    routing = module.route_from_clicks(module.load_clicks(clicks))
    assert module.anatomies_from_routing(routing) == (
        "Sacrum",
        "LeftHip",
        "RightHip",
    )
