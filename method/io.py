"""Load a trained model back from disk.

A training run (`scripts/train.py` / `scripts/train_phase2.py`) writes two
artifacts under `runs/<timestamp>/`:

  * ``phase1_checkpoint_<iter>.pt`` -- a dict of module ``state_dict``s
    (the ``CHECKPOINT_KEYS`` in ``scripts/train.py``) plus the two
    ``log_alpha`` tensors;
  * ``phase2_detector.pt`` -- a bare ``state_dict`` for the InDiD
    ``CausalLocalWindowTransformer``.

This module rebuilds the live objects from those files -- frozen, in eval
mode -- so meta-test / video / evaluation code has one place to load from
instead of re-deriving the key list each time.

Typical use:

    from method.io import load_model
    trainer, detector = load_model()                       # newest run
    trainer, detector = load_model(
        "runs/20260906_162246/phase1_checkpoint_2999.pt",
        "runs/20260906_182413_phase2only/phase2_detector.pt")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.scenario import Scenario
from method.training.phase1 import Phase1Trainer, Phase1Config
from method.training.phase2 import Phase2Config
from method.detector.indid import CausalLocalWindowTransformer

# Modules a Phase 1 checkpoint carries -- must match scripts/train.py.
PHASE1_MODULE_KEYS = (
    "flow", "flow_momentum", "budget_encoder", "budget_decoder",
    "exploration_policies", "exploration_critics", "exploration_critics_target",
    "execution_policies", "execution_critics", "execution_critics_target",
    "mixer", "mixer_target",
)
PHASE1_TENSOR_KEYS = ("log_alpha_exp", "log_alpha_exe")


# --------------------------------------------------------------------------
# run-dir discovery
# --------------------------------------------------------------------------
def latest_run_with(filename: str, runs_dir: str | Path = "runs") -> Path | None:
    """Newest ``runs/<...>/`` dir that actually contains ``filename``.

    Skips the phase2-only / smoke dirs that lack a ``phase1_log.csv`` (etc.),
    so ``latest_run_with("phase1_log.csv")`` and
    ``latest_run_with("phase2_detector.pt")`` can resolve to different dirs.
    """
    hits = [d for d in Path(runs_dir).glob("*/") if (d / filename).is_file()]
    return max(hits, key=lambda d: (d / filename).stat().st_mtime) if hits else None


def latest_phase1_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    cks = sorted(run_dir.glob("phase1_checkpoint_*.pt"),
                 key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not cks:
        raise FileNotFoundError(f"no phase1_checkpoint_*.pt in {run_dir}")
    return cks[-1]


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------
def _phase1_config_for(ckpt_path: Path, override: Phase1Config | None) -> tuple[Phase1Config, str]:
    cfg = override or Phase1Config()
    cfg_file = "world_config.yaml"
    args_json = ckpt_path.parent / "args.json"
    if args_json.is_file():
        a = json.loads(args_json.read_text())
        cfg_file = a.get("config_file", cfg_file)
        if override is None:
            cfg = Phase1Config(n_envs=a.get("n_envs", cfg.n_envs),
                               horizon=a.get("horizon", cfg.horizon))
    return cfg, cfg_file


def load_phase1(ckpt_path: str | Path, *, scenario_factory=None,
                config: Phase1Config | None = None, device=None,
                freeze: bool = True) -> Phase1Trainer:
    """Rebuild a ``Phase1Trainer`` and load ``ckpt_path`` into it.

    ``config`` / ``scenario_factory`` default to the values recorded in the
    checkpoint dir's ``args.json`` (falling back to ``Phase1Config()`` and
    ``world_config.yaml``). With ``freeze=True`` every module is put in
    ``eval()`` and has ``requires_grad_(False)``.
    """
    ckpt_path = Path(ckpt_path)
    config, cfg_file = _phase1_config_for(ckpt_path, config)
    if scenario_factory is None:
        scenario_factory = lambda: Scenario(config_file=cfg_file)

    trainer = Phase1Trainer(scenario_factory, config, device=device)
    state = torch.load(ckpt_path, map_location=trainer.device)

    missing = []
    for k in PHASE1_MODULE_KEYS:
        if k in state:
            getattr(trainer, k).load_state_dict(state[k])
        elif k.endswith("_target") and k[:-7] in state:
            getattr(trainer, k).load_state_dict(state[k[:-7]])   # copy online -> target
            missing.append(k + " (copied from online)")
        else:
            missing.append(k)
    for k in PHASE1_TENSOR_KEYS:
        if k in state:
            with torch.no_grad():
                getattr(trainer, k).copy_(state[k])
    if missing:
        print(f"[load_phase1] not in checkpoint, used fallback: {missing}")

    if freeze:
        for k in PHASE1_MODULE_KEYS:
            m = getattr(trainer, k)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    return trainer


# --------------------------------------------------------------------------
# Phase 2 detector
# --------------------------------------------------------------------------
def _detector_dims_from_state(sd: dict) -> dict:
    """Infer architecture dims from the detector state_dict so the rebuilt
    module always matches the file, even if Phase2Config drifted."""
    d_model, code_dim = sd["input_proj.weight"].shape
    return dict(
        code_dim=int(code_dim),
        d_model=int(d_model),
        max_len=int(sd["pos_embedding"].shape[1]),
        n_layers=1 + max(int(k.split(".")[2]) for k in sd
                         if k.startswith("transformer.layers.")),
        dim_feedforward=int(sd["transformer.layers.0.linear1.weight"].shape[0]),
    )


def load_detector(detector_path: str | Path, *, phase1_trainer: Phase1Trainer | None = None,
                  config: Phase2Config | None = None, device=None,
                  freeze: bool = True) -> CausalLocalWindowTransformer:
    """Rebuild the InDiD detector and load ``detector_path``.

    Shape-dependent dims (``code_dim``, ``d_model``, ``n_layers``,
    ``dim_feedforward``, ``max_len``) are read from the state_dict itself;
    ``n_heads`` and ``window`` come from ``config`` (``Phase2Config()`` by
    default) since they leave no trace in the weights.
    """
    device = device or (phase1_trainer.device if phase1_trainer is not None else "cpu")
    sd = torch.load(detector_path, map_location=device)
    dims = _detector_dims_from_state(sd)
    cfg = config or Phase2Config()

    if phase1_trainer is not None and dims["code_dim"] != phase1_trainer.code_dim:
        print(f"[load_detector] WARNING: detector code_dim={dims['code_dim']} != "
              f"phase1 code_dim={phase1_trainer.code_dim} -- checkpoints may not match")

    det = CausalLocalWindowTransformer(
        code_dim=dims["code_dim"], d_model=dims["d_model"], n_heads=cfg.n_heads,
        n_layers=dims["n_layers"], window=cfg.window,
        dim_feedforward=dims["dim_feedforward"], max_len=dims["max_len"],
    ).to(device)
    det.load_state_dict(sd)
    if freeze:
        det.eval()
        for p in det.parameters():
            p.requires_grad_(False)
    return det


# --------------------------------------------------------------------------
# both halves
# --------------------------------------------------------------------------
def load_model(phase1_ckpt: str | Path | None = None,
               detector_path: str | Path | None = None, *,
               scenario_factory=None, config: Phase1Config | None = None,
               detector_config: Phase2Config | None = None, device=None):
    """Load the whole trained model.

    Any path left as ``None`` is auto-resolved to the newest ``runs/<...>/``
    dir that has the matching artifact (``phase1_log.csv`` for Phase 1,
    ``phase2_detector.pt`` for the detector).

    Returns ``(phase1_trainer, detector)``; ``detector`` is ``None`` if no
    ``phase2_detector.pt`` exists anywhere under ``runs/``.
    """
    if phase1_ckpt is None:
        run = latest_run_with("phase1_log.csv")
        if run is None:
            raise FileNotFoundError("no run dir with phase1_log.csv under runs/")
        phase1_ckpt = latest_phase1_checkpoint(run)
        print(f"[load_model] phase1: {phase1_ckpt}")
    trainer = load_phase1(phase1_ckpt, scenario_factory=scenario_factory,
                          config=config, device=device)

    if detector_path is None:
        run = latest_run_with("phase2_detector.pt")
        detector_path = (run / "phase2_detector.pt") if run is not None else None
        if detector_path is not None:
            print(f"[load_model] detector: {detector_path}")
    detector = (load_detector(detector_path, phase1_trainer=trainer, config=detector_config)
                if detector_path is not None else None)
    if detector is None:
        print("[load_model] no phase2_detector.pt found -- detector is None")
    return trainer, detector
