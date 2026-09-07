"""Interactive meta-test playback for a notebook.

Runs one scripted meta-test episode (Algorithm 10.3) with the trained model
*to its real end* (victim rescued, or victim health decays to 0), on the
full arena, and returns a `matplotlib.animation.FuncAnimation`. In a
notebook:

    from method.viz import play
    from IPython.display import HTML
    HTML(play(scenario="change", seed=2000).to_jshtml())

`.to_jshtml()` gives a JS player -- play / pause / step / a frame slider to
scrub the whole episode / loop / fps -- no ipywidgets needed. Each frame is
the VMAS render (whole map: agents, lidar rays, obstacles, victim) plus a
per-step overlay of each agent's mode / detector p / reset, the victim's
health, and the scripted-event marker.

Episode end (`utils/scenario.py`): `done` is true once the victim is
`rescued` OR its `health <= 0`. With `initial_health=100`, `decay_rate=0.2`
and `required_rescuers=1`, an un-rescued victim dies at step 500; a rescue
needs *one* agent within `rescue_range=0.5` of it.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from method.io import load_model
from method.training.meta_test import (
    MetaTestRunner,
    MetaTestConfig,
    RegimeChangeEvent,
    AdversarialAttackEvent,
)
from method.training.regime import Regime
from utils.scenario import Scenario


def _victim_state(env):
    """(min_health, all_rescued, done) for env 0, or (nan, False, False) if no survivals."""
    scen = env.scenario
    if not getattr(scen, "_survivals", None):
        return float("nan"), False, False
    hs = [float(s.health[0]) for s in scen._survivals]
    rescued = all(bool(s.rescued[0]) for s in scen._survivals)
    done = bool(scen.done()[0])
    return min(hs), rescued, done


def run_metatest_episode(
    trainer,
    detector,
    *,
    scenario: str = "change",
    seed: int = 2000,
    change_step: int = 16,
    attack_step: int = 16,
    until_done: bool = True,
    max_steps: int = 520,
    config_file: str = "world_config.yaml",
    fit_map: bool = True,
    viewer_zoom: float | None = 2.65,
    mt_config: MetaTestConfig | None = None,
):
    """Roll out one scripted episode to its end.

    `viewer_zoom` overrides the render camera's half-extent (shown extent is
    roughly `viewer_zoom ** 2`); 2.65 -> ~+/-7 units, a bit wider than the
    10x10 map so the walls and full arena are visible. `None` keeps the
    scenario's own `fit_map` value.

    Returns (frames, metas, extra, ev_by_step, title, summary) where
    `extra[t]` = {"health","rescued","done"} and `summary` is a dict of
    episode-level outcomes.
    """
    torch.manual_seed(seed)
    n = trainer.n_agents
    nominal = Regime(mass=0.9, linear_friction=0.03)
    shifted = Regime(mass=1.2, linear_friction=0.15)
    base = [RegimeChangeEvent(0, i, nominal) for i in range(n)]

    if scenario == "control":
        events, title = base, "meta-test: no change"
    elif scenario == "change":
        events = base + [RegimeChangeEvent(change_step, 0, shifted)]
        title = f"meta-test: agent-0 regime change @ {change_step}"
    elif scenario == "attack":
        events = base + [AdversarialAttackEvent(attack_step, 0, True, True)]
        title = f"meta-test: adversarial attack on agent 0 @ {attack_step}"
    else:
        raise ValueError(f"scenario must be control|change|attack, got {scenario!r}")

    runner = MetaTestRunner(
        trainer, detector,
        scenario_factory=lambda: Scenario(config_file=config_file, fit_map=fit_map),
        config=mt_config or MetaTestConfig(),
    )
    if viewer_zoom is not None:
        runner.env.scenario.viewer_zoom = viewer_zoom  # read fresh by the shared viewer each render()
    runner.schedule(events)
    runner.reset_state()

    ev_by_step: dict[int, list[str]] = {}
    for e in events:
        label = (f"regime change: agent {e.agent}" if isinstance(e, RegimeChangeEvent)
                 else f"attack ON: agent {e.agent}")
        ev_by_step.setdefault(e.step, []).append(label)

    frames, metas, extra = [], [], []
    total_reward = 0.0
    n_resets = 0
    end_step, end_reason = max_steps, "max_steps reached (episode not finished)"

    for t in range(max_steps):
        res = runner.step(render=True)
        if res.frame is None:
            raise RuntimeError("env.render returned None -- no rgb backend available?")
        health, rescued, done = _victim_state(runner.env)
        frames.append(np.asarray(res.frame))
        metas.append(res)
        extra.append({"health": health, "rescued": rescued, "done": done})
        total_reward += float(np.sum(res.rewards))
        n_resets += int(any(res.reset_fired))
        if until_done and done:
            end_step = t
            end_reason = "victim RESCUED" if rescued else "victim died (health <= 0)"
            break

    summary = {
        "steps": len(frames),
        "ended_at_step": end_step,
        "end_reason": end_reason,
        "victim_rescued": extra[-1]["rescued"],
        "victim_final_health": round(extra[-1]["health"], 2),
        "cumulative_reward": round(total_reward, 3),
        "detector_resets": n_resets,
    }
    return frames, metas, extra, ev_by_step, title, summary


def animate(frames, metas, extra, ev_by_step, *, title="", fps=6, frame_stride=None,
            max_player_frames=90):
    """Build a FuncAnimation. `frame_stride` subsamples frames so a long
    episode does not produce a giant .to_jshtml() blob."""
    plt.rcParams["animation.embed_limit"] = 60  # MB; default 20 is too small here
    n = len(frames)
    if frame_stride is None:
        frame_stride = max(1, -(-n // max_player_frames))  # ceil division
    idx = list(range(0, n, frame_stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)  # always show the final (episode-end) frame

    h, w = frames[0].shape[:2]
    txt_w = 4.8
    fig = plt.figure(figsize=(w / 110 + txt_w, max(h / 110, 3.2)), dpi=90)
    gs = fig.add_gridspec(1, 2, width_ratios=[w / 100, txt_w], wspace=0.02)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    for ax in (ax_img, ax_txt):
        ax.axis("off")

    def draw(k):
        i = idx[k]
        ax_img.clear(); ax_img.axis("off")
        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        ax_img.imshow(frames[i])
        m, x = metas[i], extra[i]
        lines = [title, "", f"step {i:>3} / {n - 1}"]
        if i in ev_by_step:
            lines += [f">>> {s} <<<" for s in ev_by_step[i]]
        lines.append("")
        for a in range(len(m.mode)):
            tag = "  <-- RESET" if m.reset_fired[a] else ""
            atk = "  [ATTACKED]" if m.active_attacks[a] else ""
            lines.append(f"agent {a}: {m.mode[a].upper():7s} p={m.detector_p[a]:.3f}{tag}{atk}")
        lines += ["",
                  f"victim health: {x['health']:5.1f} / 100",
                  f"rescued: {'YES' if x['rescued'] else 'no'}",
                  f"step reward: {np.sum(m.rewards):+.3f}"]
        if x["done"]:
            lines += ["", ">>> EPISODE ENDED <<<"]
        ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=10, transform=ax_txt.transAxes)

    anim = FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps, blit=False)
    plt.close(fig)
    return anim


class _ScriptedStep:
    """Minimal stand-in for meta_test.StepResult so `animate()` can render a
    scripted (non-policy) rollout unchanged."""
    def __init__(self, rewards, n_agents):
        self.rewards = rewards
        self.mode = ["goto-victim"] * n_agents
        self.detector_p = [float("nan")] * n_agents
        self.reset_fired = [False] * n_agents
        self.active_attacks = [False] * n_agents


def rescue_demo(
    *,
    seed: int = 0,
    which_agents: str = "all",          # "all" or "first"
    max_steps: int = 400,
    config_file: str = "world_config.yaml",
    viewer_zoom: float | None = 2.65,
    fps: int = 10,
    frame_stride: int | None = None,
):
    """Scripted rollout: each chosen agent drives straight at the victim's
    true position (NOT the trained policy). Shows the rescue path -- victim
    health, the `rescued` flip, the reward spike, and `done` firing early.

    Returns a FuncAnimation; prints an outcome summary.
    """
    import vmas
    torch.manual_seed(seed)
    scen = Scenario(config_file=config_file, fit_map=True)
    if viewer_zoom is not None:
        scen.viewer_zoom = viewer_zoom
    env = vmas.make_env(scenario=scen, num_envs=1, device="cpu",
                        continuous_actions=True, wrapper=None)
    obs = env.reset()
    n = len(env.agents)
    movers = range(n) if which_agents == "all" else [0]

    frames, metas, extra = [], [], []
    total_reward = 0.0
    end_step, end_reason = max_steps, "max_steps reached"
    for t in range(max_steps):
        vic = env.scenario._survivals[0].state.pos          # (1, 2)
        actions = []
        for i, ag in enumerate(env.agents):
            if i in movers:
                d = vic - ag.state.pos
                d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
                actions.append(d * ag.u_range)
            else:
                actions.append(torch.zeros(1, env.scenario.__dict__.get("action_dim", 2)))
        _obs, rewards, dones, _ = env.step(actions)
        h, rescued, done = _victim_state(env)
        frames.append(np.asarray(env.render(mode="rgb_array", env_index=0)))
        metas.append(_ScriptedStep([float(r[0]) for r in rewards], n))
        extra.append({"health": h, "rescued": rescued, "done": done})
        total_reward += float(sum(float(r[0]) for r in rewards))
        if done:
            end_step = t
            end_reason = "victim RESCUED" if rescued else "victim died (health <= 0)"
            break

    summary = {
        "steps": len(frames), "ended_at_step": end_step, "end_reason": end_reason,
        "victim_rescued": extra[-1]["rescued"],
        "victim_final_health": round(extra[-1]["health"], 2),
        "cumulative_reward": round(total_reward, 3),
    }
    print(f"--- scripted rescue demo (agents drive at victim; seed {seed}) ---")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    return animate(frames, metas, extra, {}, title="scripted rescue demo (not the trained policy)",
                   fps=fps, frame_stride=frame_stride)


def play(
    trainer=None,
    detector=None,
    *,
    scenario: str = "change",
    seed: int = 2000,
    change_step: int = 16,
    attack_step: int = 16,
    until_done: bool = True,
    max_steps: int = 520,
    fps: int = 6,
    frame_stride: int | None = None,
    config_file: str = "world_config.yaml",
    fit_map: bool = True,
    viewer_zoom: float | None = 2.65,
    mt_config: MetaTestConfig | None = None,
):
    """Load the model if not given, run one episode to its end, print an
    outcome summary, and return a FuncAnimation.

    Notebook: ``from IPython.display import HTML; HTML(play(...).to_jshtml())``
    """
    if trainer is None or detector is None:
        trainer, detector = load_model()
    frames, metas, extra, ev, title, summary = run_metatest_episode(
        trainer, detector, scenario=scenario, seed=seed, change_step=change_step,
        attack_step=attack_step, until_done=until_done, max_steps=max_steps,
        config_file=config_file, fit_map=fit_map, viewer_zoom=viewer_zoom,
        mt_config=mt_config,
    )
    print(f"--- {title} (seed {seed}) ---")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    return animate(frames, metas, extra, ev, title=title, fps=fps, frame_stride=frame_stride)
