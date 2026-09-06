# Two-Phase Exchangeable-Context CCM–FMASAC — Implementation Spec

Source of truth: `ch-proposed.tex` in the thesis repo (`v2/ch-proposed.tex`), chapter
"Proposed Method: Two-Phase Exchangeable-Context CCM–FMASAC". This document distills that
chapter into an implementation-ready spec — no thesis prose, all equations and algorithms
kept exact. If anything here and the thesis chapter disagree, the thesis chapter is
authoritative; re-pull this file from it.

## 0. One-paragraph summary

A cooperative MARL team (FMASAC backbone) where each agent runs two policies: an
**exploration** policy driven by a closed-form information-gain reward from a per-agent
**exchangeable-process (BRUNO-style) context posterior**, and an **execution** policy/critic
conditioned on that context posterior *and* a team-shared **perturbation-budget posterior**
(both as full distributions, VariBAD-style, not samples). The execution critic is trained
against a worst-case one-step (M3DDPG/MAAL) perturbation of teammate actions. Training is
two-phase: Phase 1 trains encoders + both policies + critic on episodes with a known
per-agent task switch; Phase 2 freezes everything and trains a supervised (InDiD-style)
change-point detector on the same episodes. At deployment the detector resets an agent's
context posterior and hands it back to exploration when it flags a regime change.

## 1. Requirements this design satisfies (R1–R7)

| Req | What it demands | Where it's met |
|---|---|---|
| R1 | Robustness without a nominal-performance tax | Execution actor/critic condition on the full budget posterior → conservatism collapses to zero when inferred budget is zero (§4, §6) |
| R2 | Perturb the teammate-action channel, contained to execution | MAAL perturbation only touches execution stream (§4) |
| R3 | Track unbounded drift, don't assume it bounded | Phase-2 detector flags change online, triggers reset (§8) |
| R4 | Amortized task inference inside a team learner | Context encoder sits on FMASAC backbone (§3, §6) |
| R5 | Per-agent, not team-wide, change | Task sampling, posterior, detector, reset are all per-agent (§2, §3, §9) |
| R6 | Separate "world changed" from "adversary is strong" | $z_i$ and $\rho$ inferred as disentangled quantities (§7) |
| R7 | Evaluate both properties on the same runs | Search-and-Rescue env with independent dials (§11) |

## 2. Notation and per-episode sampling

Cooperative Dec-POMDP, agents $i=1,\ldots,n$, local histories $\tau_i$, joint history $\tau$,
global state $s$ (centralized training only). Each training episode, horizon $H$:

```
for each agent i:
    mu_1i, mu_2i ~ p(mu)                      # two local env regimes
    vartheta_i ~ GEOM(p_sw)
    vartheta_i_tilde = min(vartheta_i, H)     # per-agent switch time
epsilon_a ~ p_eps                             # ONE team-wide perturbation budget draw
```

Agent $i$'s active regime: $\mu_i(t) = \mu_{1,i}$ for $t < \tilde\vartheta_i$, else
$\mu_{2,i}$. So the environment can change for **one agent while others stay stationary**.
Agents with $\tilde\vartheta_i = H$ are valid no-switch episodes (needed as Phase-2
stationary negatives). $\epsilon_a$ is a single scalar shared by the whole team, drawn
independently of every $\mu_i$, and is the quantity the adversary's ball must respect:
$\|\check{\mathbf a}^{-i} - \hat{\mathbf a}^{-i}\| \le \epsilon_a$.

**Belief object**, per agent $i$:
```
b_i = { (mu_z_i, Sigma_z_i),   # per-agent context posterior
        (mu_rho, Sigma_rho) }  # team-shared budget posterior (same object for every i)
```
Policies:
- exploration: `pi_explore_i(a_i | o_i)` — **not** conditioned on `b_i`
- execution: `pi_execute_i(a_i | o_i, b_i)` — conditioned on the **whole distribution**, not a sample

## 3. Exchangeable context encoder (per-agent)

Invertible flow $f_\phi$ (shared params, per-agent input) maps each transition to a code:
```
x_i,t = (o_i,t, a_i,t)              # flow input
y_i,t = (r_i,t, o_i,t+1)            # flow target
c_i,t = f_phi(y_i,t | x_i,t)        # latent code in R^d
```
Within a regime, $\{c_{i,t}\}$ is modeled as an exchangeable Gaussian process:
$\Sigma_{tt}=\nu$, $\Sigma_{tt'}=\kappa$ ($t\ne t'$). By de Finetti this is equivalent to
$c_{i,t}$ conditionally i.i.d. given a latent $z_i$ whose posterior has an **exact O(1)
recursive conjugate update**:

```
beta_(t-1) = kappa / (nu + kappa*(t-2))
mu_i,t     = (1 - beta_(t-1)) * mu_i,(t-1)    + beta_(t-1) * c_i,(t-1)
sigma2_i,t = (1 - beta_(t-1)) * (sigma2_i,(t-1) - nu + kappa)
```
Init: `mu_i,0 = 0`, `sigma2_i,0 = kappa`. **Reset to this prior** whenever a change point is
flagged for agent $i$ (ground-truth $\tilde\vartheta_i$ in Phase 1; detector's own signal at
test time) — never let $z_i$ be estimated across a straddled regime boundary.

$(\mu_{i,t}, \sigma_{i,t}^2)$ = the context half of $b_i$. A sample
$z_{i,t}\sim\mathcal N(\mu_{i,t},\sigma_{i,t}^2 I)$ is drawn **only** inside the
reconstruction loss below — never used for policy conditioning.

### Encoder training loss (per agent, summed over agents)

```
L_enc = L_ELBO - lambda_CPC * L_CPC

L_ELBO = -E_{q_phi(z_i|tau_i)}[ log p_psi(y_i,1:T | z_i, x_i,1:T) ]
         + KL( q_phi(z_i|tau_i) || p(z) )

L_CPC  = log( exp(f(z_q, z_pos)) / sum_j exp(f(z_q, z_j)) )     # InfoNCE
       where f(z, z') = z^T W z'   (W learnable)
```
- $p_\psi$ (decoder): reconstructs $y_{i,t}=(r_{i,t}, o_{i,t+1})$ from $z_i$ — the VariBAD
  target. This is what forces $z_i$ to be predictive of return/dynamics, not just
  distributionally exchangeable-shaped.
- CPC term: $z_q = e_\phi(b_n^q)$ online embedding of a segment from the agent's current
  regime; $z_{\mathrm{pos}} = \bar e_\phi(b_n^k)$ momentum-encoder embedding of a *different*
  segment from the *same* regime; $\{z_j\}$ = momentum-encoder embeddings of other-regime
  segments (pre-/post-switch halves of the same episode count as different regimes).
  Momentum encoder update: $\bar\phi \leftarrow \tau_m\bar\phi + (1-\tau_m)\phi$.
- Input $x_{i,t}$ is **always the clean observation** (never adversary-touched) — required
  for the disentanglement guarantee in §7.

**$\lambda_{\mathrm{CPC}}$ knob:** set to 0 if the non-stationary env parameters vary
continuously/unimodally (ELBO alone suffices — nearby params → nearby $z$). Set $>0$
(default) when the non-stationarity is multimodal (a few qualitatively distinct regimes,
e.g. icy/wet/dry) — the ELBO alone blurs modes together; InfoNCE keeps them
mode-discriminative. Two-task episodes here are mode-structured, so default is
$\lambda_{\mathrm{CPC}} > 0$.

## 4. Information-gain exploration reward

Because the posterior is exactly Gaussian, one-step info gain is closed-form:
```
r_aux_i,t = I(z_i; c_i,t | c_i,1:t-1) = 0.5 * log( sigma2_i,(t-1) / sigma2_i,t )
r_e_i,t   = r_env_i,t + alpha_exp * r_aux_i,t
```
Each `pi_explore_i` updated independently by SAC on `r_e_i` through its own critic
`Q_exp_i` and its own entropy temperature `alpha_exp` — **no mixing network** (exploration
critics never feed `Q_tot`, see §5 invariant 1).

Fallback (if the exchangeable-Gaussian assumption breaks down in practice): CCM's InfoNCE
bound-difference estimator `L_upper - L_lower`, drop-in replacement for `r_aux_i,t`.

## 5. Two structural invariants (enforce everywhere)

1. **Critic separation.** `Q_tot` is built ONLY from execution critics `{Q_exe_i}`.
   Exploration critics `{Q_exp_i}` never contribute to `Q_tot` (different, incommensurable
   reward signal).
2. **Perturbation scoping.** The MAAL perturbation `check_a^{-i}` and the budget latent `rho`
   touch ONLY the execution stream (execution actor, its critic, its critic target).
   Exploration policy and `Q_exp_i` never see `check_a^{-i}` or `rho`.

## 6. Adversarial perturbation branch (M3DDPG/MAAL)

Robust critic target (conceptual, intractable inner min):
```
Q_rob_i = min_{||a^-i - a_hat^-i|| <= eps_a}  Q_tot(tau, (a_i, a^-i), s, b)
    where a_hat_j ~ pi_execute_j(.|o_j, b_j)  for all j
```
Approximated by **one projected gradient step** (MAAL) on the *target* critic:
```
check_a^-i = Proj_{eps_a}[ a_hat^-i - alpha_adv * grad_{a^-i} Q_tot_target(tau, (a_i, a_hat^-i), s, b) ]
```
`Proj_{eps_a}` = projection onto the $\epsilon_a$-ball. No separate adversary network — this
is a closed-form one-step quantity computed at update time (unlike RARL). The perturbed joint
action `(a_i, check_a^-i)` replaces the nominal one in both the actor gradient and the critic
TD target.

### Perturbation-budget encoder (single, team-shared)

Structured like the context encoder but pools **execution** histories across all agents:
```
x_exe = (x_1_exe, ..., x_n_exe)
(mu_rho, Sigma_rho) = q_eta(x_exe)              # single Gaussian posterior, team-shared
```
Decoder reconstructs the **radius** $\epsilon_a$, not the realized one-step displacement:
```
rho ~ q_eta
eps_hat = p_eta(rho)
L_rho = KL(q_eta(rho|x_exe) || p(rho)) + (eps_a - eps_hat)^2
```
Rationale: $\epsilon_a$ is the stationary quantity defining the threat model; the realized
MAAL displacement is a noisy, gradient/critic-dependent by-product — wrong reconstruction
target. At deployment (no adversary run), `q_eta` infers the *effective* perturbation level
from trajectory statistics — same kind of well-posed inference as the context encoder
inferring the task.

Actor/critic conditioning (VariBAD-style, always on the full belief, never on a sample):
```
a_i_exe ~ pi_execute_i(. | o_i, b_i)
b_i = { (mu_z_i, Sigma_z_i), (mu_rho, Sigma_rho) }
```

## 7. Disentangling $z_i$ and $\rho$

Generative independence target: $z_i$ carries the local regime $\mu_i$ (switches at
$\tilde\vartheta_i$); $\rho$ carries the adversary budget $\epsilon_a$ (one team-wide draw,
independent of every $\mu_i$). Enforce via **data path**, with a **cleanup penalty**:

**Data path (primary):** context encoder input `x_i,t` must only ever come from
adversary-free trajectories — exploration transitions, and *clean-action* execution
transitions. `check_a^-i` is a critic-side quantity computed at update time, **never written
to any buffer**, so there is structurally no path from the adversary to the flow input. This
alone gives $I(z_i;\rho)=0$ from the computation graph, not from optimization — get this
right and the penalty below is just cleanup, not the main defense.

**Cleanup penalty** (residual linear leakage, e.g. via regime-correlated execution
histories or shared upstream features):
```
# over a minibatch of size B, mu_z_b = agent's context posterior mean at step t,
# mu_rho_b = team budget posterior mean at the SAME step t
C_z_rho  = (1/B) * sum_b (mu_z_b - mean(mu_z)) @ (mu_rho_b - mean(mu_rho)).T
L_cov    = frobenius_norm(C_z_rho) ** 2
```
Only the cross-covariance block is penalized (not full joint covariance) — marginal
variances of $z$, $\rho$ stay free. Acts on posterior *means*, not reparameterized samples
(targets systematic dependence, not sampling noise). This is a **linear**-dependence-only
guarantee; if you need nonlinear independence, that's HSIC/MI-bound territory (heavier,
likely unnecessary if the data path is clean).

**Phase-1 representation objective:**
```
L_rep = L_enc + L_rho + lambda_cov * L_cov         # small lambda_cov
```

## 8. Factored execution critic (FMASAC)

```
Q_i(tau_i, a_i, b; zeta_i)                          # per-agent execution critic, full belief b = (b_1..b_n)
Q_tot(tau, a, s, b) = g_omega(s, {Q_i(tau_i, a_i, b; zeta_i)}_i)   # IGM-satisfying mixer
```
Built only from execution critics (invariant 1). Standard FMASAC actor/critic objectives,
but evaluated at the **MAAL-perturbed** joint action:
```
# actor loss (agent i's own action nominal, teammates' perturbed)
L(pi_execute) = E[ alpha_exe * sum_i log pi_execute_i(a_i|o_i,b_i) - Q_tot(tau, (a_i, check_a^-i), s, b) ]

# critic loss (fully perturbed joint action, TD target from target networks)
L_Q(zeta, omega) = E_{D_exe}[ (Q_tot(tau, check_a, s, b) - y_tot)^2 ]
```
`y_tot` = standard soft bootstrapped target from target nets `zeta_bar, theta_x_bar,
omega_bar`. `alpha_exe` adapted against a min-entropy constraint `H_0`, independent of
`alpha_exp`.

## 9. Replay buffers

Two disjoint buffers:
- **`D_exp`**: `(o_i,t, a_i,t^exp, o_i,t+1)` tagged with active regime label `mu_i(t)`.
  Drives exploration policy + contrastive encoder.
- **`D_exe`**: `(o_i,t, a_i,t^exe, o_i,t+1, r_i,t, mu_i(t))`. Also receives every exploration
  transition with intrinsic reward stripped (so execution-side context inference sees the
  full trajectory).

**Never stored** (always recomputed at sample time, since they depend on current params):
intrinsic reward `r_aux`, sampled contexts `z_i`, MAAL perturbation `check_a^-i`. This is
also what keeps the context encoder's input adversary-free (§7).

## 10. Training algorithms

### Phase 1 — joint context/exploration/execution training

```
Require: p(mu), p_sw, p_eps, H, encoders {f_phi, f_phi_bar, q_eta}, decoders {p_psi, p_eta},
         policies {pi_explore_i}, {pi_execute_i}, critics {Q_exp_i}, {Q_exe_i} + targets,
         mixer g_omega, buffers D_exp/D_exe,
         hyperparams (nu, kappa), tau_m, alpha_adv, lambda_CPC, lambda_cov

for each training iteration:
    for each agent i: sample mu_1i, mu_2i ~ p(mu); vartheta_i ~ GEOM(p_sw);
                       vartheta_i_tilde = min(vartheta_i, H)
    sample ONE eps_a ~ p_eps
    init each agent's posterior: mu_i,0 = 0, sigma2_i,0 = kappa

    for t = 1..H:
        for each agent i:
            active regime mu_i(t) = mu_1i if t < vartheta_i_tilde else mu_2i
            (mu_rho, Sigma_rho) = q_eta(x_t^exe)
            b_i = { (mu_i,(t-1), sigma2_i,(t-1) * I), (mu_rho, Sigma_rho) }
            a_i_exp ~ pi_explore_i(.|o_i)
            a_i_exe ~ pi_execute_i(.|o_i, b_i)
        execute joint action; observe r_i, o_i'
        store (o_i, a_i_exp, o_i') tagged mu_i(t) in D_exp
              (+ intrinsic-reward-free copy in D_exe)
        store (o_i, a_i_exe, o_i', r_i, mu_i(t)) in D_exe
        for each agent i:
            c_i,t = f_phi(y_i,t | x_i,t)
            update (mu_i,t, sigma2_i,t) via §3 recursion
            if t == vartheta_i_tilde: reset mu_i,t = 0, sigma2_i,t = kappa   # ground-truth reset

    # --- updates (once per iteration, off the rollout above) ---
    sample same-/cross-regime batches from D_exp; compute L_enc and batched means (mu_z_b, mu_rho_b)
    r_aux_i,t = 0.5*log(sigma2_i,(t-1)/sigma2_i,t)
    r_e_i,t   = r_env_i,t + alpha_exp * r_aux_i,t
    update pi_explore_i, Q_exp_i via SAC on r_e_i; adapt alpha_exp

    minibatch from D_exe; recompute each z_i with current phi
    MAAL step: check_a^-i = Proj_eps_a[ a_hat^-i - alpha_adv * grad Q_tot_target(...) ]
    (mu_rho, Sigma_rho) = q_eta(x_exe); rho ~ q_eta; eps_hat = p_eta(rho)
    descend L_rep = L_enc + L_rho + lambda_cov * L_cov  w.r.t. (phi, psi, eta)
    phi_bar <- tau_m * phi_bar + (1 - tau_m) * phi

    evaluate y_tot and the soft-policy objective at the MAAL-perturbed action
    descend L_Q w.r.t. {zeta_i}, omega
    update {theta_x_i} via factored soft-policy objective; adapt alpha_exe
    periodically Polyak-update zeta_bar, theta_x_bar, omega_bar

return phi, psi, eta, {theta_e_i}, {theta_x_i}, {zeta_i}, omega
```

### Phase 2 — supervised change-point detection

```
Require: frozen f_phi; InDiD transformer h_xi (causal + local-window attention mask);
         labeled episodes {(tau_i, vartheta_i_tilde)}; false-alarm weight lambda_FA

for each training iteration:
    sample/replay a labeled episode
    for each agent i:
        frozen embeddings c_i,1:H = f_phi(tau_i,1:H)
        p_i,t = h_xi(c_i,1:t)  for t = 1..H   (under the causal/local-window mask)
    L_CPD = sum_i [ L_delay(p_i,1:H, vartheta_i_tilde) + lambda_FA * L_FA(p_i,1:H, vartheta_i_tilde) ]
    descend L_CPD w.r.t. xi

return xi
```

### Meta-test — change-point-triggered re-exploration (per agent, deployment)

```
Require: everything frozen; threshold C; re-exploration budget K; uncertainty floor sigma2_min

for each i: mu_i = 0, sigma2_i = kappa, mode_i = EXPLORE, k_i = 0

for t = 1, 2, ...:
    for each agent i:
        if mode_i == EXPLORE:
            a_i ~ pi_explore_i(.|o_i); k_i += 1
        else:
            (mu_rho, Sigma_rho) = q_eta(x_exe)
            b_i = { (mu_i, sigma2_i * I), (mu_rho, Sigma_rho) }
            a_i ~ pi_execute_i(.|o_i, b_i)
    execute joint action; observe r_i, o_i'
    for each agent i:
        c_i,t = f_phi(y_i,t|x_i,t); update (mu_i,t, sigma2_i,t) via §3 recursion
        p_i,t = h_xi(c_i,1:t)
        if p_i,t >= C:
            reset: mu_i=0, sigma2_i=kappa, mode_i=EXPLORE, k_i=0        # change point for agent i
        if mode_i == EXPLORE and (k_i >= K or sigma2_i,t <= sigma2_min):
            mode_i = EXECUTE
```

## 11. Evaluation environment (Search-and-Rescue, VMAS)

### Stack
- **VMAS** (`bettini2022vmas`) — batched, GPU-vectorized continuous-control multi-agent sim
- **BenchMARL** (`bettini2024benchmarl`) on **TorchRL** (`bou2024torchrl`) for training
- Continuous action interface for the FMASAC backbone; discrete interface for the
  value-factorization baseline below.

### Arena / scenario
`Map`: bounded, optionally walled arena with agents (circles), obstacles (collidable boxes /
spheres), victims (point targets with scalar `weight`). Buildable from YAML or random-gen,
with save/reload round trip. `Scenario` wraps a `Map` into a VMAS world.
```
k_v = ceil(weight_v / 10)     # rescuers required for victim v — the coordination requirement
```
Health/rescue-state/counters as tensors of shape `(batch_dim,)` — no shared episode state
across vectorized copies.

### Observation (two disjoint, asymmetric channels)
```
o_i = [ pos(2), vel(2), lidar(n_rays), ears(2 * n_victims) ]
```
- **Lidar** (VMAS ray sensor, `n_rays=12`, range `3.0`): obstacle-only distances, filtered to
  exclude victims/agents. Navigation channel — geometry is directly, fully observable.
- **Ears** (`EaredAgent`): per victim, a left/right distance from two virtual ear offset
  points — never a relative position vector. The *only* channel carrying victim info; models
  sound-based detection. Deliberately the weaker, indirect cue — makes the disturbance-
  relevant signal the hardest part of the state to infer.

### Rescue mechanics / reward
```
h_v starts at 100; loses decay_rate=0.2 health/step while active
rescued the first time k_v agents are simultaneously within rescue_range=0.5 (sticky, one-shot)
r_rescue_v = rescue_reward * (h_v_at_rescue / 100)          # per rescuing agent
shared per-step penalty: -decay_penalty * sum_v(delta h_v)
episode ends when every victim is rescued or dead (h_v <= 0)
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_rays` | 12 | lidar rays/agent |
| lidar range | 3.0 | max lidar reading |
| `k_v` | `ceil(weight_v/10)` | agents needed to rescue victim v |
| `rescue_range` | 0.5 | presence distance |
| `h_v` initial | 100 | victim health at episode start |
| `decay_rate` | 0.2 | health lost/active step |
| `rescue_reward` | config | rescue bonus scale |
| `decay_penalty` | config | shared health-loss penalty scale |

### Domain-randomization hook
`PhysicsTaskSampler` draws 4 world params — `drag`, `linear_friction`, `angular_friction`,
`agent_mass` — from configurable uniform ranges; scenario applies them at world/agent
construction. Entry point for sim-to-real / ADR axis and MAML-style task distributions.
Implemented + tested; not yet used in a training run.

### Distributionally robust baseline (DrIGM stand-in)
`RobustQmix` / `RobustVdn` subclass BenchMARL's stock `Qmix`/`Vdn`, changing only the TD(0)
discount: `gamma * (1 - rho)` in place of `gamma`. Reproduces the $\rho$-contamination robust
Bellman target `r + gamma*(1-rho)*max_a Q(s',a)*(1-done)` exactly, no mixer/loss/replay
reimplementation, stays batched. `RescueTaskClass` (BenchMARL `VmasClass`) wraps `Scenario`
as a task, building a fresh `Scenario` per env (train/eval don't share a mutable object).
`rho=0` recovers plain VDN/QMIX — one parameter sweeps both the baseline and the robust
variant.

### Validation status — smoke-tested only, nothing trained to convergence
- Rescue mechanics: forcing `k_v` agents onto a victim ⇒ `rescued=true`, reward exactly
  `rescue_reward * h_v_at_rescue/100`, health frozen post-rescue, `done=true`; unreached
  victims decay, never trigger the bonus.
- Lidar: short reads near obstacles, max-range reads when clear, victims never register.
- Shapes: obs/reward tensors match `n_agents`/`n_victims` on fixed + random maps, incl.
  YAML round trip.
- Rendering: fixed a camera-origin bug and a `viewer_zoom` squaring bug (VMAS's auto-fit
  camera squares zoom past ~1×); explicit `fit_map` option computes correct
  `sqrt(half_extent)`.
- Training pipeline: a few end-to-end BenchMARL iterations run error-free; registered
  value-estimator discount confirmed `= gamma*(1-rho)` (e.g. `0.99*(1-0.3)=0.693`).

### What remains before the full comparison
1. No adaptive-MARL baseline (change-point / sliding-window) wired up yet.
2. No layered unbounded drift + bounded adversarial perturbation together — currently
   static/reset-time-randomized only.
3. Shared metric set (worst-case degradation + tracking speed) not implemented.
4. Nothing trained to convergence or evaluated.

### Baselines / ablations for the full study (§10.7 of thesis, planned eval)
- **Baselines:** plain FMASAC (nominal/adaptation reference), standalone M3DDPG (robust
  exemplar), a detector-based adaptive exemplar, a single-agent meta-RL exemplar applied
  per-agent.
- **Ablations:** exchangeable posterior → CCM mean pooling; closed-form info gain → InfoNCE
  bound-difference; remove MAAL (nominal critic target) or random-instead-of-MAAL teammate
  perturbation; fix `eps_a` instead of sampling (budget encoder degenerates); InDiD →
  classical online detector; merge the two training phases into one.

### Success/failure criteria (what would confirm or refute the method)
**Confirms**, on the same runs: (i) matches plain FMASAC's nominal return with no
perturbation, no switch; (ii) exceeds standalone M3DDPG's worst-case return under
perturbation; (iii) shorter recovery time after a regime switch than the detector-based
exemplar; (iv) keeps (ii) and (iii) when perturbation + switch happen together, and each
ablation degrades performance.
**Refutes/weakens**: nominal return drops below plain FMASAC (over-conservative execution
policy); InDiD's delay/false-alarm rate is no better than a classical detector; exchangeable
posterior gives no advantage over mean pooling on recovery time.

## 12. Known limitations (carry into the implementation, don't silently "fix")

- Exact info gain + O(1) update hold only under the exchangeable-Gaussian code assumption;
  fallback is the InfoNCE bound-difference estimator.
- Two-phase training ⇒ detector can't shape the representation it consumes; InDiD needs
  ground-truth switch times (simulator-only).
- Perturbation budget $\epsilon_a$ is **team-shared**, not per-agent — can't model an
  adversary that targets one agent's action channel harder than the rest.
- MAAL = one linearized gradient step ⇒ local/first-order worst case, can under-estimate a
  strong joint teammate deviation.
- Budget encoder is supervised by $\epsilon_a$ in Phase 1 but must infer it unsupervised at
  deployment from trajectory statistics — calibration outside the training distribution
  $p_\epsilon$ is unverified.
- Heavy stack: 1 invertible flow, 2 variational encoder/decoder pairs feeding the critic, 2
  full SAC stacks (only one factored), 1 transformer detector.
- Inherits FMASAC's monotonic-mixer restriction and CTDE's centralized-training-state
  requirement.

## 13. Suggested module layout

```
method/
  encoders/
    context_encoder.py      # f_phi (flow), p_psi (decoder), BRUNO recursion (§3)
    budget_encoder.py        # q_eta, p_eta (§6)
  policies/
    exploration_policy.py    # pi_explore_i, not belief-conditioned
    execution_policy.py      # pi_execute_i, belief-conditioned
  critics/
    exploration_critic.py    # Q_exp_i (no mixer)
    execution_critic.py      # Q_exe_i + g_omega mixer (IGM), FMASAC objectives (§8)
  adversary/
    maal.py                  # one-step projected-gradient perturbation (§6)
  detector/
    indid.py                 # transformer h_xi, causal+local-window mask (§10 Phase 2)
  disentangle/
    cov_penalty.py           # L_cov (§7)
  buffers/
    replay.py                # D_exp / D_exe semantics (§9)
  training/
    phase1.py
    phase2.py
    meta_test.py
env/
  map.py                     # Map (YAML load/save, random gen)
  scenario.py                # VMAS Scenario wrapper
  physics_task_sampler.py    # domain-randomization hook
  robust_baselines.py        # RobustQmix / RobustVdn (gamma*(1-rho) discount)
  rescue_task_class.py       # BenchMARL VmasClass wrapper
configs/
  hyperparams.yaml           # nu, kappa, tau_m, alpha_adv, lambda_CPC, lambda_cov,
                              # p_sw, p_eps, H, C, K, sigma2_min, rescue params (Table §11)
```

## 14. Full symbol glossary

| Symbol | Meaning |
|---|---|
| $i$, $n$ | agent index, number of agents |
| $\tau_i$, $\tau$ | agent $i$'s local history; joint history |
| $s$ | global state (centralized training only) |
| $\mu_{1,i},\mu_{2,i}$ | agent $i$'s two sampled local regimes |
| $\tilde\vartheta_i$ | agent $i$'s switch time (clipped to horizon $H$) |
| $\epsilon_a$ | team-wide adversary perturbation-ball radius (per episode) |
| $z_i$ | agent $i$'s context latent (local task/regime) |
| $\rho$ | team-shared perturbation-budget latent |
| $b_i$ | agent $i$'s belief $\{(\mu_{z,i},\Sigma_{z,i}), (\mu_\rho,\Sigma_\rho)\}$ |
| $f_\phi$ | invertible flow, transition → code $c_{i,t}$ |
| $\nu,\kappa$ | exchangeable-process covariance hyperparameters |
| $p_\psi$ | context decoder (reconstructs $r,o'$) |
| $q_\eta,p_\eta$ | budget encoder / decoder (reconstructs $\epsilon_a$) |
| $\lambda_{\mathrm{CPC}}$ | contrastive-term weight (0 unimodal, >0 multimodal) |
| $\pi_{\theta_e,i}$, $\pi_{\theta_x,i}$ | exploration / execution policy |
| $Q_i^{\mathrm{exp}}$, $Q_i^{\mathrm{exe}}$ | exploration / execution per-agent critic |
| $g_\omega$ | IGM mixing network → $Q_{\mathrm{tot}}$ |
| $\alpha_{\mathrm{exp}},\alpha_{\mathrm{exe}}$ | independent SAC entropy temperatures |
| $\alpha_{\mathrm{adv}}$ | MAAL step size |
| $\check{\mathbf a}^{-i}$ | MAAL-perturbed teammate joint action |
| $h_\xi$ | InDiD change-point detector (transformer) |
| $C$ | detection threshold |
| $K$ | meta-test re-exploration step budget |
| $\sigma^2_{\min}$ | meta-test uncertainty floor (alt. stop condition) |
