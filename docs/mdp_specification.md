# Markov Decision Process (MDP) Specification for Topology Optimization

**Project:** Deep Reinforcement Learning for Structural Topology Optimization  
**Supervisor:** Prof. Asur Vijaya Kumar · Department of Mechanical Engineering, IIT Kharagpur  
**Author:** Praveen Kumar  
**Document Status:** Complete & Audited — **Ready for Supervisor Sign-Off** (Stage 2 Gate Deliverable)  

---

## 1. Overview & Objective

The goal is to formulate 2D minimum-compliance topology optimization as a sequential decision-making process under the Markov Decision Process (MDP) framework $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. 

A policy network $\pi_\theta(a_t \mid s_t)$ acts as an iterative structural designer: observing the current material distribution and stress field, applying bounded continuous updates to element densities, and receiving feedback from a weighted combination of physical factors: stiffness gain, volume constraint satisfaction, grayness, action smoothness and final design quality.

---

## 2. Mathematical Definition of the MDP Tuple

### A. State Space ($\mathcal{S}$)
The state $s_t$ represents the physical and mechanical state of the design domain at step $t$, structured as a multi-channel 2D tensor of shape $(C, N_y, N_x)$ where $C = 4$:

| Channel | Description | Value Range | Physical Intuition |
| :--- | :--- | :--- | :--- |
| **Channel 0: Physical Density ($\mathbf{x}_{\text{phys}, t}$)** | Filtered element densities | $[x_{\min}, 1.0]$ | Current structural layout ($x_{\min} = 10^{-3}$, $1 = \text{solid}$). |
| **Channel 1: Strain Energy Density ($\mathbf{w}_t$)** | Element strain energy: $\frac{1}{2} \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$ | $[0.0, 1.0]$ (normalized) | Identifies critical load paths and high-stress regions. |
| **Channel 2: Load Vector Mask ($\mathbf{M}_F$)** | Point force spatial location and direction | $\{0, 1\}$ | Informs the agent of applied external boundary loads. |
| **Channel 3: Support Mask ($\mathbf{M}_U$)** | Fixed displacement boundary conditions | $\{0, 1\}$ | Informs the agent of structural anchorage points. |

*Total State Dimension:* $(4, N_y, N_x)$ float32 tensor.

---

### B. Action Space ($\mathcal{A}$)
The policy emits normalized continuous actions:
$$a_t \in [-1.0, 1.0]^{N_y \times N_x}$$

Inside the environment, actions are scaled by the maximum move limit $\Delta x_{\max} = 0.05$:
$$\Delta \mathbf{x}_t = \Delta x_{\max} \cdot a_t$$

* **Why normalized $[-1, 1]$ actions?**  
  Standard continuous RL algorithms (e.g. PPO Gaussian policies with initial standard deviation $\sigma_0$) operate smoothly in $[-1, 1]$. Scaling by $\Delta x_{\max}$ inside the environment prevents immediate action saturation and bang-bang noise.

---

### C. State Transition Dynamics ($\mathcal{P}$)
The state transition $s_{t+1} \sim \mathcal{P}(s_{t+1} \mid s_t, a_t)$ is **deterministic**, governed by linear elasticity:

1. **Tentative Density Update:**  
   $$\mathbf{x}_{t+1}^{\text{unconstrained}} = \text{clip}(\mathbf{x}_t + \Delta \mathbf{x}_t, \, x_{\min}, \, 1.0)$$

2. **Strict Volume Projection (Bisection Shift):**  
   To strictly satisfy the target volume fraction $V^*$ without violating physical bounds $[x_{\min}, 1.0]$, we solve for a scalar shift $s^*$ via bisection:
   $$\text{mean}\left( \text{clip}(\mathbf{x}_{t+1}^{\text{unconstrained}} + s^*, \, x_{\min}, \, 1.0) \right) = V^*$$
   This guarantees $\left| \frac{\sum_e x_{t+1, e}}{N} - V^* \right| < 10^{-6}$ at every step.

3. **Physical Filtering:**  
   $$\mathbf{x}_{\text{phys}, t+1} = \frac{\mathbf{H} \mathbf{x}_{t+1}}{\mathbf{H} \mathbf{1}}$$
   This enforces mesh independence and eliminates checkerboard instability.

4. **FEA Equilibrium Solve:**  
   Assemble the global stiffness matrix $\mathbf{K}(\mathbf{x}_{\text{phys}, t+1}) = \sum_e [E_{\min} + x_e^p (E_0 - E_{\min})] \mathbf{k}_0$ ($E_{\min} = 10^{-9}$) and solve:
   $$\mathbf{K} \mathbf{U}_{t+1} = \mathbf{F}$$

5. **Physics Extraction:**  
   Compute updated compliance $C_{t+1} = \mathbf{U}_{t+1}^T \mathbf{K} \mathbf{U}_{t+1}$ and update strain energy channels.

---

### D. Reward Function ($\mathcal{R}$): Weighted Multi-Factor Formulation
In RL the agent optimizes only what the scalar reward measures. A stiffness-only reward drives the agent to add material everywhere; a weight-only reward drives it to delete material. The reward is therefore a weighted sum of normalized physical factors:

$$R_t = w_{\text{stiff}}\,\mathcal{F}_{\text{stiffness}} + w_{\text{weight}}\,\mathcal{F}_{\text{weight}} + w_{\text{gray}}\,\mathcal{F}_{\text{grayness}} + w_{\text{smooth}}\,\mathcal{F}_{\text{smoothness}} + w_{\text{term}}\,\mathcal{F}_{\text{terminal}}$$

$$R_t^{\text{clipped}} = \text{clip}\left(R_t,\, -R_{\max},\, R_{\max}\right), \qquad R_{\max} = 2.0$$

| Factor | Formula | Typical range | Config key (default) |
| :--- | :--- | :---: | :--- |
| **Stiffness** | $\mathcal{F}_{\text{stiffness}} = \dfrac{C_{t-1} - C_t}{C_0}$ | $[0.01, 0.10]$ per step | `w_stiff` (1.0) |
| **Weight / volume** | $\mathcal{F}_{\text{weight}} = -\left\lvert V_t - V^* \right\rvert$ or $-\left(\dfrac{V_t - V^*}{V^*}\right)^2$, with $V_t = \frac{1}{N}\sum_e x_{\text{phys},e}$ | $\approx 0$ with projection | `w_weight` (10.0, formerly `lam_v`), `weight_penalty` (`abs` / `sq_rel`) |
| **Grayness** | $\mathcal{F}_{\text{grayness}} = -\dfrac{1}{N}\sum_e 4\,x_{\text{phys},e}(1 - x_{\text{phys},e})$ | $[-1, 0]$ | `w_gray` (0.0; ablation 0.05 to 0.2) |
| **Action smoothness** | $\mathcal{F}_{\text{smoothness}} = -\dfrac{1}{N}\sum_e a_{t,e}^2$ | $[-1, 0]$ | `w_smooth` (0.0; ablation 0.01) |
| **Terminal** | $\mathcal{F}_{\text{terminal}} = -\dfrac{C_T}{C_0}$ if $t = T_{\max}$, else $0$ | $[-0.5, -0.2]$ at $t = T$ | `w_term` (0.0, formerly `beta`) |

* **Normalization by $C_0$:** $C_0$ is the compliance of the uniform initial design. Absolute compliance ranges over orders of magnitude with mesh and load; dividing by $C_0$ keeps the stiffness and terminal factors comparable across load cases and avoids badly scaled PPO gradients. A positive $\mathcal{F}_{\text{stiffness}}$ means the structure got stiffer.
* **Grayness:** $4x(1-x)$ is $0$ for solid ($x=1$) or void ($x=0$) elements and peaks at $1$ for $x = 0.5$, so it pushes toward manufacturable 0/1 layouts.
* **Smoothness:** penalizes large updates and element thrashing between steps. In the residual policy (Stage 4) $a_t$ is the policy's **raw residual output**, not the blended update $a_{\text{base}} + \alpha a_t$, so the sensitivity base direction is not penalized.
* **Terminal:** gives credit for the quality of the finished design at the horizon, helping long-term credit assignment. It is added before clipping.
* **Safety bound:** a disconnected structure makes $\mathbf{K}$ near-singular and compliance explode. The total is clipped to $[-R_{\max}, R_{\max}]$, and a non-finite compliance returns $-R_{\max}$. Normal design updates are never clipped (clip events are counted in `info["total_clip_events"]`).
* **No artificial $\times 100$ multiplier:** the stiffness factor directly measures fractional stiffness gain.

**Interaction with the exact volume projection.** Because the bisection projection (Section C.2) enforces $|V_t - V^*| < 10^{-6}$ at every step, $\mathcal{F}_{\text{weight}} \approx 0$ and $w_{\text{weight}}$ is effectively redundant in the projected setting. It is kept so that the same reward is valid when `project_volume: false` (pure soft-penalty approach), which is a planned ablation.

**Audited configuration (Stage 3 and 4 results).** The defaults $w_{\text{stiff}} = 1$, $w_{\text{weight}} = 10$, $w_{\text{gray}} = w_{\text{smooth}} = w_{\text{term}} = 0$, $R_{\max} = 2$ reproduce exactly the reward used for all reported results (residual PPO $C = 441.9 \pm 1.0$):
$$r_t = \frac{C_{t-1} - C_t}{C_0} - 10 \left\lvert V_t - V^* \right\rvert$$

**Recommended ablation weights** (to be evaluated, not yet trained): $w_{\text{gray}} = 0.05$, $w_{\text{smooth}} = 0.01$, $w_{\text{term}} = 1.0$. With these, the grayness contribution is about $-0.01$ to $-0.04$ per step and churn about $-0.001$ to $-0.01$, small relative to the stiffness signal of $+0.02$ to $+0.08$.

Each factor and its weighted contribution are returned per step in `info["reward_terms"]` for logging and ablation analysis.

**Implementation** (`src/env/topopt_env.py`, `TopOptEnv.reward_terms`):
```python
F_stiff  = -(c - C_prev) / C0
F_weight = -abs(V - volfrac)                     # or -((V - volfrac) / volfrac)**2
F_gray   = -mean(4 * x_phys * (1 - x_phys))
F_smooth = -mean(policy_action ** 2)             # raw policy action
F_term   = -(c / C0) if t == max_steps else 0.0
R = w_stiff*F_stiff + w_weight*F_weight + w_gray*F_gray + w_smooth*F_smooth + w_term*F_term
r = clip(R, -reward_clip, reward_clip)
```

---

### E. Termination & Episode Budget ($\mathcal{T}$, $\gamma$)
* **Episode Horizon:** Fixed step budget of $T_{\max} = 30$ steps.
* **Fixed-Horizon Truncation ONLY:** No early-termination threshold is applied. Early termination introduces loopholes where an agent prematurely stops upon encountering negative reward gradients.
* **Discount Factor:** $\gamma = 0.99$, ensuring the policy values long-term load-path connectivity over short-term myopic moves.

---

## 3. Comparison of Action Formulations

| Formulation | Pros | Cons | Decision |
| :--- | :--- | :--- | :--- |
| **(A) Single Element Selection** | Discrete action space | Requires $N_x \times N_y$ steps per design; intractable credit assignment | Rejected |
| **(B) Continuous Density Delta Field** | Co-adapts all elements simultaneously; fast convergence in 20–30 steps | Continuous action space requires PPO/SAC | **Selected (Stage 3 & 4)** |
| **(C) Sensitivity Residual Policy** | Policy outputs residual update $\Delta a$ on top of sensitivity heuristic | Monotonically improves upon classical sensitivity heuristic | **Selected for Stage 4** |
| **(D) One-Shot Generation** | Instant inference | Bypasses sequential decision-making; sensitive to load variations | Evaluated in Stage 5 |

---

## 4. Verification & Gate Checklist

- [x] **Analytical element stiffness matrix symmetry verified** ($\Delta < 1.11 \times 10^{-16}$).
- [x] **Zero strain energy confirmed for all 3 rigid-body modes** ($\Delta < 10^{-10}$).
- [x] **Energy balance equality $\mathbf{U}^T \mathbf{K} \mathbf{U} \equiv \mathbf{F}^T \mathbf{U}$ verified** ($< 5.4 \times 10^{-11}$ error).
- [x] **Classical SIMP ground truth established** in `baselines.csv` ($E_{\min} = 10^{-9}$).
- [ ] **Stage 2 Sign-off by Supervisor (Prof. Asur):** Scheduled for upcoming meeting.
- [ ] **Stage 3 Prototype Gate:** Partial — PPO agent beats Uniform and Random baselines; closing the gap to Sensitivity Heuristic via Stage 4 Residual Policy.
