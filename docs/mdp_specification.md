# Markov Decision Process (MDP) Specification for Topology Optimization

**Project:** Deep Reinforcement Learning for Structural Topology Optimization  
**Supervisor:** Prof. Asur Vijaya Kumar · Department of Mechanical Engineering, IIT Kharagpur  
**Author:** Praveen Kumar  
**Document Status:** Complete & Audited — **Ready for Supervisor Sign-Off** (Stage 2 Gate Deliverable)  

---

## 1. Overview & Objective

The goal is to formulate 2D minimum-compliance topology optimization as a sequential decision-making process under the Markov Decision Process (MDP) framework $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. 

A policy network $\pi_\theta(a_t \mid s_t)$ acts as an iterative structural designer: observing the current material distribution and stress field, applying bounded continuous updates to element densities, and receiving feedback based on structural stiffness gains and volume constraint satisfaction.

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

### D. Reward Function ($\mathcal{R}$)
The reward function drives compliance minimization while penalizing any residual volume constraint deviation:

$$r_t = \frac{C_t - C_{t+1}}{C_0} - \lambda_v \left| \frac{\sum_e x_{\text{phys}, t+1, e}}{N} - V^* \right|$$

* **Relative Compliance Improvement:**  
  Dividing by the initial uniform compliance $C_0$ normalizes the reward across arbitrary load magnitudes and grid sizes. A positive reward $r_t > 0$ strictly signifies a stiffer structure.
* **No Artificial $\times 100$ Multiplier:**  
  The reward directly measures fractional stiffness gains (typically $+0.01$ to $+0.08$ per step), keeping gradients well-conditioned.
* **Safety Bound:**  
  Reward is safely clipped to $[-2.0, 2.0]$ to guard against numerical singularities if a structure becomes disconnected, while never clipping normal physical design updates.

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
