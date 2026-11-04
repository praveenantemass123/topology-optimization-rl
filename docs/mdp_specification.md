# Markov Decision Process (MDP) Specification for Topology Optimization

**Project:** Deep Reinforcement Learning for Structural Topology Optimization  
**Supervisor:** Prof. Asur Vijaya Kumar · Department of Mechanical Engineering, IIT Kharagpur  
**Author:** Praveen Kumar  
**Document Status:** Working Specification (Stage 2 Gate Deliverable)  

---

## 1. Overview & Objective

The goal is to formulate 2D minimum-compliance topology optimization as a sequential decision-making process under the Markov Decision Process (MDP) framework $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. 

A policy network $\pi_\theta(a_t \mid s_t)$ acts as an iterative structural designer: observing the current material distribution and stress field, applying bounded continuous updates to element densities, and receiving feedback based on structural stiffness gains and volume constraint satisfaction.

---

## 2. Mathematical Definition of the MDP Tuple

### A. State Space ($\mathcal{S}$)
The state $s_t$ represents the full physical and mechanical state of the design domain at step $t$. It is represented as a multi-channel 2D tensor of shape $(C, N_y, N_x)$:

| Channel | Description | Value Range | Physical Intuition |
| :--- | :--- | :--- | :--- |
| **Channel 0: Density Map ($\mathbf{x}_t$)** | Relative material density of each element | $[x_{\min}, 1.0]$ | Structural layout (0 = void, 1 = solid). |
| **Channel 1: Strain Energy Density ($\mathbf{w}_t$)** | Element strain energy: $\frac{1}{2} \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$ | $[0.0, 1.0]$ (normalized) | Identifies critical load paths and high-stress regions. |
| **Channel 2: Load Vector Mask ($\mathbf{M}_F$)** | Point force location and direction $(F_x, F_y)$ | $[-1.0, 1.0]$ | Informs the agent of applied external boundary loads. |
| **Channel 3: Support Mask ($\mathbf{M}_U$)** | Fixed displacement boundary conditions | $\{0, 1\}$ | Informs the agent of structural anchorage to ground/walls. |

*Total State Dimension:* $(4, N_y, N_x)$, or $(2, N_y, N_x)$ for fixed boundary condition experiments.

---

### B. Action Space ($\mathcal{A}$)
We formulate the action space as a **continuous per-element density modification field**:
$$a_t \in [-m, m]^{N_y \times N_x}$$
where $m = 0.05$ (or $0.1$) is the maximum move limit per step.

* **Action Regularization (Mesh Filter):**  
  Raw policy outputs $a_t^{\text{raw}}$ are passed through the linear mesh-independency filter:
  $$\tilde{a}_t = \frac{\mathbf{H} a_t^{\text{raw}}}{\mathbf{H} \mathbf{1}}$$
  This eliminates spatial discontinuities, numerical checkerboards, and disconnected floating material islands.

---

### C. State Transition Dynamics ($\mathcal{P}$)
The state transition $s_{t+1} \sim \mathcal{P}(s_{t+1} \mid s_t, a_t)$ is **deterministic**, governed by the laws of linear elasticity:

1. **Density Update:**  
   $$\mathbf{x}_{t+1}^{\text{unconstrained}} = \text{clip}(\mathbf{x}_t + \tilde{a}_t, \, x_{\min}, \, 1.0)$$

2. **Hard Volume Constraint Enforcement (Projection Step):**  
   To prevent the agent from drifting into structurally invalid volume fractions, densities are strictly projected onto the target volume fraction $V^*$ via proportional scaling:
   $$\mathbf{x}_{t+1} = \text{clip}\left( \mathbf{x}_{t+1}^{\text{unconstrained}} \times \frac{V^* \cdot N}{\sum_e x_{t+1, e}^{\text{unconstrained}}}, \, x_{\min}, \, 1.0 \right)$$

3. **FEA Equilibrium Solve:**  
   Assemble the global stiffness matrix $\mathbf{K}(\mathbf{x}_{t+1}) = \sum_e [x_{t+1, e}^p E_0] \mathbf{k}_0$ and solve:
   $$\mathbf{K}(\mathbf{x}_{t+1}) \mathbf{U}_{t+1} = \mathbf{F}$$

4. **Physics Extraction:**  
   Compute updated compliance $C_{t+1} = \mathbf{U}_{t+1}^T \mathbf{K} \mathbf{U}_{t+1}$ and update strain energy channels.

---

### D. Reward Function ($\mathcal{R}$)
The reward function drives compliance minimization while penalizing any residual volume constraint deviation:

$$r_t = -\frac{C_{t+1} - C_t}{C_0} - \lambda \left| \frac{\sum_e x_{t+1, e}}{N} - V^* \right|$$

* **Normalization by $C_0$:** Compliance values vary by orders of magnitude across different load geometries. Dividing by the initial compliance $C_0$ scales the reward to $[-1, 1]$, stabilizing PPO value estimation and gradient updates.
* **Penalty Coefficient $\lambda$:** $\lambda = 10.0$ penalizes volume violation (acts as a safeguard if projection is soft).

---

### E. Termination & Episode Budget ($\mathcal{T}$, $\gamma$)
* **Episode Horizon:** Fixed step budget of $T_{\max} = 30$ to $50$ steps.
* **Early Convergence Termination:** If $|\Delta C / C_t| < 10^{-4}$ for 3 consecutive steps after step 5.
* **Discount Factor:** $\gamma = 0.99$, ensuring the agent values long-term structural connectivity over short-term myopic moves.

---

## 3. Comparison of Action Formulations

| Formulation | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **(A) Single Element Selection** | Small discrete action space | Requires $N_x \times N_y$ steps per design; extremely slow credit assignment | Rejected for grids $> 20 \times 10$ |
| **(B) Continuous Density Delta Field** | Co-adapts all elements simultaneously; fast convergence in 20–30 steps | Continuous action space requires PPO/SAC | **Selected (Primary Formulation)** |
| **(C) One-Shot Generation** | Instant inference | Bypasses RL sequential benefits; effectively supervised learning | Baseline comparison in Stage 5 |

---

## 4. Verification & Gate Checklist
- [x] Analytical element stiffness matrix symmetry verified.
- [x] Zero strain energy confirmed for all 3 rigid-body modes.
- [x] Energy balance equality $\mathbf{U}^T \mathbf{K} \mathbf{U} \equiv \mathbf{F}^T \mathbf{U}$ verified ($< 10^{-11}$ error).
- [x] Classical SIMP benchmark convergence numbers logged in `baselines.csv`.
