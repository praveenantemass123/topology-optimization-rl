# Professor's Reference SIMP Codes (Without Phase Field)

This directory contains the original MATLAB reference implementations by **Prof. Pavan Kumar Asur Vijaya Kumar** (IIT Kharagpur) from his research repository [`Pavan-asur/SIMP_Topology_Optimisation_FGM`](https://github.com/Pavan-asur/SIMP_Topology_Optimisation_FGM).

These codes represent the **"Without Phase Field"** framework, focusing purely on structural compliance minimization using the SIMP (Solid Isotropic Material with Penalization) method.

---

### File Inventory & Descriptions

1. **`half_three_point_bending.m`**
   * **Description:** 2D SIMP topology optimization for a Half Three-Point Bending (Half-MBB) beam under point load.
   * **Mechanics:** Plane-stress Q4 finite element analysis, 88-line vectorization, conical sensitivity filtering ($r_{\min} = 4$), and Optimality Criteria (OC) update loop.
   * **Role in BTP:** Serves as the analytical baseline for our vectorized Python baseline in `src/simp/simp.py` and benchmark tests in `tests/test_fea.py`.

2. **`mult_load.m`**
   * **Description:** SIMP topology optimization extended to multiple independent load cases acting simultaneously or sequentially on the domain.
   * **Role in BTP:** Direct reference for Stage 5 multi-load generalization in the Reinforcement Learning pipeline.

3. **`mult_load_WIthHole.m`**
   * **Description:** Multi-load SIMP optimization with geometric passive void regions (fixed non-design hole zones).
   * **Role in BTP:** Reference for passive element masking and non-design domain constraints.

4. **`TBP_withHole.m`**
   * **Description:** Three-point bending beam with an embedded passive void/hole obstacle.

---

### Academic Citation & Attribution
If using or referencing these foundational MATLAB implementations, cite:
1. *Pavan Kumar Asur Vijaya Kumar et al.*, "A SIMP-phase field topology optimization framework to maximize quasi-brittle fracture resistance of 2D and 3D composites", *Theoretical and Applied Fracture Mechanics*, Vol. 114, 102919, 2021. DOI: [10.1016/j.tafmec.2021.102919](https://doi.org/10.1016/j.tafmec.2021.102919)
2. *Pavan Kumar Asur Vijaya Kumar et al.*, "SIMP Phase-field topology optimisation framework to maximise fracture resistance in FGMs", *Composite Structures*, 117750, 2023. DOI: [10.1016/j.compstruct.2023.117750](https://doi.org/10.1016/j.compstruct.2023.117750)
