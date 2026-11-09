"""
run_prof_codes.py
=================
Executes exact 1-to-1 Python translations of Prof. Pavan Asur's 4 MATLAB SIMP codes 
from `prof_codes_without_phase/`:
1. half_three_point_bending.m
2. mult_load.m
3. mult_load_WIthHole.m
4. TBP_withHole.m

Generates numerical iteration logs, convergence metrics, and high-resolution topology plots.
"""

import os
import time
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def get_88line_ke(nu=0.3):
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], dtype=float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=float)
    
    A = np.block([[A11, A12], [A12.T, A11]])
    B = np.block([[B11, B12], [B12.T, B11]])
    KE = (1.0 / (1.0 - nu**2) / 24.0) * (A + nu * B)
    return KE


def build_mesh_indices(nelx, nely):
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((1 + nely, 1 + nelx), order='F')
    edofVec = (2 * nodenrs[:-1, :-1] + 1).flatten(order='F')
    
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    edofMat = edofVec[:, None] + offsets[None, :] - 1  # 0-indexed in Python
    
    iK = np.kron(edofMat, np.ones((8, 1))).T.flatten(order='F').astype(int)
    jK = np.kron(edofMat, np.ones((1, 8))).T.flatten(order='F').astype(int)
    return edofMat, iK, jK


def build_filter(nelx, nely, rmin):
    iH = []
    jH = []
    sH = []
    
    r_ceil = int(np.ceil(rmin))
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1 - 1
            i2_min = max(i1 - (r_ceil - 1), 1)
            i2_max = min(i1 + (r_ceil - 1), nelx)
            j2_min = max(j1 - (r_ceil - 1), 1)
            j2_max = min(j1 + (r_ceil - 1), nely)
            for i2 in range(i2_min, i2_max + 1):
                for j2 in range(j2_min, j2_max + 1):
                    e2 = (i2 - 1) * nely + j2 - 1
                    dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                    weight = max(0.0, rmin - dist)
                    iH.append(e1)
                    jH.append(e2)
                    sH.append(weight)
                    
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()
    
    # Precompute adjoint volume weight vector: sum(xPhys) == np.dot(w_vol, xnew)
    H_scaled = H.multiply(1.0 / Hs[:, None]).tocsr()
    w_vol = np.array(H_scaled.sum(axis=0)).flatten()
    return H, Hs, w_vol


# =========================================================================
# CODE 1: half_three_point_bending.m
# =========================================================================
def run_half_three_point_bending(max_iter=150, tol=0.01):
    print("\n" + "="*70, flush=True)
    print(" RUNNING: 1. half_three_point_bending.m (Prof. Pavan Asur)", flush=True)
    print("="*70, flush=True)
    nelx = 200
    nely = 100
    volfrac = 0.5
    rmin = 4.0
    penal = 3.0
    nu = 0.3
    
    E1 = 210.0
    E2 = E1 * 10.0
    VF_alp = 4.0
    
    E0 = np.zeros((nely, nelx), order='F')
    for elx in range(1, nelx + 1):
        for ely in range(1, nely + 1):
            E0[ely - 1, elx - 1] = E1 + (E2 - E1) * (((ely - 1) / nely)**VF_alp)
    E0[-1, :] = E2
    Emin = np.full((nely, nelx), 0.001, order='F')
    
    KE = get_88line_ke(nu)
    edofMat, iK, jK = build_mesh_indices(nelx, nely)
    
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros(ndof)
    F[1] = -1.0
    
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([ndof - 1]))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    
    H, Hs, w_vol = build_filter(nelx, nely, rmin)
    
    x = np.full((nely, nelx), volfrac, order='F')
    xPhys = x.copy()
    loop = 0
    change = 1.0
    
    history_c = []
    history_ch = []
    
    t0 = time.time()
    while change > tol and loop < max_iter:
        loop += 1
        x_flat = xPhys.flatten(order='F')
        E0_flat = E0.flatten(order='F')
        Emin_flat = Emin.flatten(order='F')
        
        sK_mat = KE.flatten()[:, None] * (x_flat**penal * E0_flat + (1.0 - x_flat**penal) * Emin_flat)[None, :]
        sK = sK_mat.flatten(order='F')
        
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0
        
        U = np.zeros(ndof)
        U[freedofs] = spsolve(K[freedofs][:, freedofs], F[freedofs])
        
        U_e = U[edofMat]
        ce = np.sum((U_e @ KE) * U_e, axis=1).reshape((nely, nelx), order='F')
        c = np.sum((xPhys**penal * E0 + (1.0 - xPhys**penal) * Emin) * ce)
        dc = (E0 - Emin) * (-penal) * (xPhys**(penal - 1.0)) * ce
        dv = np.ones((nely, nelx))
        
        dc_filt = (H.dot(dc.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        dv_filt = (H.dot(dv.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        target_vol = volfrac * nelx * nely
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            step_val = x * np.sqrt(np.maximum(0.0, -dc_filt / (dv_filt * lmid + 1e-12)))
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, step_val))))
            
            # Fast volume constraint via precomputed adjoint weight
            vol_eval = np.dot(w_vol, xnew.flatten(order='F'))
            if vol_eval > target_vol:
                l1 = lmid
            else:
                l2 = lmid
                
        xPhys = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        history_c.append(c)
        history_ch.append(change)
        
        if loop % 10 == 0 or loop == 1 or change <= tol:
            print(f" It.: {loop:4d} | Obj.: {c:12.6f} | Vol.: {np.mean(xPhys):6.3f} | Change: {change:8.5f}", flush=True)
            
    elapsed = time.time() - t0
    print(f"--> Done in {loop} iterations ({elapsed:.2f}s). Final Obj: {c:.6f}, Vol: {np.mean(xPhys):.3f}", flush=True)
    return {"name": "half_three_point_bending", "xPhys": xPhys, "c": c, "iters": loop, "history_c": history_c, "time": elapsed}


# =========================================================================
# CODE 2: mult_load.m
# =========================================================================
def run_mult_load(max_iter=150, tol=0.01):
    print("\n" + "="*70, flush=True)
    print(" RUNNING: 2. mult_load.m (Prof. Pavan Asur)", flush=True)
    print("="*70, flush=True)
    nelx = 60
    nely = 60
    volfrac = 0.4
    penal = 3.0
    rmin = 6.0
    nu = 0.3
    
    E0 = np.ones((nely, nelx), order='F')
    Emin = np.full((nely, nelx), 0.001, order='F')
    
    KE = get_88line_ke(nu)
    edofMat, iK, jK = build_mesh_indices(nelx, nely)
    
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros((ndof, 2))
    dof_load1 = 2 * (nely + 1) * nelx + 2 - 1
    dof_load2 = 2 * (nely + 1) * (nelx + 1) - 1
    F[dof_load1, 0] = 1.0
    F[dof_load2, 1] = -1.0
    
    fixeddofs = np.arange(0, 2 * nely + 1)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    
    H, Hs, w_vol = build_filter(nelx, nely, rmin)
    x = np.full((nely, nelx), volfrac, order='F')
    xPhys = x.copy()
    loop = 0
    change = 1.0
    
    history_c = []
    t0 = time.time()
    target_vol = volfrac * nelx * nely
    while change > tol and loop < max_iter:
        loop += 1
        x_flat = xPhys.flatten(order='F')
        E0_flat = E0.flatten(order='F')
        Emin_flat = Emin.flatten(order='F')
        
        sK_mat = KE.flatten()[:, None] * (x_flat**penal * E0_flat + (1.0 - x_flat**penal) * Emin_flat)[None, :]
        sK = sK_mat.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0
        
        U = np.zeros((ndof, 2))
        U[freedofs, 0] = spsolve(K[freedofs][:, freedofs], F[freedofs, 0])
        U[freedofs, 1] = spsolve(K[freedofs][:, freedofs], F[freedofs, 1])
        
        c = 0.0
        dc = np.zeros((nely, nelx), order='F')
        for i_case in range(2):
            Ui = U[:, i_case]
            U_e = Ui[edofMat]
            ce = np.sum((U_e @ KE) * U_e, axis=1).reshape((nely, nelx), order='F')
            c += np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
            dc += -penal * (E0 - Emin) * (xPhys**(penal - 1.0)) * ce
            
        dv = np.ones((nely, nelx))
        dc_filt = (H.dot(dc.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        dv_filt = (H.dot(dv.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            step_val = x * np.sqrt(np.maximum(0.0, -dc_filt / (dv_filt * lmid + 1e-12)))
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, step_val))))
            vol_eval = np.dot(w_vol, xnew.flatten(order='F'))
            if vol_eval > target_vol:
                l1 = lmid
            else:
                l2 = lmid
                
        xPhys = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        history_c.append(c)
        if loop % 10 == 0 or loop == 1 or change <= tol:
            print(f" It.: {loop:4d} | Obj.: {c:12.4f} | Vol.: {np.mean(xPhys):6.3f} | Change: {change:8.4f}", flush=True)
            
    elapsed = time.time() - t0
    print(f"--> Done in {loop} iterations ({elapsed:.2f}s). Final Obj: {c:.4f}, Vol: {np.mean(xPhys):.3f}", flush=True)
    return {"name": "mult_load", "xPhys": xPhys, "c": c, "iters": loop, "history_c": history_c, "time": elapsed}


# =========================================================================
# CODE 3: mult_load_WIthHole.m
# =========================================================================
def run_mult_load_with_hole(max_iter=150, tol=0.01):
    print("\n" + "="*70, flush=True)
    print(" RUNNING: 3. mult_load_WIthHole.m (Prof. Pavan Asur)", flush=True)
    print("="*70, flush=True)
    nelx = 60
    nely = 60
    volfrac = 0.4
    penal = 3.0
    rmin = 6.0
    nu = 0.3
    
    E0 = np.ones((nely, nelx), order='F')
    Emin = np.full((nely, nelx), 0.001, order='F')
    
    KE = get_88line_ke(nu)
    edofMat, iK, jK = build_mesh_indices(nelx, nely)
    
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros((ndof, 2))
    dof_load1 = 2 * (nely + 1) * nelx + 2 - 1
    dof_load2 = 2 * (nely + 1) * (nelx + 1) - 1
    F[dof_load1, 0] = 1.0
    F[dof_load2, 1] = -1.0
    
    fixeddofs = np.arange(0, 2 * nely + 1)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    
    H, Hs, w_vol = build_filter(nelx, nely, rmin)
    
    # Passive circular hole at (nelx/3, nely/2)
    passive = np.zeros((nely, nelx), dtype=bool, order='F')
    for i in range(1, nelx + 1):
        for j in range(1, nely + 1):
            if np.sqrt((j - nely / 2.0)**2 + (i - nelx / 3.0)**2) < nely / 3.0:
                passive[j - 1, i - 1] = True
                
    x = np.full((nely, nelx), volfrac, order='F')
    x[passive] = 0.001
    xPhys = x.copy()
    loop = 0
    change = 1.0
    
    history_c = []
    t0 = time.time()
    target_vol = volfrac * nelx * nely
    while change > tol and loop < max_iter:
        loop += 1
        x_flat = xPhys.flatten(order='F')
        E0_flat = E0.flatten(order='F')
        Emin_flat = Emin.flatten(order='F')
        
        sK_mat = KE.flatten()[:, None] * (x_flat**penal * E0_flat + (1.0 - x_flat**penal) * Emin_flat)[None, :]
        sK = sK_mat.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0
        
        U = np.zeros((ndof, 2))
        U[freedofs, 0] = spsolve(K[freedofs][:, freedofs], F[freedofs, 0])
        U[freedofs, 1] = spsolve(K[freedofs][:, freedofs], F[freedofs, 1])
        
        c = 0.0
        dc = np.zeros((nely, nelx), order='F')
        for i_case in range(2):
            Ui = U[:, i_case]
            U_e = Ui[edofMat]
            ce = np.sum((U_e @ KE) * U_e, axis=1).reshape((nely, nelx), order='F')
            c += np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
            dc += -penal * (E0 - Emin) * (xPhys**(penal - 1.0)) * ce
            
        dv = np.ones((nely, nelx))
        dc_filt = (H.dot(dc.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        dv_filt = (H.dot(dv.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            step_val = x * np.sqrt(np.maximum(0.0, -dc_filt / (dv_filt * lmid + 1e-12)))
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, step_val))))
            xnew[passive] = 0.001
            xPhys_eval = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
            xPhys_eval[passive] = 0.001
            vol_eval = np.sum(xPhys_eval)
            if vol_eval > target_vol:
                l1 = lmid
            else:
                l2 = lmid
                
        xPhys = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
        xPhys[passive] = 0.001
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        history_c.append(c)
        if loop % 10 == 0 or loop == 1 or change <= tol:
            print(f" It.: {loop:4d} | Obj.: {c:12.4f} | Vol.: {np.mean(xPhys):6.3f} | Change: {change:8.4f}", flush=True)
            
    elapsed = time.time() - t0
    print(f"--> Done in {loop} iterations ({elapsed:.2f}s). Final Obj: {c:.4f}, Vol: {np.mean(xPhys):.3f}", flush=True)
    return {"name": "mult_load_WIthHole", "xPhys": xPhys, "c": c, "iters": loop, "history_c": history_c, "time": elapsed}


# =========================================================================
# CODE 4: TBP_withHole.m
# =========================================================================
def run_tbp_with_hole(max_iter=150, tol=0.01):
    print("\n" + "="*70, flush=True)
    print(" RUNNING: 4. TBP_withHole.m (Prof. Pavan Asur)", flush=True)
    print("="*70, flush=True)
    nelx = 200
    nely = 100
    volfrac = 0.5
    rmin = 4.0
    penal = 3.0
    nu = 0.3
    
    E1 = 210.0
    E2 = E1 * 10.0
    VF_alp = 4.0
    
    E0 = np.zeros((nely, nelx), order='F')
    for elx in range(1, nelx + 1):
        for ely in range(1, nely + 1):
            E0[ely - 1, elx - 1] = E1 + (E2 - E1) * (((ely - 1) / nely)**VF_alp)
    E0[-1, :] = E2
    Emin = np.full((nely, nelx), 0.001, order='F')
    
    KE = get_88line_ke(nu)
    edofMat, iK, jK = build_mesh_indices(nelx, nely)
    
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros(ndof)
    F[1] = -1.0
    
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([ndof - 1]))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    
    H, Hs, w_vol = build_filter(nelx, nely, rmin)
    
    # Passive circular hole at (nelx/1.5, nely/2)
    passive = np.zeros((nely, nelx), dtype=bool, order='F')
    for i in range(1, nelx + 1):
        for j in range(1, nely + 1):
            if np.sqrt((j - nely / 2.0)**2 + (i - nelx / 1.5)**2) < nely / 3.0:
                passive[j - 1, i - 1] = True
                
    x = np.full((nely, nelx), volfrac, order='F')
    x[passive] = 0.001
    xPhys = x.copy()
    loop = 0
    change = 1.0
    
    history_c = []
    t0 = time.time()
    target_vol = volfrac * nelx * nely
    while change > tol and loop < max_iter:
        loop += 1
        x_flat = xPhys.flatten(order='F')
        E0_flat = E0.flatten(order='F')
        Emin_flat = Emin.flatten(order='F')
        
        sK_mat = KE.flatten()[:, None] * (x_flat**penal * E0_flat + (1.0 - x_flat**penal) * Emin_flat)[None, :]
        sK = sK_mat.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0
        
        U = np.zeros(ndof)
        U[freedofs] = spsolve(K[freedofs][:, freedofs], F[freedofs])
        
        U_e = U[edofMat]
        ce = np.sum((U_e @ KE) * U_e, axis=1).reshape((nely, nelx), order='F')
        c = np.sum((xPhys**penal * E0 + (1.0 - xPhys**penal) * Emin) * ce)
        dc = (E0 - Emin) * (-penal) * (xPhys**(penal - 1.0)) * ce
        dv = np.ones((nely, nelx))
        
        dc_filt = (H.dot(dc.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        dv_filt = (H.dot(dv.flatten(order='F') / Hs)).reshape((nely, nelx), order='F')
        
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            step_val = x * np.sqrt(np.maximum(0.0, -dc_filt / (dv_filt * lmid + 1e-12)))
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, step_val))))
            xnew[passive] = 0.001
            xPhys_eval = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
            xPhys_eval[passive] = 0.001
            vol_eval = np.sum(xPhys_eval)
            if vol_eval > target_vol:
                l1 = lmid
            else:
                l2 = lmid
                
        xPhys = (H.dot(xnew.flatten(order='F')) / Hs).reshape((nely, nelx), order='F')
        xPhys[passive] = 0.001
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        history_c.append(c)
        if loop % 10 == 0 or loop == 1 or change <= tol:
            print(f" It.: {loop:4d} | Obj.: {c:12.6f} | Vol.: {np.mean(xPhys):6.3f} | Change: {change:8.5f}", flush=True)
            
    elapsed = time.time() - t0
    print(f"--> Done in {loop} iterations ({elapsed:.2f}s). Final Obj: {c:.6f}, Vol: {np.mean(xPhys):.3f}", flush=True)
    return {"name": "TBP_withHole", "xPhys": xPhys, "c": c, "iters": loop, "history_c": history_c, "time": elapsed}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    results.append(run_half_three_point_bending())
    results.append(run_mult_load())
    results.append(run_mult_load_with_hole())
    results.append(run_tbp_with_hole())
    
    # Save individual topology plots (winter colormap as in professor's code)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, res in enumerate(results):
        ax = axes[i]
        # In MATLAB: colormap(winter); imagesc(1-xPhys); caxis([0 1]);
        im = ax.imshow(1.0 - res["xPhys"], cmap="winter", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"{i+1}. {res['name']}\nObj: {res['c']:.4f} | Iters: {res['iters']} | Vol: {np.mean(res['xPhys']):.2f}", fontsize=11, fontweight="bold")
        ax.axis("off")
        
        # Save individual plot
        single_fig, single_ax = plt.subplots(figsize=(8, 4.5))
        single_ax.imshow(1.0 - res["xPhys"], cmap="winter", vmin=0.0, vmax=1.0, aspect="equal")
        single_ax.set_title(f"Prof. Asur Reference: {res['name']}\nFinal Compliance: {res['c']:.6f} ({res['iters']} iterations, {res['time']:.1f}s)", fontsize=12, fontweight="bold")
        single_ax.axis("off")
        single_out = os.path.join(out_dir, f"{res['name']}_result.png")
        single_fig.tight_layout()
        single_fig.savefig(single_out, dpi=200)
        plt.close(single_fig)
        
    plt.suptitle("Output Topologies of Prof. Pavan Asur's SIMP Baseline Codes (Without Phase Field)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_plot = os.path.join(out_dir, "prof_codes_summary_grid.png")
    fig.savefig(summary_plot, dpi=200)
    plt.close(fig)
    print(f"\n[Done] All plots saved to {out_dir}/", flush=True)
    print(f"Summary 4-panel visual saved to {summary_plot}", flush=True)


if __name__ == "__main__":
    main()
