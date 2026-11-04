"""
topopt_solver.py
================
Python implementation of Ole Sigmund's seminal 99-line Topology Optimization code
using the SIMP (Solid Isotropic Material with Penalization) method.

Reference:
Ole Sigmund, "A 99 line topology optimization code written in MATLAB",
Structural and Multidisciplinary Optimization, 21, 120-127 (2001).
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def get_element_stiffness(nu=0.3):
    """
    Computes the 8x8 element stiffness matrix k0 for a 4-node 2D plane stress quad.
    E = 1.0 (normalized), nu = Poisson's ratio.
    """
    E = 1.0
    k = np.array([
        1/2 - nu/6,
        1/8 + nu/8,
        -1/4 - nu/12,
        -1/8 + 3*nu/8,
        -1/4 + nu/12,
        -1/8 - nu/8,
        nu/6,
        1/8 - 3*nu/8
    ])
    
    KE = (E / (1 - nu**2)) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


class TopOpt2D:
    """
    2D Topology Optimization solver using SIMP and Finite Element Analysis (FEA).
    """
    def __init__(self, nelx=60, nely=20, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='mbb'):
        """
        nelx: number of elements along X
        nely: number of elements along Y
        volfrac: target volume fraction (e.g. 0.4 for 40% material)
        penal: penalization power p in SIMP (E = x^p * E0)
        rmin: filter radius to eliminate checkerboard patterns
        bc_type: 'mbb' (half MBB beam) or 'cantilever'
        """
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.bc_type = bc_type
        
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = get_element_stiffness(nu=0.3)
        
        # Precompute element degrees of freedom mapping (edofMat)
        self._setup_dofs_and_bcs()
        
        # Precompute convolution filter weights
        self._setup_filter()
        
        # Initialize density field
        self.reset_density()

    def _setup_dofs_and_bcs(self):
        """Precomputes DOF mappings, boundary conditions, and load vector F."""
        nely, nelx = self.nely, self.nelx
        
        # Node index grid
        nodenrs = np.arange(1, (nelx + 1) * (nely + 1) + 1).reshape((nelx + 1, nely + 1)).T
        
        # Element corner nodes
        edof = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = nodenrs[ely, elx]
                n2 = nodenrs[ely, elx + 1]
                n3 = nodenrs[ely + 1, elx + 1]
                n4 = nodenrs[ely + 1, elx]
                
                # DOFs for nodes: 2*node-2 (X) and 2*node-1 (Y) in 0-based indexing
                edof[el, :] = [
                    2*n1 - 2, 2*n1 - 1,
                    2*n2 - 2, 2*n2 - 1,
                    2*n3 - 2, 2*n3 - 1,
                    2*n4 - 2, 2*n4 - 1
                ]
        self.edofMat = edof
        
        # Global coordinate vectors for sparse assembly
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten().astype(int)
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten().astype(int)
        
        # Boundary Conditions and Loads
        self.F = np.zeros(self.ndof)
        
        if self.bc_type == 'mbb':
            # Half MBB Beam:
            # Roller along left vertical edge (fix X disp)
            fixed_left_x = 2 * nodenrs[:, 0] - 2
            # Bottom-right corner pinned vertically (fix Y disp)
            fixed_bottom_right_y = 2 * nodenrs[-1, -1] - 1
            self.fixed_dofs = np.unique(np.concatenate([fixed_left_x, [fixed_bottom_right_y]]))
            # Downward point load at top-left corner
            self.F[2 * nodenrs[0, 0] - 1] = -1.0
            
        elif self.bc_type == 'cantilever':
            # Fixed wall along entire left edge (fix both X and Y disp)
            fixed_left = np.concatenate([2 * nodenrs[:, 0] - 2, 2 * nodenrs[:, 0] - 1])
            self.fixed_dofs = np.unique(fixed_left)
            # Downward point load at center of right edge
            mid_y_node = nodenrs[nely // 2, -1]
            self.F[2 * mid_y_node - 1] = -1.0
        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")
            
        self.all_dofs = np.arange(self.ndof)
        self.free_dofs = np.setdiff1d(self.all_dofs, self.fixed_dofs)

    def _setup_filter(self):
        """Precomputes the mesh-independency sensitivity filter weights."""
        nelx, nely, rmin = self.nelx, self.nely, self.rmin
        self.H = np.zeros((nelx * nely, nelx * nely))
        self.Hs = np.zeros(nelx * nely)
        
        for i in range(nelx):
            for j in range(nely):
                e1 = i * nely + j
                i_min = max(i - int(np.floor(rmin)), 0)
                i_max = min(i + int(np.floor(rmin)) + 1, nelx)
                j_min = max(j - int(np.floor(rmin)), 0)
                j_max = min(j + int(np.floor(rmin)) + 1, nely)
                
                for k in range(i_min, i_max):
                    for l in range(j_min, j_max):
                        e2 = k * nely + l
                        dist = np.sqrt((i - k)**2 + (j - l)**2)
                        weight = max(0.0, rmin - dist)
                        self.H[e1, e2] = weight
                        self.Hs[e1] += weight

    def reset_density(self, init_val=None):
        """Resets design variable grid x (shape: nely, nelx)."""
        if init_val is None:
            init_val = self.volfrac
        self.x = np.full((self.nely, self.nelx), init_val, dtype=np.float64)
        return self.x.copy()

    def solve_fea(self, x=None):
        """
        Solves K(x) * U = F using Finite Element Analysis.
        Returns:
            U: global displacement vector (size ndof)
            c: compliance (scalar)
            dc: raw sensitivity (gradient) array (size nely, nelx)
        """
        if x is None:
            x = self.x
            
        x_flat = x.T.flatten()  # Column-major order matching MATLAB element ordering
        
        # SIMP material interpolation: E_min = 1e-3 (avoids singular matrix)
        E_min = 1e-3
        E_0 = 1.0
        E_elem = E_min + (x_flat ** self.penal) * (E_0 - E_min)
        
        # Assemble global sparse stiffness matrix K
        sK = (self.KE.flatten()[None, :] * E_elem[:, None]).flatten()
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        
        # Extract free DOFs submatrix and solve
        K_free = K[self.free_dofs, :][:, self.free_dofs]
        F_free = self.F[self.free_dofs]
        
        U = np.zeros(self.ndof)
        U[self.free_dofs] = spsolve(K_free, F_free)
        
        # Compute compliance C and sensitivities dc
        # c = sum_e (x_e^p * u_e^T * k0 * u_e)
        U_elem = U[self.edofMat]  # shape (nelx*nely, 8)
        # u_e^T * k0 * u_e
        ce = np.sum((U_elem @ self.KE) * U_elem, axis=1)
        
        c = np.sum(E_elem * ce)
        dc_flat = -self.penal * (E_0 - E_min) * (x_flat ** (self.penal - 1)) * ce
        dc = dc_flat.reshape((self.nelx, self.nely)).T
        
        return U, c, dc

    def filter_sensitivities(self, x, dc):
        """Applies mesh-independency filter to sensitivities dc."""
        x_flat = x.T.flatten()
        dc_flat = dc.T.flatten()
        
        # dcn = (H * (x * dc)) / (Hs * x)
        dcn_flat = (self.H @ (x_flat * dc_flat)) / (self.Hs * np.maximum(x_flat, 1e-3))
        dcn = dcn_flat.reshape((self.nelx, self.nely)).T
        return dcn

    def optimality_criteria_update(self, x, dc, move=0.2):
        """
        Updates densities using the classic Optimality Criteria (OC) bisection method.
        """
        l1 = 0.0
        l2 = 1e9
        
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4 and (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            # OC update formula
            x_step = x * np.sqrt(np.maximum(-dc / (lmid + 1e-12), 1e-12))
            x_new = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x_step))))
            
            if np.mean(x_new) > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid
                
        return x_new

    def run_optimization(self, max_iters=50, tol=0.01, verbose=True):
        """
        Runs the full classical SIMP optimization loop.
        """
        self.reset_density()
        history = {'compliance': [], 'change': []}
        
        if verbose:
            print(f"=== Starting Classical SIMP TopOpt ({self.bc_type.upper()}) ===")
            print(f"Grid: {self.nelx}x{self.nely} | Target Volfrac: {self.volfrac:.2f} | Penal: {self.penal}")
            print(f"{'Iter':<6}{'Compliance':<15}{'Mean Density':<15}{'Change':<12}")
            print("-" * 48)
            
        for it in range(1, max_iters + 1):
            x_old = self.x.copy()
            
            # 1. FEA solve
            U, c, dc = self.solve_fea(self.x)
            
            # 2. Filter sensitivities
            dc_filtered = self.filter_sensitivities(self.x, dc)
            
            # 3. OC update
            self.x = self.optimality_criteria_update(self.x, dc_filtered)
            
            change = np.max(np.abs(self.x - x_old))
            history['compliance'].append(c)
            history['change'].append(change)
            
            if verbose and (it == 1 or it % 5 == 0 or change < tol):
                print(f"{it:<6}{c:<15.4f}{np.mean(self.x):<15.4f}{change:<12.4f}")
                
            if change < tol:
                if verbose:
                    print(f"--> Converged at iteration {it} with change = {change:.4f}")
                break
                
        return history

    def plot_structure(self, save_path=None, title="Topology Optimization Result"):
        """Plots the binary/density distribution of the structure."""
        fig, ax = plt.subplots(figsize=(8, 4))
        # 0 = void (white), 1 = solid (black)
        im = ax.imshow(1 - self.x, cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("X elements")
        ax.set_ylabel("Y elements")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"[Saved] Structure figure saved to {save_path}")
        plt.close()
