"""
generate_presentation.py
========================
Regenerates BTP_TopOpt_RL_Progress_Praveen.pptx with fully audited,
rigorous numbers adhering to Claude's review and SPEC.md.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    c_navy = RGBColor(16, 44, 87)
    c_blue = RGBColor(53, 95, 142)
    c_dark = RGBColor(33, 37, 41)
    
    blank_layout = prs.slide_layouts[6]
    
    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p1 = tf1.paragraphs[0]
    p1.text = "Deep Reinforcement Learning Framework for\n2D Structural Topology Optimization"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = c_navy
    
    p2 = tf1.add_paragraph()
    p2.text = "BTP Progress Update & Research Workflow Blueprint"
    p2.font.size = Pt(22)
    p2.font.color.rgb = c_blue
    p2.space_before = Pt(16)
    
    tb1_sub = s1.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(11.3), Inches(1.5))
    p3 = tb1_sub.text_frame.paragraphs[0]
    p3.text = "Student: Praveen Kumar  |  Department of Mechanical Engineering, IIT Kharagpur\nSupervisor: Prof. Asur Vijaya Kumar"
    p3.font.size = Pt(16)
    p3.font.color.rgb = c_dark
    
    # -------------------------------------------------------------
    # SLIDE 2: STAGE 1 - CLASSICAL SIMP VALIDATION
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    p = tb2.text_frame.paragraphs[0]
    p.text = "Stage 1: Classical SIMP Baseline & FEA Physics Validation"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c_navy
    
    tb2_body = s2.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.5))
    tf2 = tb2_body.text_frame
    tf2.word_wrap = True
    
    bullets = [
        ("Core Mechanics Verified:", True),
        ("• Minimum compliance C = U^T K U subject to V <= 0.4*V0 and K U = F.", False),
        ("• SIMP Penalization: E(x) = Emin + x^p (E0 - Emin) with p = 3, Emin = 1e-9.", False),
        ("• Spatial Filtering: Density & sensitivity filters with rmin = 1.5 elements.", False),
        ("FEA Physics Engine Audited (src/fem/):", True),
        ("• Element stiffness k0: Route 1 (analytical) == Route 2 (2x2 Gauss quad) to < 1.11e-16.", False),
        ("• Energy balance exact: U^T K U == F^T U (relative error < 5.4e-11).", False),
        ("• 41 automated pytest tests passing across all FEM and SIMP modules.", False),
    ]
    for text, bold in bullets:
        p = tf2.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if not bold else 15)
        p.font.color.rgb = c_navy if bold else c_dark
        p.space_after = Pt(4)
        
    if os.path.exists("cantilever_optimized.png"):
        s2.shapes.add_picture("cantilever_optimized.png", Inches(7.2), Inches(1.5), width=Inches(5.3))
    if os.path.exists("mbb_beam_optimized.png"):
        s2.shapes.add_picture("mbb_beam_optimized.png", Inches(7.2), Inches(4.3), width=Inches(5.3))
        
    # -------------------------------------------------------------
    # SLIDE 3: STAGE 1 - AUDITED BASELINES TABLE
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    p = tb3.text_frame.paragraphs[0]
    p.text = "Reference Ground Truth: baselines.csv (Audited with Emin=1e-9)"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c_navy
    
    tb3_body = s3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.2))
    tf3 = tb3_body.text_frame
    tf3.word_wrap = True
    
    p = tf3.paragraphs[0]
    p.text = "Gate 1 Cleared: True Convergence & Grayness Reported"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = c_blue
    
    b_rows = [
        "True mathematical convergence (tol=0.005) with fresh FEA solves:",
        "",
        "• Cantilever (30x10, V=0.4): C = 399.38 | 95 iters | Gray: 66.7%",
        "• Cantilever (60x20, V=0.4): C = 247.80 | 250 iters| Gray: 33.5%",
        "• Half-MBB (30x10, V=0.5):   C = 225.67 | 113 iters| Gray: 39.7%",
        "• Half-MBB (60x20, V=0.5):   C = 203.17 | 164 iters| Gray: 21.0%",
        "",
        "Note on 30x10: Higher grayness (66.7%) is expected on coarse meshes",
        "under rmin=1.5 and serves as an honest baseline for RL comparison."
    ]
    for line in b_rows:
        p = tf3.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = c_dark
        p.space_after = Pt(3)
        
    if os.path.exists("convergence_history.png"):
        s3.shapes.add_picture("convergence_history.png", Inches(7.0), Inches(1.6), width=Inches(5.6))

    # -------------------------------------------------------------
    # SLIDE 4: STAGE 2 - FORMAL MDP SPECIFICATION
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    p = tb4.text_frame.paragraphs[0]
    p.text = "Stage 2: Topology Optimization as an MDP (Audited Specification)"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c_navy
    
    tb4_body = s4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))
    tf4 = tb4_body.text_frame
    tf4.word_wrap = True
    
    mdp_points = [
        ("State Space S_t (4-Channel 2D Tensor):", True),
        ("  • Channel 0: Physical Density Map x_phys in [xmin, 1.0]", False),
        ("  • Channel 1: Normalized Strain Energy Density w_e = 0.5 * u_e^T * k0 * u_e", False),
        ("  • Channels 2 & 3: Load location mask M_F and Support constraint mask M_U", False),
        ("Action Space A_t (Bounded Continuous Delta):", True),
        ("  • Policy outputs a_t in [-1.0, 1.0], scaled inside env by delta_max = 0.05.", False),
        ("  • Filtered via density filter on state to ensure physical feature size.", False),
        ("Hard Volume Constraint Enforcement (Exact Bisection Shift):", True),
        ("  • Bisection shift s ensures mean(clip(x + s, xmin, 1.0)) == volfrac to < 1e-6 error.", False),
        ("Reward Function R_t (C0-Normalized with Fixed Horizon):", True),
        ("  • r_t = -(C_t - C_t-1) / C0 - lambda * |V_t - V*|, clipped to [-10, 10].", False),
        ("  • Fixed-horizon truncation ONLY at T_max = 30 (no early-stopping loopholes).", False)
    ]
    for text, bold in mdp_points:
        p = tf4.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if not bold else 15)
        p.font.color.rgb = c_navy if bold else c_dark
        p.space_after = Pt(3)

    # -------------------------------------------------------------
    # SLIDE 5: STAGE 3 - AUDITED SANITY BASELINES & PPO
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    p = tb5.text_frame.paragraphs[0]
    p.text = "Stage 3: Audited Sanity Baselines & Gate 3 Clearance"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c_navy
    
    tb5_body = s5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.2))
    tf5 = tb5_body.text_frame
    tf5.word_wrap = True
    
    p = tf5.paragraphs[0]
    p.text = "Benchmark Hierarchy (Cantilever 30x10, Emin=1e-9):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = c_blue
    
    s_rows = [
        "1. Random Exploration (3 seeded runs):   C = 2395.38 ± 295.42",
        "2. Uniform Baseline (x_e = 0.40):         C = 1826.10",
        "3. Trained PPO Agent (Gate 3):            C = 1054.34  [42% reduction!]",
        "4. Canonical Sensitivity Heuristic:      C = 609.84",
        "5. Converged Classical SIMP:             C = 399.38",
        "",
        "Gate 3 Verdict: PASSED",
        "• PPO trained with log_std_init = -1.5 beats Random (1054 vs 2395)",
        "  and beats Uniform (1054 vs 1826) by a wide margin in 20k steps.",
        "• Moving to Stage 4: Fully Convolutional policy to close gap with SIMP."
    ]
    for line in s_rows:
        p = tf5.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = c_dark
        p.space_after = Pt(3)
        
    if os.path.exists("rl_environment_comparison.png"):
        s5.shapes.add_picture("rl_environment_comparison.png", Inches(7.0), Inches(1.8), width=Inches(5.7))

    # -------------------------------------------------------------
    # SLIDE 6: MEETING AGENDA & DECISIONS WITH PROF. ASUR
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    p = tb6.text_frame.paragraphs[0]
    p.text = "Meeting Agenda & Research Scope Alignment (Section 5b)"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c_navy
    
    tb6_body = s6.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))
    tf6 = tb6_body.text_frame
    tf6.word_wrap = True
    
    agenda = [
        ("1. Progress Demonstrated Today (Gates 1, 2, & 3):", True),
        ("  • Validated classical SIMP baseline: 41 passing unit tests and audited baselines.csv.", False),
        ("  • Formal MDP specification document: 4-channel tensor state, volume projection, C0-normalized reward.", False),
        ("  • Prototype PPO agent beats random and uniform baselines (C = 1054 vs 2395).", False),
        ("2. Open Decisions for Discussion:", True),
        ("  • Residual on a heuristic: PPO policy outputting residuals on top of sensitivity step.", False),
        ("  • FEA-in-the-loop vs. Zero-shot inference: evaluating trade-offs between step-wise FEA and mask-only inference.", False),
        ("3. The Main Scientific Contribution (Stage 5):", True),
        ("  • Generalization across variable load positions and angles.", False),
        ("  • Inclusion of a Supervised-CNN baseline on converged SIMP solutions for fair comparison.", False),
        ("4. Departmental Logistics:", True),
        ("  • Access to GPU compute cluster for multi-seed Stage 5 training.", False)
    ]
    for text, bold in agenda:
        p = tf6.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if not bold else 15)
        p.font.color.rgb = c_navy if bold else c_dark
        p.space_after = Pt(3)
        
    out_file = "BTP_TopOpt_RL_Progress_Praveen.pptx"
    prs.save(out_file)
    print(f"[Saved] Professional presentation created at {out_file}")

if __name__ == "__main__":
    create_deck()
