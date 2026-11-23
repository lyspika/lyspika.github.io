import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
import matplotlib.pyplot as plt

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98847e30
MTSUN = G * M_SUN / C**3

def compute_charge_params(eta, k1, k2):
    disc = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * eta))
    chi1 = 0.5 * (1.0 + disc)
    chi2 = 0.5 * (1.0 - disc)
    Q = k1 * k2
    Z = 1.0 - Q
    zeta0 = chi1 * k1**2 + chi2 * k2**2
    xi0 = (k1 * chi1 + k2 * chi2) * (chi1 - chi2)
    xi1 = k1 - k2
    xi2 = k1 * chi2 - k2 * chi1
    xi3 = k1 * chi2**2 - k2 * chi1**2
    zeta2 = k1 * chi2 + k2 * chi1
    zeta3 = k1 * chi2**2 + k2 * chi1**2
    zeta4 = k1 * chi2**3 + k2 * chi1**3
    return {
        'chi1': chi1, 'chi2': chi2, 'Q': Q, 'Z': Z, 'zeta0': zeta0,
        'xi0': xi0, 'xi1': xi1, 'xi2': xi2, 'xi3': xi3,
        'zeta2': zeta2, 'zeta3': zeta3, 'zeta4': zeta4
    }

def compute_Ca_Da(eta, p):
    z0 = p['zeta0']
    xi0, xi1, xi2, xi3 = p['xi0'], p['xi1'], p['xi2'], p['xi3']
    z2, z3, z4 = p['zeta2'], p['zeta3'], p['zeta4']
    
    C0 = xi1**2 * (5*z0/3 + 5*eta/3 - 1) + 24*z2**2/5 - 2*xi0*xi1 - 8*xi1*xi2
    C1 = xi1**2 * (-z0 + 4*eta + 6) - 48*z2**2/5 + 2*xi0*xi1 + xi1*(8*xi2 - 2*xi3) - 192/5
    C2 = xi1**2 * (4*z0/3 - 11*eta - 28/3) + 24*z2**2/5 + 2*xi1*xi3 + 96/5
    C3 = xi1**2 * (16*eta/3 + 1/3)
    C_list = [C0, C1, C2, C3]
    
    D0 = (xi1**2 * (z0*(11*eta/6 - 13/6) - 16*z0**2/9 + eta**2/2 + eta + 0.5)
          + xi0*xi1*(4*z0/3 - 5*eta/3 + 1) + xi1*xi2*(16*z0/3 - 20*eta/3 + 4)
          + z2**2*(36*eta/5 - 12) + (-96*z3/5 - 16*z4/7)*z2
          + z0*(6*z2**2 + 16/5) + (7314*eta/175 - 3657/350)*xi3**2)
    
    D1 = (xi1**2 * (z0*(103/6 - 7*eta/6) + 10*z0**2/9 + 11*eta**2/3 + 3*eta + 10)
          + xi0*xi1*(-4*z0/3 - eta/3 - 13) + xi1*xi2*(-16*z0/3 - 4*eta/3 - 52)
          + xi1*xi3*(4*z0/3 - 5*eta/3 + 1) + z2**2*(228/5 - 12*eta/5)
          + (288*z3/5 + 64*z4/7)*z2 + z0*(176/5 - 14*z2**2)
          + (7314/175 - 29256*eta/175)*xi3**2 + 1584*eta/5 + 6952/35)
    
    D2 = (xi1**2 * (z0*(-17*eta/2 - 143/6) + 4*z0**2/3 - 14*eta**2/3 + 15*eta/2 - 22)
          + xi0*xi1*(-4*z0/3 + 11*eta + 73/3) + xi1*xi2*(-16*z0/3 + 44*eta + 292/3)
          + xi1*xi3*(-4*z0/3 - eta/3 - 13) + z2**2*(-244*eta/5 - 402/5)
          + (-288*z3/5 - 96*z4/7)*z2 + (66*z2**2/5 + 96/5)*z0
          + (43884*eta/175 - 10971/175)*xi3**2 - 3264*eta/5 - 220/7)
    
    D3 = (xi1**2 * (z0*(37*eta/2 + 39/2) - 14*z0**2/9 - 21*eta**2 - 343*eta/6 - 13/2)
          + xi0*xi1*(4*z0/3 - 43*eta/3 - 41/3) + xi1*xi2*(16*z0/3 - 172*eta/3 - 164/3)
          + xi1*xi3*(-4*z0/3 + 11*eta + 73/3) + z2**2*(444*eta/5 + 366/5)
          + (96*z3/5 + 64*z4/7)*z2 + (-42*z2**2/5 - 208/5)*z0
          + (7314/175 - 29256*eta/175)*xi3**2 + 2928*eta/5 + 7456/35)
    
    D4 = (xi1**2 * (z0*(-16*eta - 34/3) + 8*z0**2/9 + 281*eta**2/6 + 457*eta/6 + 19)
          + xi0*xi1*(16*eta/3 + 4/3) + xi1*xi2*(64*eta/3 + 16/3)
          + xi1*xi3*(4*z0/3 - 43*eta/3 - 41/3) + z2**2*(-288*eta/5 - 138/5)
          - 16*z4*z2/7 + (16*z2**2/5 + 64/5)*z0
          + (7314*eta/175 - 3657/350)*xi3**2 - 216*eta - 398/7)
    
    D5 = (xi1**2 * (z0*(16*eta/3 + 2/3) - 100*eta**2/3 - 191*eta/6 - 5/6)
          + z2**2*(64*eta/5 + 6/5) + (16*eta/3 + 4/3)*xi1*xi3 + 96*eta/5 + 24/5)
    
    D6 = xi1**2 * (8*eta**2 + 4*eta/3 - 1/6)
    D_list = [D0, D1, D2, D3, D4, D5, D6]
    
    return C_list, D_list

def compute_f_ISCO(Mc_solar, eta, k1, k2):
    m_total_sec = (Mc_solar * eta**(-3.0/5.0)) * MTSUN
    p = compute_charge_params(eta, k1, k2)
    x_isco = (p['Z']**(4.0/3.0) / 6.0) * (
        1.0 + p['zeta0']/6.0 + (0.5 + eta)*p['Q']
    )
    return (x_isco**1.5) / (np.pi * m_total_sec)

def omega_dot(f_grid, Mc_solar, eta, k1, k2, pn_order=1, xi1_sq_override=None):
    m_sec = (Mc_solar * eta**(-3.0/5.0)) * MTSUN
    omega = np.pi * f_grid
    x = (m_sec * omega)**(2.0/3.0)
    
    p = compute_charge_params(eta, k1, k2)
    C_list, D_list = compute_Ca_Da(eta, p)
    Q, Z = p['Q'], p['Z']
    
    xi1_sq = p['xi1']**2 if xi1_sq_override is None else xi1_sq_override
    term_m1PN = 2.0 * xi1_sq
    
    if pn_order == -1:
        term_0PN = x * (96.0 / 5.0)
        term_1PN = (x**2) * (-264.0*eta/5.0 - 1486.0/35.0)
    elif pn_order == 0:
        sum_C = sum(C_list[a] * (Q**a) for a in range(4))
        term_0PN = (x / Z**(4.0/3.0)) * (96.0 / 5.0 + sum_C)
        term_1PN = 0.0
    else:
        sum_C = sum(C_list[a] * (Q**a) for a in range(4))
        sum_D = sum(D_list[a] * (Q**a) for a in range(7))
        term_0PN = (x / Z**(4.0/3.0)) * (96.0 / 5.0 + sum_C)
        term_1PN = (x**2 / Z**(8.0/3.0)) * (-264.0*eta/5.0 - 1486.0/35.0 + sum_D)
        
    prefactor = eta * m_sec * (omega**3)
    return prefactor * (term_m1PN + term_0PN + term_1PN)

def aligo_psd(f):
    f0 = 215.0
    x = f / f0
    S0 = 1e-49
    psd = S0 * (x**(-4.14) - 5.0 * x**(-2.0) + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
    return np.maximum(psd, 1e-50)

def spa_phase_and_amp(f_grid, tc, phic, ln_Mc, ln_eta, k1, k2, pn_order=1, xi1_sq_override=None):
    Mc_solar = np.exp(ln_Mc)
    eta = np.exp(ln_eta)
    omega = np.pi * f_grid
    
    wdot = omega_dot(f_grid, Mc_solar, eta, k1, k2, pn_order=pn_order, xi1_sq_override=xi1_sq_override)
    
    inv_wdot_rev = (1.0 / wdot)[::-1]
    w_over_wdot_rev = (omega / wdot)[::-1]
    omega_rev = omega[::-1]
    
    I0_rev = cumulative_trapezoid(inv_wdot_rev, omega_rev, initial=0.0)
    I1_rev = cumulative_trapezoid(w_over_wdot_rev, omega_rev, initial=0.0)
    I0 = -I0_rev[::-1]
    I1 = -I1_rev[::-1]
    
    psi = 2.0 * omega * tc - phic - np.pi / 4.0 + 2.0 * (omega * I0 - I1)
    
    p = compute_charge_params(eta, k1, k2)
    m_sec = (Mc_solar * eta**(-3.0/5.0)) * MTSUN
    x = (m_sec * omega)**(2.0/3.0)
    amp = p['Z'] * m_sec * C * x * np.sqrt(np.pi / wdot)
    
    return psi, amp

def compute_fisher_matrix(f_grid, psd, fiducial, param_names, target_snr=20.0):
    def eval_waveform(p_dict, pn_order=1):
        xi1_sq_ov = p_dict.get('xi1_sq', None)
        psi, amp = spa_phase_and_amp(
            f_grid, p_dict['tc'], p_dict['phic'], p_dict['ln_Mc'], p_dict['ln_eta'],
            p_dict['k1'], p_dict['k2'], pn_order=pn_order, xi1_sq_override=xi1_sq_ov
        )
        return psi, amp
        
    psi0, amp0 = eval_waveform(fiducial, pn_order=1)
    
    raw_snr_sq = 4.0 * simpson((amp0**2) / psd, x=f_grid)
    norm_factor = target_snr / np.sqrt(raw_snr_sq)
    amp0 *= norm_factor
    
    steps = {
        'tc': 1e-6, 'phic': 1e-6, 'ln_Mc': 1e-6, 'ln_eta': 1e-6,
        'xi1_sq': 1e-7, 'xi1': 1e-6, 'zeta2': 1e-5, 'Q': 1e-5
    }
    
    n_params = len(param_names)
    dpsi_list = []
    damp_list = []
    
    for name in param_names:
        eps = steps[name]
        p_plus = fiducial.copy()
        p_minus = fiducial.copy()
        
        if name in ['tc', 'phic', 'ln_Mc', 'ln_eta', 'xi1_sq']:
            p_plus[name] = fiducial.get(name, 0.0) + eps
            p_minus[name] = fiducial.get(name, 0.0) - eps
        elif name == 'xi1':
            p_plus['k1'] += 0.5 * eps
            p_plus['k2'] -= 0.5 * eps
            p_minus['k1'] -= 0.5 * eps
            p_minus['k2'] += 0.5 * eps
        elif name == 'zeta2':
            p_plus['k1'] += eps
            p_plus['k2'] += eps
            p_minus['k1'] -= eps
            p_minus['k2'] -= eps
            
        psi_p, amp_p = eval_waveform(p_plus, pn_order=1)
        psi_m, amp_m = eval_waveform(p_minus, pn_order=1)
        
        dpsi_list.append((psi_p - psi_m) / (2.0 * eps))
        damp_list.append(norm_factor * (amp_p - amp_m) / (2.0 * eps))
        
    fisher = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            integrand = (damp_list[i] * damp_list[j] + (amp0**2) * dpsi_list[i] * dpsi_list[j]) / psd
            val = 4.0 * simpson(integrand, x=f_grid)
            fisher[i, j] = val
            fisher[j, i] = val
            
    for i, name in enumerate(param_names):
        if name in ['xi1', 'zeta2', 'xi1_sq']:
            fisher[i, i] += 1.0 / (1.0**2)
            
    cov = np.linalg.pinv(fisher, rcond=1e-15)
    
    psi_approx, _ = eval_waveform(fiducial, pn_order=-1)
    delta_psi = psi0 - psi_approx
    sys_vec = np.array([
        4.0 * simpson(((amp0**2) * dpsi_list[j] * delta_psi) / psd, x=f_grid)
        for j in range(n_params)
    ])
    sys_bias = cov @ sys_vec
    
    return fisher, cov, sys_bias

def main():
    m1, m2 = 10.0, 10.0
    m_tot = m1 + m2
    eta0 = (m1 * m2) / (m_tot**2)
    eta_fid = 0.2499
    Mc_fid = m_tot * (eta_fid**(3.0/5.0))
    snr_target = 20.0
    
    cases = [
        {
            "label": "Case (a): Neutral BHs (Null Test)",
            "k1": 0.0, "k2": 0.0,
            "params": ["tc", "phic", "ln_Mc", "ln_eta", "xi1_sq"]
        },
        {
            "label": "Case (b): Charged + Neutral BH",
            "k1": 0.6, "k2": 0.0,
            "params": ["tc", "phic", "ln_Mc", "ln_eta", "xi1", "zeta2"]
        },
        {
            "label": "Case (c): Equal Same-Sign Charges",
            "k1": 0.6, "k2": 0.6,
            "params": ["tc", "phic", "ln_Mc", "ln_eta", "zeta2"]
        },
        {
            "label": "Case (d): Equal Opposite-Sign Charges",
            "k1": 0.6, "k2": -0.6,
            "params": ["tc", "phic", "ln_Mc", "ln_eta", "xi1", "zeta2"]
        }
    ]
    
    print("=" * 82)
    print(f"FISHER INFORMATION MATRIX ANALYSIS: m1={m1} M_sun, m2={m2} M_sun, SNR={snr_target}")
    print("=" * 82)
    
    results_store = {}
    
    for case in cases:
        k1, k2 = case["k1"], case["k2"]
        f_isco = compute_f_ISCO(Mc_fid, eta_fid, k1, k2)
        f_grid = np.geomspace(10.0, f_isco, 4000)
        psd = aligo_psd(f_grid)
        
        fid = {
            'tc': 0.0, 'phic': 0.0,
            'ln_Mc': np.log(Mc_fid), 'ln_eta': np.log(eta_fid),
            'k1': k1, 'k2': k2
        }
        if "xi1_sq" in case["params"]:
            fid['xi1_sq'] = (k1 - k2)**2
            
        fisher, cov, sys_bias = compute_fisher_matrix(
            f_grid, psd, fid, case["params"], target_snr=snr_target
        )
        sigmas = np.sqrt(np.diag(cov))
        cond_num = np.linalg.cond(fisher)
        
        results_store[case["label"]] = (case["params"], sigmas, cov)
        
        print(f"\n>>> {case['label']}  [k1={k1:+.1f}, k2={k2:+.1f} | f_ISCO = {f_isco:.1f} Hz]")
        print(f"    Matrix Condition Number: {cond_num:.2e}")
        print(f"    {'Parameter':<10} | {'1-sigma Uncertainty':<22} | {'Bias if NLO/1PN Charge Omitted'}")
        print("    " + "-" * 70)
        for idx, pname in enumerate(case["params"]):
            unit = " [s]" if pname == "tc" else (" [rad]" if pname == "phic" else "")
            print(f"    {pname + unit:<10} | {sigmas[idx]:<22.4e} | {sys_bias[idx]:+.4e}")
            
        if "xi1_sq" in case["params"]:
            bound_dk = np.sqrt(sigmas[case["params"].index("xi1_sq")])
            print(f"    => Implied 1-sigma upper bound on |k1 - k2|: <= {bound_dk:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    
    params_b, sigmas_b, cov_b = results_store["Case (b): Charged + Neutral BH"]
    corr_b = cov_b / np.outer(sigmas_b, sigmas_b)
    im = axes[0].imshow(corr_b, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(params_b)))
    axes[0].set_yticks(range(len(params_b)))
    pretty_labels = [r"$t_c$", r"$\phi_c$", r"$\ln\mathcal{M}_c$", r"$\ln\eta$",
                     r"$\xi_1 (\kappa_1-\kappa_2)$", r"$\bar{\kappa} (\frac{\kappa_1+\kappa_2}{2})$"]
    axes[0].set_xticklabels(pretty_labels, fontsize=11)
    axes[0].set_yticklabels(pretty_labels, fontsize=11)
    for i in range(len(params_b)):
        for j in range(len(params_b)):
            axes[0].text(j, i, f"{corr_b[i, j]:+.2f}", ha="center", va="center",
                         color="white" if abs(corr_b[i, j]) > 0.65 else "black", fontsize=9)
    axes[0].set_title("Correlation Matrix: Case (b) ($\kappa_1=0.6, \kappa_2=0$)", fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    params_a, sigmas_a, cov_a = results_store["Case (a): Neutral BHs (Null Test)"]
    idx_mc = params_a.index("ln_Mc")
    idx_xi = params_a.index("xi1_sq")
    sub_cov = cov_a[np.ix_([idx_mc, idx_xi], [idx_mc, idx_xi])]
    
    vals, vecs = np.linalg.eigh(sub_cov)
    theta_grid = np.linspace(0, 2*np.pi, 300)
    
    for n_sig, ls, lbl in [(1.0, "-", r"$1\sigma$ (68% CL)"), (np.sqrt(4.61), "--", r"$90\%$ CL ($\Delta\chi^2=4.61$)")]:
        ell = vecs @ (np.sqrt(vals)[:, None] * np.array([np.cos(theta_grid), np.sin(theta_grid)])) * n_sig
        axes[1].plot(ell[0]*1e3, ell[1]*1e4, linestyle=ls, lw=2, label=lbl)
        
    axes[1].axhline(0, color="gray", lw=0.8, ls=":")
    axes[1].axvline(0, color="gray", lw=0.8, ls=":")
    axes[1].set_xlabel(r"Fractional Chirp Mass Error $\Delta\mathcal{M}_c / \mathcal{M}_c \times 10^3$", fontsize=11)
    axes[1].set_ylabel(r"Dipole Parameter $\xi_1^2 = (\kappa_1 - \kappa_2)^2 \times 10^4$", fontsize=11)
    axes[1].set_title(r"Null-Test Degeneracy Ellipse: Case (a) ($\kappa_1=\kappa_2=0$)", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("fig1_fisher_results.png", dpi=300)
    print("\nSaved correlation heatmap and Fisher confidence ellipse to 'fig1_fisher_results.png'.")

if __name__ == "__main__":
    main()
