import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import tqdm

def input_priors_csv(file_name):
    input_priors = pd.read_csv(file_name)
    C = len(input_priors) # Number of segments

    w_c = input_priors.w_c.values # Relative segment sizes (sum to 1)
    mu_c = input_priors.mu_c.values # Prior means of the treatment effects (by segment)
    sigma_c = input_priors.sigma_c.values # Prior std dev of the treatment effects (by segment)
    s_c = input_priors.s_c.values # Customer outcome variances (by segment)

    assert (1-sum(w_c))<1e-3
    assert sum(w_c>=0) == C
    
    print(f"Number of segments = %d"%C)
    print(f"Average Treatment Effect = %.2f"%np.sum(w_c*mu_c))
    
    return w_c,mu_c,sigma_c,s_c

def get_expected_perf(mu_c, sigma_c, s_c, w_c, n_tr):
    if (n_tr == 0):
        mu_delta = np.sum(w_c*mu_c*(mu_c > 0))
        var_delta = 0
    else:
        k_c = np.sqrt(1+4*s_c**2/w_c/n_tr/(sigma_c**2))
        pdf_c = norm.pdf(mu_c*k_c/sigma_c)
        cdf_c = norm.cdf(mu_c*k_c/sigma_c)

        mu_delta = np.sum(w_c*(sigma_c/k_c*pdf_c+mu_c*cdf_c))
        var_delta = np.sum(w_c**2*(mu_c**2*cdf_c*(1-cdf_c)+sigma_c **
                           2/k_c**2*(cdf_c-pdf_c**2)+mu_c*sigma_c/k_c*pdf_c*(1-2*cdf_c)))

    return mu_delta, var_delta

def get_expected_perf_limit(mu_c, sigma_c, s_c, w_c):
    pdf_c = norm.pdf(mu_c/sigma_c)
    cdf_c = norm.cdf(mu_c/sigma_c)
    mu_limit = np.sum(w_c*(sigma_c*pdf_c+mu_c*cdf_c))
    return mu_limit


def solve_d_expected_improvement_search(delta, mu_c, sigma_c, s_c, w_c, N_MIN, N_MAX):
    if (N_MAX // 1)==(N_MIN // 1):
        return N_MIN

    N_MID = (N_MIN+N_MAX)/2
    
    mid_val = get_expected_perf(mu_c, sigma_c, s_c, w_c, N_MID)[0]-delta

    if mid_val <= 0:
        return solve_d_expected_improvement_search(delta, mu_c, sigma_c, s_c, w_c, N_MID, N_MAX)
    else:
        return solve_d_expected_improvement_search(delta, mu_c, sigma_c, s_c, w_c, N_MIN, N_MID)


def solve_d_expected_improvement(delta, mu_c, sigma_c, s_c, w_c):
    # Algorithm 1
    mu_delta_0, _ = get_expected_perf(mu_c, sigma_c, s_c, w_c, 0)
    if mu_delta_0 >= delta:
        return 0

    mu_delta_limit = get_expected_perf_limit(mu_c, sigma_c, s_c, w_c)
    if mu_delta_limit <= delta:
        return -1
    
    step = 1e7
    ntr_curr = step
    while (get_expected_perf(mu_c, sigma_c, s_c, w_c, ntr_curr)[0]<=delta):
        ntr_curr += step

    n_opt = solve_d_expected_improvement_search(
        delta, mu_c, sigma_c, s_c, w_c, ntr_curr-step, ntr_curr)
    return math.ceil(n_opt)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def stats_g_prob_d_expected_improvement(delta, gamma, mu_c, sigma_c, s_c, w_c, ntr_set, B):
    if (B==0):
        mu_c_expand = mu_c[:, np.newaxis]
        sigma_c_expand = sigma_c[:, np.newaxis]
        s_c_expand = s_c[:, np.newaxis]
        w_c_expand = w_c[:, np.newaxis]
        ntr_expand = ntr_set[np.newaxis, :]

        k_c = np.sqrt(1+4*s_c_expand**2/w_c_expand /
                      (ntr_expand+1e-30)/(sigma_c_expand**2))
        pdf_c = norm.pdf(mu_c_expand*k_c/sigma_c_expand)
        cdf_c = norm.cdf(mu_c_expand*k_c/sigma_c_expand)

        mu_delta = np.sum(w_c_expand*(sigma_c_expand/k_c *
                          pdf_c+mu_c_expand*cdf_c), axis=0)
        var_delta = np.sum(w_c_expand**2*(mu_c_expand**2*cdf_c*(1-cdf_c)+sigma_c_expand**2 /
                           k_c**2*(cdf_c-pdf_c**2)+mu_c_expand*sigma_c_expand/k_c*pdf_c*(1-2*cdf_c)), axis=0)

        prob = norm.cdf((mu_delta-delta)/np.sqrt(var_delta))
        quant = mu_delta - norm.ppf(1-gamma)*np.sqrt(var_delta)

    else:
        C = len(mu_c)
        N = len(ntr_set)

        mu_c_expand = mu_c[:, np.newaxis, np.newaxis]
        sigma_c_expand = sigma_c[:, np.newaxis, np.newaxis]
        w_c_expand = w_c[:, np.newaxis, np.newaxis]
        s_c_expand = s_c[:, np.newaxis, np.newaxis]
        ntr_expand = ntr_set[np.newaxis, :, np.newaxis]

        mu_c_post_var = sigma_c_expand**4 / \
            (sigma_c_expand**2+4*s_c_expand**2/w_c_expand/(ntr_expand+1e-30))
        mu_c_post_sample = np.random.normal(
            mu_c_expand, np.sqrt(mu_c_post_var), size=(C, N, B))

        v_post = np.sum(w_c_expand*mu_c_post_sample *
                        (mu_c_post_sample >= 0), axis=0)
        prob = np.mean(v_post >= delta, axis=1)
        quant = np.quantile(v_post, gamma, axis=1)

    return prob, quant

def solve_g_prob_d_expected_improvement(delta, gamma, mu_c, sigma_c, s_c, w_c, n_max, parallel=1000000, B=0):
    # B>0: Algorithm 4 (simulation); requires Parallel & B to avoid memory overflow
    # B=0: Algorithm 5 (analytical approximation); Parallel & B are irrelevants
    
    ntr_full_set = np.arange(0, n_max+1, 1)
    ntr_opt = -1
    for ntr_group in chunker(ntr_full_set, parallel):
        _,quant = stats_g_prob_d_expected_improvement(delta, gamma, mu_c, sigma_c, s_c, w_c, ntr_group, B)
        cond_check = quant >= delta
        
        if np.sum(cond_check) > 0:
            ntr_opt = ntr_group[np.where(cond_check)[0][0]]
            break
    return ntr_opt


def get_prob_ab_cert_sim(n_tr, n_ce, alpha, mu_c, sigma_c, s_c, w_c, B=10000):
    C = len(mu_c)

    mu_c_post_var = sigma_c**4/(sigma_c**2+4*s_c**2/w_c/(n_tr+1e-30))
    mu_c_post_var = mu_c_post_var[:, np.newaxis]
    mu_c_post_mean = mu_c[:, np.newaxis]

    mu_c_post_sample = np.random.normal(
        mu_c_post_mean, np.sqrt(mu_c_post_var), size=(C, B))

    v_post = np.sum(w_c[:, np.newaxis]*mu_c_post_sample *
                    (mu_c_post_sample >= 0), axis=0)
    var_c_post = 1/(1/sigma_c**2+w_c*n_tr/4/s_c**2)

    w_U = w_c**2*(var_c_post+4*s_c**2/w_c/n_ce)

    U = np.sum(w_U[:, np.newaxis]*(mu_c_post_sample >= 0), axis=0)
    Y = v_post/np.sqrt(U+1e-10)
    prob = np.mean(norm.cdf(Y-norm.ppf(1-alpha)))

    return prob


def get_prob_ab_cert_aprx(n_tr, n_ce, alpha, mu_c, sigma_c, s_c, w_c, taylor=0):
    # Estimate LHS Equation 41
    k_c = np.sqrt(1+4*s_c**2/w_c/(n_tr+1e-30)/(sigma_c**2))
    pdf_c = norm.pdf(mu_c*k_c/sigma_c)
    cdf_c = norm.cdf(mu_c*k_c/sigma_c)
    var_c_post = 1/(1/sigma_c**2+w_c*n_tr/4/s_c**2)

    mu_D = np.sum(w_c*(sigma_c/k_c*pdf_c+mu_c*cdf_c))
    p_c = cdf_c
    mu_U = np.sum(w_c**2*(var_c_post+4*s_c**2/w_c/n_ce)*p_c)
    
    prob = norm.cdf((mu_D/np.sqrt(mu_U)-norm.ppf(1-alpha)))
    
    if taylor:
        # Higher-order Taylor series (Appendix D2) 
        var_D = np.sum(w_c**2*(mu_c**2*cdf_c*(1-cdf_c)+sigma_c**2 /
                   k_c**2*(cdf_c-pdf_c**2)+mu_c*sigma_c/k_c*pdf_c*(1-2*cdf_c)))
    
        var_U = np.sum(p_c*(1-p_c)*w_c**4*(var_c_post+4*s_c**2/w_c/n_ce)**2)

        var_DU = var_D/mu_D**2+var_U/(4*mu_U**2)-1/mu_D/mu_U*np.sum(
            w_c**3*(1-p_c)*(sigma_c/k_c*pdf_c+mu_c*cdf_c)*(var_c_post+4*s_c**2/w_c/n_ce))
        
        prob = norm.cdf((mu_D-norm.ppf(1-alpha)*np.sqrt(mu_U)) /
                        np.sqrt(mu_U+mu_D**2*var_DU))

    return prob


def get_prob_ab_cert_aprx_array(ntr, nce_set, alpha, mu_c, sigma_c, s_c, w_c, taylor=0):
    w_c = w_c[:, np.newaxis]
    mu_c = mu_c[:, np.newaxis]
    sigma_c = sigma_c[:, np.newaxis]
    nce_set = nce_set[np.newaxis, :]+1e-30
    s_c = s_c[:, np.newaxis]
    ntr = ntr+1e-30

    k_c = np.sqrt(1+4*s_c**2/w_c/ntr/(sigma_c**2))
    pdf_c = norm.pdf(mu_c*k_c/sigma_c)
    cdf_c = norm.cdf(mu_c*k_c/sigma_c)
    var_c_post = 1/(1/sigma_c**2+w_c*ntr/4/s_c**2)

    mu_D = np.sum(w_c*(sigma_c/k_c*pdf_c+mu_c*cdf_c), axis=0)
    p_c = cdf_c
    mu_U = np.sum(w_c**2*(var_c_post+4*s_c**2/w_c/nce_set)*p_c, axis=0)
    
    prob = norm.cdf((mu_D/np.sqrt(mu_U)-norm.ppf(1-alpha)))
    
    if taylor:
        var_D = np.sum(w_c**2*(mu_c**2*cdf_c*(1-cdf_c)+sigma_c**2/k_c **
                   2*(cdf_c-pdf_c**2)+mu_c*sigma_c/k_c*pdf_c*(1-2*cdf_c)), axis=0)

        var_U = np.sum(p_c*(1-p_c)*w_c**4*(var_c_post +
                       4*s_c**2/w_c/nce_set)**2, axis=0)

        var_DU = var_D/mu_D**2+var_U/(4*mu_U**2)-1/mu_D/mu_U*np.sum(w_c**3*(1-p_c)*(
            sigma_c/k_c*pdf_c+mu_c*cdf_c)*(var_c_post+4*s_c**2/w_c/nce_set), axis=0)

        prob = norm.cdf((mu_D-norm.ppf(1-alpha)*np.sqrt(mu_U)) /
                        np.sqrt(mu_U+mu_D**2*var_DU))

    return prob


def solve_nce_aprx(ntr_set, alpha, beta,mu_c, sigma_c, s_c, w_c):
    Z_ab = norm.ppf(1-alpha)+norm.ppf(1-beta) 

    mu_c_expand = mu_c[:, np.newaxis]
    sigma_c_expand = sigma_c[:, np.newaxis]
    s_c_expand = s_c[:, np.newaxis]
    w_c_expand = w_c[:, np.newaxis]
    ntr_expand = ntr_set[np.newaxis, :]

    sigma_c_p = np.sqrt(
        1/(1/sigma_c_expand**2+w_c_expand*ntr_expand/4/s_c_expand**2))

    k_c = np.sqrt(1+4*s_c_expand**2/w_c_expand /
                  (ntr_expand+1e-30)/(sigma_c_expand**2))
    pdf_c = norm.pdf(mu_c_expand*k_c/sigma_c_expand)
    cdf_c = norm.cdf(mu_c_expand*k_c/sigma_c_expand)
    mu_delta = np.sum(w_c_expand*(sigma_c_expand/k_c *
                      pdf_c+mu_c_expand*cdf_c), axis=0)
    p_c = cdf_c

    nce = 4*Z_ab**2*np.sum(p_c*w_c_expand*s_c_expand**2, axis=0) / \
        (mu_delta**2-Z_ab**2*np.sum(w_c_expand**2*sigma_c_p**2*p_c, axis=0))

    return np.ceil(nce)


def solve_nce_sim_inner(ntr, nce_min, nce_max, alpha, beta, mu_c, sigma_c, s_c, w_c, B):
    if (nce_max // 1)==(nce_min // 1):
        return nce_min
    
    nce_mid = (nce_min+nce_max)/2

    mid_val = get_prob_ab_cert_sim(
        ntr, nce_mid, alpha, mu_c, sigma_c, s_c, w_c, B)-(1-beta)

    if mid_val <= 0:
        return solve_nce_sim_inner(ntr, nce_mid, nce_max, alpha, beta, mu_c, sigma_c, s_c, w_c, B)
    else:
        return solve_nce_sim_inner(ntr, nce_min, nce_mid, alpha, beta, mu_c, sigma_c, s_c, w_c, B)


def solve_nce_sim(ntr, nce_max, alpha, beta, mu_c, sigma_c, s_c, w_c, B):
    nce_opt = -1
    max_prob = get_prob_ab_cert_sim(
        ntr, nce_max, alpha, mu_c, sigma_c, s_c, w_c, B)
    if (max_prob > 1-beta):
        nce_opt = solve_nce_sim_inner(
            ntr, 0, nce_max, alpha, beta, mu_c, sigma_c, s_c, w_c, B)
    return math.ceil(nce_opt)


def solve_nce_taylor(ntr, nce_max, alpha, beta, mu_c, sigma_c, s_c, w_c):
    nce_full_set = np.arange(nce_max+1)
    nce_opt = -1
    for nce_group in chunker(nce_full_set, 150000):
        prob_group = get_prob_ab_cert_aprx_array(
            ntr, nce_group, alpha, mu_c, sigma_c, s_c, w_c, taylor=1)
        if np.sum(prob_group >= (1-beta)) > 0:
            nce_opt = nce_group[np.where(prob_group >= (1-beta))[0][0]]
            break
    return nce_opt


def solve_ab_certification(alpha, beta, mu_c, sigma_c, s_c, w_c, n_max, B=0, taylor=0):
    # Key solver for (a,b)-certification
    # B > 0: simulated solution
    # B==0: analytical approximation; taylor = 1 uses a more-precise but slower approximation (Appendix D2)
    
    ntr_full_set = np.arange(n_max+1)
    ntr_opt = -1
    nce_opt = -1
    ntotal_opt = n_max+1

    if (B>0):
        # Algorithm 2
        for ntr in tqdm.tqdm(ntr_full_set):
            if (ntr >= ntotal_opt): continue
            nce = solve_nce_sim(ntr, ntotal_opt-ntr, alpha, beta, mu_c, sigma_c, s_c, w_c, B)
            if (nce > 0):
                ntr_opt = ntr
                nce_opt = nce
                ntotal_opt = ntr+nce
                
    if (B==0)&(taylor==0):
        # Algorithm 3
        for ntr_group in chunker(ntr_full_set, 100000):
            nce_group = solve_nce_aprx(
                ntr_group, alpha, beta, mu_c, sigma_c, s_c, w_c)
            nce_group[nce_group < 0] = n_max+1
            ntotal_group = ntr_group+nce_group
            
            ntotal_group_min = min(ntotal_group)
            
            if (ntotal_group_min < ntotal_opt):
                idx = np.where(ntotal_group == ntotal_group_min)[0][0]
                ntr_opt = ntr_group[idx]
                nce_opt = nce_group[idx]
                ntotal_opt = ntotal_group_min

            if (ntr_group[-1] >= ntotal_opt):
                break

    if (B==0)&(taylor==1):
        # Using higher-order Taylor series (Appendix D2)
        for ntr in tqdm.tqdm(ntr_full_set):
            if (ntr >= ntotal_opt): continue
            nce = solve_nce_taylor(ntr, ntotal_opt-ntr, alpha, beta, mu_c, sigma_c, s_c, w_c)
            if (nce > 0):
                ntr_opt = ntr
                nce_opt = nce
                ntotal_opt = ntr+nce

    return ntr_opt, int(nce_opt)
