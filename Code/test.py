# Import everything from utils.py
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



if __name__ == "__main__":
    # Read data from simulated_priors csv
    # input priors_csv is the function defined in utils
    w_c, mu_c, sigma_c, s_c = input_priors_csv("simulated_priors.csv")

    # Parameters set up (Managerial Requirements)
    DELTA = 1.5 # Expected Performance Requirement (see Section 3)
    GAMMA = 0.3 # Probability Requirement (see Section 3.5)
    N_MAX = 100000 # Maximum experiment size (Algorithms 2-5)
    ALPHA = 0.05 # (1-ALPHA) indicates confidence in a statistical test (see Section 4)
    BETA = 0.3 # (1-BETA) indicates power in a statistical test (see Section 4)
    
    # Sample size calculation for the d-expected improvement
    # Algorithm 1
    # assert one case with DELTA as 1.5 defined before
    assert (solve_d_expected_improvement(DELTA,mu_c,sigma_c,s_c,w_c) == 4068)
    print()

    # assert sample size calculation is the almost same as the ntr_set
    ntr_set = np.linspace(0,N_MAX,100,endpoint=True)
    perf_mean = stats_g_prob_d_expected_improvement(0,0.5,mu_c,sigma_c,s_c,w_c,ntr_set,0)[1]
    sample_d_exp = [solve_d_expected_improvement(delta,mu_c,sigma_c,s_c,w_c) for delta in perf_mean]
    
    # Include the plot on the website 
    plt.plot(ntr_set,perf_mean,c='black',label=f"Control")
    plt.plot(sample_d_exp, perf_mean, c='green', label=f'Expectation')
    plt.xlabel("Size of Training Data")
    plt.title(r"Equation 13: Mean ( $V_{\pi_t - \pi_0}^{post}$ )")
    plt.ylabel("Expected Profit Improvement")
    plt.legend()
    #plt.savefig("test_algorithm1.png")
    plt.show()

    