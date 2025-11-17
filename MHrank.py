# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)] 
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to player i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))

    w = np.zeros(num_players)  # skill for each player
    accepted = 0  # count accepted moves
    for itr in tqdm(range(num_its)):
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob 
            lp1 = norm.logpdf(w[i]) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))

            # proposed new skill and log-prob
            # TODO

            # accept or reject move:
            # TODO

            # 1. Propose a new skill w_prop from a Gaussian centered at the current skill w[i]
            # A standard deviation of 0.1 is a reasonable starting point for tuning.
            proposal_std = 0.1 
            w_prop = w[i] + np.random.normal(0, proposal_std)
            
            # 2. Calculate the log-probability of the proposed skill
            lp2 = norm.logpdf(w_prop) + np.sum(norm.logcdf(outcome*(w_prop-w[j])))

            # accept or reject move:
            # TODO
            
            # 3. Calculate the acceptance ratio in log-space
            log_acceptance_ratio = lp2 - lp1
            
            # 4. Accept the move if the new state is more probable,
            #    or with a probability corresponding to the acceptance ratio
            if np.log(np.random.rand()) < log_acceptance_ratio:
                accepted += 1
                w[i] = w_prop
            # (Else, do nothing and keep w[i] as it is, i.e., reject the move)

        skill_samples[:, itr] = w

    return skill_samples, accepted
