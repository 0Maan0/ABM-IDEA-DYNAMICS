from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm


def ellips_scaling(x, p):
    """
    This function uses ellips scaling to scale the influence of scientist using
    h-index as parameter. An h_index (p) value below 0.5 makes a scientist less
    believable, vice versa for h_index > 0.5.

    Args:
        x (float): The input value to be scaled.
        p (float): The h-index of the scientist, scaled to [0, 1] range.

    Returns:
        float: The scaled value based on the ellips function.

    source:
    https://www.reddit.com/r/desmos/comments/1ijfv6f/i_made_a_variety_of_01_scaling_functions/
    """
    # scale p to [-0.5, 0.5], so 0 now is actually the center point
    p -= 0.5
    scale_factor = 1  # increase the aggressiveness of the scaling
    a = 1 / (scale_factor * abs(p) + 1)

    # normal ellips (above midway point of 0)
    ellnorm = lambda x, a: (-((1 - x) ** a - 1)) ** (1 / a)
    # inverse ellips (below the midway point of 0)
    ellinv = lambda x, a: - (-x ** a + 1)**(1 / a) + 1

    return ellnorm(x, a) if p < 0 else ellinv(x, a)


class SuperScientistAgent(Agent):
    """
    An agent representing a scientist in Zollman's (2007) model
    (https://www.researchgate.net/publication/228973111_The_Communication_Structure_of_Epistemic_Communities).
    So every scientist has a belief about which theory (old or new) is true. Each step they choose which theory
    they want to believe/work with based on expected utility. Then their beieves are updated,
    based on experimental results.
    """
    def __init__(self, unique_id, model, initial_belief=0.9, belief_strength=1.0,h_index=1):
        super().__init__(unique_id, model)

        self.belief_in_new_theory = initial_belief  # Start with optimistic belief in new theory
        self.belief_strength = belief_strength # Belief strength parameter ==> if this value is higher, the scientist is more resistant to change
        self.h_index = h_index # H-index of the scientist, which can be used to measure their influence or reputation
        self.current_choice = self._choose_theory() # The current theory choice ("old" or "new")

        # Current round's experimental results
        self.current_old_theory_result = None
        self.current_new_theory_result = None

    def _choose_theory(self):
        """
        Choose between theories based on expected success probability.
        According to paper:
        - Old theory has fixed success probability of 0.5
        - For new theory, expected success probability is:
          P(success) = P(success|New better)*P(New better) + P(success|New worse)*P(New worse)
        """
        # Old theory has fixed success probability
        old_theory_prob = self.model.old_theory_payoff  # 0.5

        # For new theory, calculate expected probability of success
        p_new_better = self.belief_in_new_theory
        p_success_if_better = self.model.new_theory_payoffs[1]  # 0.4
        p_success_if_worse = self.model.new_theory_payoffs[0]   # 0.6

        new_theory_prob = (
            p_success_if_better * p_new_better +  # P(success|New better)*P(New better)
            p_success_if_worse * (1 - p_new_better)  # P(success|New worse)*P(New worse)
        )

        return "new" if new_theory_prob > old_theory_prob else "old"

    def update_belief(self, success, theory ,weight=1.0):
        """
        Bayesian update based on experimental result
        """
        # Get probabilities for each theory
        p_success_if_new = self.model.new_theory_payoffs[1]
        p_success_if_old = self.model.new_theory_payoffs[0]

        # Current belief
        prior = self.belief_in_new_theory

        if theory == "new":
            # If testing new theory:
            if success:
                p_result_if_new = p_success_if_new
                p_result_if_old = p_success_if_old
            else:
                p_result_if_new = 1 - p_success_if_new
                p_result_if_old = 1 - p_success_if_old
        else:  # theory == "old"
            # If testing old theory
            if success:
                p_result_if_new = p_success_if_old  # Probability of old theory success if new theory is worse
                p_result_if_old = p_success_if_new  # Probability of old theory success if new theory is better
            else:
                p_result_if_new = 1 - p_success_if_old
                p_result_if_old = 1 - p_success_if_new

        # Apply H-index weighting to the evidence
        # weight=1 means maximum influence, 0.5 means neutral, 0 means lowered
        # influence
        weighted_p_result_if_new = ellips_scaling(float(p_result_if_new), weight)
        weighted_p_result_if_old = ellips_scaling(float(p_result_if_old), weight)

        # Direct Bayes update as in Zollman's paper
        numerator = weighted_p_result_if_new * prior
        denominator = weighted_p_result_if_new * prior + weighted_p_result_if_old * (1 - prior)

        # Update belief directly
        if denominator > 0:
            self.belief_in_new_theory = numerator / denominator
            self.belief_in_old_theory = 1 - self.belief_in_new_theory

    def incorporate_neighbor_evidence(self, neighbor):
        """Learn from neighbor's current round experimental results only"""
        # Calculate influence weight based on H-index using model's chosen scaling method
        influence_weight = self.calculate_influence_weight(neighbor.h_index)

        # Update based on neighbor's current round results only
        if neighbor.current_old_theory_result is not None:
            self.update_belief(neighbor.current_old_theory_result, weight=influence_weight)
        if neighbor.current_new_theory_result is not None:
            self.update_belief(neighbor.current_new_theory_result, weight=influence_weight)

    def calculate_influence_weight(self, neighbor_h_index, method="probit"):
        """
        Calculate influence weight based on neighbor's H-index.
        Using a probit function to model the influence.
        """
        if method == "probit":
            # Probit function: maps H-index to a probability-like influence weight
            # These can be adjusted based on empirical studies of academic influence
            if hasattr(self.model, 'influence_params') and self.model.influence_params:
                params = self.model.influence_params
                mean_threshold = params.get('mean_threshold', 10.0)
                std_dev = params.get('std_dev', 5.0)
                noise_level = params.get('noise_level', 0.1)
                min_influence = params.get('min_influence', 0.05)
                max_influence = params.get('max_influence', 0.95)
            else:
                # Default parameters if model doesn't have them
                mean_threshold = 10.0
                std_dev = 5.0
                noise_level = 0.1
                min_influence = 0.05
                max_influence = 0.95

            # Add some randomness to the H-index observation (modeling uncertainty)
            observed_h_index = neighbor_h_index + np.random.normal(0, noise_level * neighbor_h_index)

            # Calculate probit probability
            z_score = (observed_h_index - mean_threshold) / std_dev
            base_probability = norm.cdf(z_score)

            # Ensure minimum and maximum influence levels
            min_influence = 0.05  # Even low H-index scientists have some credibility
            max_influence = 0.95  # Even high H-index scientists aren't always right

            scaled_probability = min_influence + (max_influence - min_influence) * base_probability

            return scaled_probability
        elif method == "linear":
            # Simple linear scaling based on H-index difference
            return 1 / (1 + np.exp(-0.1 * (neighbor_h_index - self.h_index)))
        else:
            raise ValueError("Unknown influence calculation method")

    def step(self):
        # Reset current round results
        self.current_old_theory_result = None
        self.current_new_theory_result = None

        # Get probability of success for current theory choice
        prob_success = self.model.get_action_payoff(self.current_choice)

        # Run experiment and get binary success/failure
        success = random.random() < prob_success

        # Store only current round's result
        if self.current_choice == "old":
            self.current_old_theory_result = success
        else:
            self.current_new_theory_result = success

        # Update based on own result
        self.update_belief(success, weight=1.0)

        # Learn from neighbors' current round results only
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            self.incorporate_neighbor_evidence(neighbor)

        # Choose theory for next step
        self.current_choice = self._choose_theory()
