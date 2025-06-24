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

class ScientistAgent(Agent):
    """
    An agent representing a scientist in Zollman's (2007) model 
    (https://www.researchgate.net/publication/228973111_The_Communication_Structure_of_Epistemic_Communities).
    So every scientist has a belief about which theory (old or new) is true. Each step they choose which theory 
    they want to believe/work with based on expected utility. Then their beieves are updated,
    based on experimental results.
    """
    def __init__(self, unique_id, model, initial_belief=0.9, belief_strength=1.0):
        super().__init__(unique_id, model)
        
        self.belief_in_new_theory = initial_belief  # Start with optimistic belief in new theory
        self.belief_strength = belief_strength # Belief strength parameter ==> if this value is higher, the scientist is more resistant to change
    
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

        # Generate noise (Gaussian distribution) if noise is active in model. Center is 0
        if self.model.noise_active == "on":
            noise = np.random.normal(
                loc=self.model.noise_loc, scale=self.model.noise_std)
        else:
            noise = 0.0
        
        # For new theory, calculate expected probability of success
        p_new_better = self.belief_in_new_theory + noise
        p_success_if_better = self.model.new_theory_payoffs[1]  # 0.4
        p_success_if_worse = self.model.new_theory_payoffs[0]   # 0.6
        
        new_theory_prob = (
            p_success_if_better * p_new_better +  # P(success|New better)*P(New better)
            p_success_if_worse * (1 - p_new_better)  # P(success|New worse)*P(New worse)
        )
        
        return "new" if new_theory_prob > old_theory_prob else "old"

    def update_belief(self, success, theory_choice):
        """
        Bayesian update based on experimental result
        """
        # Get probabilities for each theory
        p_success_if_new = self.model.new_theory_payoffs[1]
        p_success_if_old = self.model.new_theory_payoffs[0]
        
        # Current belief
        prior = self.belief_in_new_theory
        
        # Calculate likelihood based on success/failure
        if success:
            p_result_if_new = p_success_if_new
            p_result_if_old = p_success_if_old
        else:
            p_result_if_new = 1 - p_success_if_new
            p_result_if_old = 1 - p_success_if_old
        
        # Direct Bayes update as in Zollman's paper
        numerator = p_result_if_new * prior
        denominator = p_result_if_new * prior + p_result_if_old * (1 - prior)
        
        # Update belief directly
        self.belief_in_new_theory = numerator / denominator

    def incorporate_neighbor_evidence(self, neighbor):
        """Learn from neighbor's current round experimental results only"""
        # Update based on neighbor's current round results only
        if neighbor.current_old_theory_result is not None:
            self.update_belief(neighbor.current_old_theory_result, "old")
        if neighbor.current_new_theory_result is not None:
            self.update_belief(neighbor.current_new_theory_result, "new")
            
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
        self.update_belief(success, self.current_choice)
        
        # Learn from neighbors' current round results only
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            self.incorporate_neighbor_evidence(neighbor)
        
        # Choose theory for next step
        self.current_choice = self._choose_theory()
