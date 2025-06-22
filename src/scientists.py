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
    def __init__(self, unique_id, model, initial_belief=0.5, belief_strength=1.0):
        super().__init__(unique_id, model)
        
        self.belief_in_new_theory = initial_belief  # Belief that the new theory is true
        self.belief_strength = belief_strength # Belief strength parameter ==> if this value is higher, the scientist is more resistant to change
    
        self.current_choice = self._choose_theory() # The current theory choice ("old" or "new")
        
        # Experimental results when working with old or new theory
        self.old_theory_results = [] 
        self.new_theory_results = []  

    def _choose_theory(self):
        """To choose between the theories ==> which theory to believe/work
            ==> its based on expected utility (remember game theory :) )"""
        # Always the same payoff for the old theory
        old_theory_utility = self.model.old_theory_payoff
        
        # New theory utility looks at which theory is true 
        new_theory_utility = (
            self.belief_in_new_theory * self.model.new_theory_payoffs[1] +  # Payoff if new theory is true
            (1 - self.belief_in_new_theory) * self.model.new_theory_payoffs[0]  # Payoff if new theory is false
        )
        
        return "new" if new_theory_utility > old_theory_utility else "old"

    def update_belief(self, success, theory_choice):
        """
        Paper mentioned bayesian update based on experimental result: so try this:
        P(New Theory|Result) = P(Result|New Theory) * P(New Theory) / P(Result)
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
        
        # Bayes update
        numerator = p_result_if_new * prior
        denominator = p_result_if_new * prior + p_result_if_old * (1 - prior)
        
        # Update with resistance factor
        old_belief = self.belief_in_new_theory
        bayes_update = numerator / denominator
        self.belief_in_new_theory = old_belief + (bayes_update - old_belief) / self.belief_strength

    def incorporate_neighbor_evidence(self, neighbor):
        """Learn from neighbor's experimental results"""
        # Update based on both old and new theory results
        for result in neighbor.old_theory_results:
            self.update_belief(result, "old")
        for result in neighbor.new_theory_results:
            self.update_belief(result, "new")
            
    def step(self):
        # Get binary success/failure outcome
        true_mean = self.model.get_action_payoff(self.current_choice)
        success = random.random() < true_mean
        
        if self.current_choice == "old":
            self.old_theory_results.append(success)
        else:
            self.new_theory_results.append(success)
        
        self.update_belief(success, self.current_choice)
        
        # Learn from neighbors experimental results aka evidence
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            self.incorporate_neighbor_evidence(neighbor)
        
        # Choose theory for next step
        self.current_choice = self._choose_theory()
