from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
from scipy.stats import beta
import numpy as np

# https://link.springer.com/article/10.1007/s10670-009-9194-6

class ScientistAgent(Agent):
    """
    An agent class representing a scientist in a network epistemology model.
    """
    def __init__(self, unique_id, model, prior_old_alpha=2, prior_old_beta=2, 
                 prior_new_alpha=2, prior_new_beta=2, belief_strength=1.0):
        super().__init__(unique_id, model)
        
        # Beta distribution parameters for beliefs about each theory
        self.old_alpha = prior_old_alpha
        self.old_beta = prior_old_beta
        self.new_alpha = prior_new_alpha
        self.new_beta = prior_new_beta
        
        # Belief strength parameter (higher = more resistant to change)
        self.belief_strength = belief_strength
        
        # Calculate initial beliefs from beta distributions
        self.belief_old = self.old_alpha / (self.old_alpha + self.old_beta)
        self.belief_new = self.new_alpha / (self.new_alpha + self.new_beta)
        
        # The current theory the scientists believe
        self.current_choice = 0 if self.belief_old > self.belief_new else 1
        
        # Track experimental results
        self.experiments_old = []
        self.experiments_new = []

    def update_beta_params(self, success, theory):
        if theory == 0:  # Old theory
            self.old_alpha += success
            self.old_beta += (1 - success)
            self.belief_old = self.old_alpha / (self.old_alpha + self.old_beta)
        else:  # New theory
            self.new_alpha += success
            self.new_beta += (1 - success)
            self.belief_new = self.new_alpha / (self.new_alpha + self.new_beta)

    def incorporate_neighbor_evidence(self, neighbor, weight=0.5):
        # Weight neighbor influence by belief strength
        effective_weight = weight / (1 + self.belief_strength)
        
        # Update beliefs based on neighbor's evidence
        if neighbor.current_choice == 0:
            # Transfer some of neighbor's experimental results
            if neighbor.experiments_old:
                sample_size = min(len(neighbor.experiments_old), 
                                int(len(neighbor.experiments_old) * effective_weight))
                sampled_results = random.sample(neighbor.experiments_old, sample_size)
                for result in sampled_results:
                    self.update_beta_params(result, 0)
        else:
            if neighbor.experiments_new:
                sample_size = min(len(neighbor.experiments_new), 
                                int(len(neighbor.experiments_new) * effective_weight))
                sampled_results = random.sample(neighbor.experiments_new, sample_size)
                for result in sampled_results:
                    self.update_beta_params(result, 1)

    def step(self):
        chosen = self.current_choice
        true_prob = self.model.true_probs[chosen] #"true" probability of the chosen theory
        result = 1 if random.random() < true_prob else 0 #simulate evidence gathering
        
        # Store experimental result
        if chosen == 0:
            self.experiments_old.append(result)
        else:
            self.experiments_new.append(result)
        
        # Update own beliefs using beta distribution
        self.update_beta_params(result, chosen)
        
        # Talk to neighbours
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            # Simple averaging of beliefs ==> TODO: Make more complicated?
            # TODO: for some neighbours the power of their believe is more influencial
            #TODO: If scientists from the same "group" they are more likely to influence eachother
            self.incorporate_neighbor_evidence(neighbor)
        
        # Decide which theory to choose next step
        self.current_choice = 0 if self.belief_old > self.belief_new else 1
