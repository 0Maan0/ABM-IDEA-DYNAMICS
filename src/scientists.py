from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random

class ScientistAgent(Agent):
    """
    An agent class representing a scientist in a network epistemology model.
    """
    def __init__(self, unique_id, model, prior_new, prior_old):
        super().__init__(unique_id, model)

        # Beliefs about the two scientific theories new or old
        self.belief_new = prior_new
        self.belief_old = prior_old

        # The current theory the scientists believe
        self.current_choice = 0 if prior_new > prior_old else 1

    def step(self):
        chosen = self.current_choice
        true_prob = self.model.true_probs[chosen]
        result = 1 if random.random() < true_prob else 0

        # Bayesian for now
        if chosen == 0:
            self.belief_new = (self.belief_new + result) / 2
        else:
            self.belief_old = (self.belief_old + result) / 2

        # Talk to neighbours
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            # Simple averaging of beliefs ==> TODO: Make more complicated?
            self.belief_new = (self.belief_new + neighbor.belief_new) / 2
            self.belief_old = (self.belief_old + neighbor.belief_old) / 2

        # Decide which theory to choose next step
        self.current_choice = 0 if self.belief_new > self.belief_old else 1
