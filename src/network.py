from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
from src.scientists import ScientistAgent
import numpy as np

class ScienceNetworkModel(Model):
    """
    The main model for simulating the spread of a new scientific theory in a network.  
    For now you can choose between a cycle, wheel and complete network.
    """
    def __init__(self, num_agents=10, network_type="cycle", true_probs=(0.2, 0.8)):
        self.num_agents = num_agents
        self.true_probs = true_probs  
        self.schedule = SimultaneousActivation(self)
        self.network = self._create_network(network_type)
        self.step_count = 0
        self.converged = False

        #TODO Maan: write multiple functions for different ways to start the simulation
        
        # Initiate agents such that only the first agent is very dedicated to a new theory and the rest isnt yet
        original_agent = np.random.randint(0, num_agents)
        for i in range(num_agents):
            if i == original_agent: #TODO: add original agent priors 
                prior_old = 0.2
                prior_new = 0.8 # so this agent believes that the new theory is very likely to be true 
            else: #TODO: add distribution of priors
                prior_old = 0.7
                prior_new = 0.3 # the other agents believe in the old theory mostly but also 0.4 open to the new one
            agent = ScientistAgent(i, self, prior_old, prior_new)
            self.schedule.add(agent)

    def _create_network(self, network_type):
        if network_type == "cycle":
            return nx.cycle_graph(self.num_agents)
        elif network_type == "wheel":
            return nx.wheel_graph(self.num_agents)
        elif network_type == "complete":
            return nx.complete_graph(self.num_agents)
        else:
            raise ValueError("Unknown network type")

    def step(self):
        self.step_count += 1
        self.schedule.step()
        
        # Check if the simulation has converged after each step
        if not self.converged and self.convergence_status():
            self.converged = True
            self.convergence_step = self.step_count
    
    def convergence_status(self):
        beliefs = [agent.current_choice for agent in self.schedule.agents]
        return len(set(beliefs)) == 1 # Returns True if all agents have the same belief

    def get_convergence_info(self):
        if self.converged:
            beliefs = [agent.current_choice for agent in self.schedule.agents]
            theory = "New Theory" if beliefs[0] == 1 else "Old Theory"
            return {
                'converged': True,
                'step': self.convergence_step,
                'theory': theory
            }
        else:
            return {
                'converged': False,
                'step': self.step_count,
                'theory': None
            }
