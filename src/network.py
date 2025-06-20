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
    def __init__(self, num_agents=10, network_type="cycle", true_probs=(0.2, 0.8),
                 prior_strength_range=(1, 4), belief_strength_range=(0.5, 2.0)):
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
            # Initialize beta distribution parameters
            if i == original_agent:
                # Strong prior favoring new theory
                prior_old_alpha = random.uniform(1, 2)
                prior_old_beta = random.uniform(2, 4)
                prior_new_alpha = random.uniform(2, 4)
                prior_new_beta = random.uniform(1, 2)
            else:
                # Default to slightly favoring old theory
                prior_old_alpha = random.uniform(2, 4)
                prior_old_beta = random.uniform(1, 2)
                prior_new_alpha = random.uniform(1, 2)
                prior_new_beta = random.uniform(2, 4)
            
            # Scale priors by prior strength to control initial belief extremity
            prior_strength = random.uniform(*prior_strength_range)
            prior_old_alpha *= prior_strength
            prior_old_beta *= prior_strength
            prior_new_alpha *= prior_strength
            prior_new_beta *= prior_strength
            
            # Assign random belief strength to control resistance to change
            belief_strength = random.uniform(*belief_strength_range)
            
            agent = ScientistAgent(i, self, 
                                 prior_old_alpha=prior_old_alpha,
                                 prior_old_beta=prior_old_beta,
                                 prior_new_alpha=prior_new_alpha,
                                 prior_new_beta=prior_new_beta,
                                 belief_strength=belief_strength)
            self.schedule.add(agent)

    def _create_network(self, network_type):
        """Create a network based on the specified type."""
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
