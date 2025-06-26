from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
from src.scientists import ScientistAgent
from src.influential_scientists import SuperScientistAgent
import numpy as np


class ScienceNetworkModel(Model):
    """
    The main model for simulating the spread of a new scientific theory in a network.
    For now you can choose between a cycle, wheel and complete network.
    """
    def __init__(
        self,
        agent_class=ScientistAgent,  # Default to regular ScientistAgent
        num_agents=10,
        network_type="complete",
        old_theory_payoff=0.5,  # Payoff for believing the old theory
        new_theory_payoffs=(0.4, 0.6),  # Payoffs for believing new theory when (old theory true, new theory true)
        true_theory="new",
        belief_strength_range=(0.5, 2.0),
        noise="off",
        noise_loc=0.0,
        noise_std=0.5,
        h_index=0.0
    ):
        super().__init__()
        self.num_agents = num_agents
        self.network_type = network_type
        self.old_theory_payoff = old_theory_payoff
        self.new_theory_payoffs = new_theory_payoffs
        self.true_theory = true_theory
        self.belief_strength_range = belief_strength_range
        self.schedule = SimultaneousActivation(self)
        self.network = self._create_network(network_type)
        self.step_count = 0
        self.converged = False
        self.influence_scaling = "probit"
        self.agent_class = agent_class  # Allow for different agent types (e.g., ScientistAgent or SuperScientistAgent)
        self.noise_active = noise
        self.noise_loc = noise_loc
        self.noise_std = noise_std

        # Start scientists with random beliefs about which theory is true
        # TODO: make different initial conditions for this?

        # Create agents with initial beliefs and belief strength
        for i in range(num_agents):
            # Initial belief that new theory is true
            initial_belief = random.random()  # Uniform between 0 and 1

            # Make a random belief strength within the range to determine resistance to change
            belief_strength = random.uniform(*belief_strength_range)

            if self.agent_class == SuperScientistAgent:
                # Set h_index value - could be fixed or random
                h_index_value = random.uniform(0.0, 1.0)
                agent = self.agent_class(i, self,
                                        initial_belief=initial_belief,
                                        belief_strength=belief_strength,
                                        h_index=h_index_value)
            else:
                agent = self.agent_class(i, self,
                                        initial_belief=initial_belief,
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
        elif isinstance(network_type, nx.Graph):
            # Custom network
            if network_type.number_of_nodes() != self.num_agents:
                raise ValueError("Custom network must have the same number of nodes as num_agents")
            return network_type
        else:
            raise ValueError("Unknown network type or invalid custom network")

    def get_action_payoff(self, theory_choice):
        """Get the payoff for a given theory choice"""
        if theory_choice == "old":
            return self.old_theory_payoff

        # For new theory, payoff depends on which theory is actually true
        payoff_idx = 1 if self.true_theory == "new" else 0
        return self.new_theory_payoffs[payoff_idx]

    def step(self):
        self.step_count += 1
        self.schedule.step()

        # Check if the simulation has converged after each step
        if not self.converged and self.convergence_status():
            self.converged = True
            self.convergence_step = self.step_count

    def convergence_status(self):
        """
        According to Zollman's paper, a population has finished learning if one of two conditions are met:
        1. Every agent takes action A1 (old theory) ==> thus no new information can change their minds
        2. Every agent believes in phi2 (new theory) with probability > 0.9999
        """
        actions = [agent.current_choice for agent in self.schedule.agents]
        beliefs = [agent.belief_in_new_theory for agent in self.schedule.agents]

        # Condition 1: Everyone using old theory
        if all(action == "old" for action in actions):
            return True

        # Condition 2: Everyone strongly believes in new theory
        if all(b > 0.9999 for b in beliefs):
            return True

        return False

    def get_convergence_info(self):
        if self.converged:
            actions = [agent.current_choice for agent in self.schedule.agents]
            beliefs = [agent.belief_in_new_theory for agent in self.schedule.agents]

            # Check convergence type
            if all(action == "old" for action in actions):
                theory = "Old Theory"
            elif all(b > 0.9999 for b in beliefs):
                theory = "Correct Theory" if self.true_theory == "new" else "Incorrect Theory"

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
