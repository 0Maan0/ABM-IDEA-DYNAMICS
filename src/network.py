from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
from src.scientists import ScientistAgent
from src.influential_scientists import SuperScientistAgent
import numpy as np
import itertools
import math


class ScienceNetworkModel(Model):
    """
    The main model for simulating the spread of a new scientific theory in a network.
    For now you can choose between a cycle, wheel and complete network.
    """

    # Define network constants
    network_types = ["cycle", "wheel", "complete", "bipartite", "cliques", "custom"]

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
        elif network_type == "bipartite":
            return self.generate_complete_bipartite_graph(self.num_agents)
        elif network_type == "cliques":
            G, _ = self.generate_ring_of_cliques(self.num_agents)
            return G
        elif isinstance(network_type, nx.Graph):
            # Custom network
            if network_type.number_of_nodes() != self.num_agents:
                raise ValueError("Custom network must have the same number of nodes as num_agents")
            return network_type
        else:
            raise ValueError("Unknown network type or invalid custom network")

    @staticmethod
    def generate_complete_bipartite_graph(num_nodes):
        """
        Create a complete bipartite graph with `num_nodes` nodes split as evenly as possible
        between the two sets.
        """
        if num_nodes < 2:
            raise ValueError("At least 2 nodes are required to form a bipartite graph.")

        part1 = num_nodes // 2
        part2 = num_nodes - part1  # Handles both even and odd num_nodes

        G = nx.complete_bipartite_graph(part1, part2)
        return G

    @staticmethod
    def generate_ring_of_cliques(num_nodes):
        """
        Generate a ring of cliques with exactly `num_nodes` nodes.
        Special cases:
        - num_nodes == 2: single edge between two nodes.
        - num_nodes == 3: triangle (3-node clique).
        For num_nodes >= 4:
        - Number of cliques = floor(sqrt(num_nodes))
        - Nodes per clique = base size + (1 if extra node assigned)
        - Connect first node of each clique to next to form a ring
        """
        if num_nodes < 2:
            raise ValueError("At least 2 nodes are required.")

        if num_nodes == 2:
            # Just a single edge between node 0 and 1
            G = nx.Graph()
            G.add_nodes_from([0,1])
            G.add_edge(0, 1)
            clique_sizes = [2]
            return G, clique_sizes

        if num_nodes == 3:
            # Complete graph of 3 nodes (triangle)
            G = nx.complete_graph(3)
            clique_sizes = [3]
            return G, clique_sizes

        # For num_nodes >= 4, use ring of cliques logic
        num_cliques = math.isqrt(num_nodes)
        base_clique_size = num_nodes // num_cliques
        extras = num_nodes % num_cliques

        # Distribute extras to the first few cliques
        clique_sizes = [base_clique_size + (1 if i < extras else 0) for i in range(num_cliques)]

        G = nx.Graph()
        node_counter = 0
        clique_node_lists = []

        for size in clique_sizes:
            nodes = list(range(node_counter, node_counter + size))
            G.add_nodes_from(nodes)
            G.add_edges_from(itertools.combinations(nodes, 2))  # fully connect clique
            clique_node_lists.append(nodes)
            node_counter += size

        # Connect each clique to the next to form a ring (1 edge per pair)
        for i in range(num_cliques):
            a = clique_node_lists[i][0]
            b = clique_node_lists[(i + 1) % num_cliques][0]
            G.add_edge(a, b)

        return G, clique_sizes  # clique_sizes probably not necessary

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
