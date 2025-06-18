from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
from src.scientists import ScientistAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

import seaborn as sns
import numpy as np

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# titlesize = 16
# labelsize = 20
colors = sns.color_palette("Set2", 8)
# ticksize = 18

class ScienceNetworkModel(Model):
    """
    The main model for simulating the spread of a new scientific theory in a network.  
    For now you can choose between a cycle, wheel and complete network.
    """
    def __init__(self, num_agents=10, network_type="cycle", true_probs=(0.9, 0.1)):
        self.num_agents = num_agents
        self.true_probs = true_probs  
        self.schedule = SimultaneousActivation(self)
        self.network = self._create_network(network_type)

        # Initiate agents such that only the first agent is very dedicated to a new theory and the rest isnt yet
        original_agent = np.random.randint(0, num_agents)
        for i in range(num_agents):
            if i == original_agent:
                prior_new = 0.9 # so this agent believes that theory A is very likely to be true 
                prior_old = 0.1
            else:
                prior_new = 0.4 # the other agents believe in the old theory mostly but also 0.4 open to the new one
                prior_old = 0.6
            agent = ScientistAgent(i, self, prior_new, prior_old)
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
        self.schedule.step()

    def animate(self, num_frames=200, interval=500, steps_per_frame=1):
        fig, ax = plt.subplots(figsize=(6, 6))
        G = self.network
        pos = nx.spring_layout(G, seed=69)  
        def get_colors():
            return [colors[1] if agent.current_choice == 0 else colors[0] for agent in self.schedule.agents]
        
        # Define legend patches
        legend_elements = [
            Patch(facecolor=colors[1], edgecolor='k', label='Believes New Theory'),
            Patch(facecolor=colors[0], edgecolor='k', label='Believes Old Theory')
        ]
        
        def update(frame):
            for _ in range(steps_per_frame):
                self.step()
            ax.clear()
            nx.draw(G, pos, with_labels=True, node_color=get_colors(), node_size=500, ax=ax)
            ax.set_title(f"Step {frame * steps_per_frame + 1}")
            ax.legend(handles=legend_elements, loc='upper right')
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=False)
        plt.show()
