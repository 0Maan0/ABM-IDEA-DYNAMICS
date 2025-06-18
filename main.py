from src.network import ScienceNetworkModel
#from src.scientists import ScientistAgent

if __name__ == "__main__":
    num_agents = 10
    network_type = "cycle"  # Options: 'cycle', 'wheel', 'complete'
    true_probs = (0.6, 0.4)
    num_steps = 30

    model = ScienceNetworkModel(num_agents=num_agents, network_type=network_type, true_probs=true_probs)
    model.animate(num_frames=num_steps, interval=500, steps_per_frame=1)
