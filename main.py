from datetime import datetime
from src.run_model_utils import *

if __name__ == "__main__":
    # Model parameters
    num_agents = 10  # Number of scientists in the network
    network_type = "complete"  # Options: 'cycle', 'wheel', 'complete'
    true_probs = (0.2, 0.8)  # Probabilities of the two theories being true (old, new)
    believe_strength_range = (0.5, 2.0)  # Range for belief strength of agents
    prior_strength_range = (1, 4)  # Range for prior strength of agents
    # Number of simulations to run
    num_simulations = 2000
    show_final_state = False  # Set to True if you want to see the final state of the simulation
    
    # Parameters for the animation 
    # Runs until convergence but here you can say if you want to use animation or not
    use_animation = False
    max_steps = 1000  
    
    # Animation parameters 
    animation_params = {
        'num_frames': 30,
        'interval': 500,
        'steps_per_frame': 1
    }
    
    # Run the simulations with the above chosen parameters
    run_simulations_until_convergence(
        num_simulations=num_simulations,
        num_agents=num_agents,
        network_type=network_type,
        true_probs=true_probs,
        use_animation=use_animation,
        max_steps=max_steps,
        animation_params=animation_params,
        show_final_state=show_final_state
    )
