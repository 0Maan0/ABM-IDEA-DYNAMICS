from datetime import datetime
from src.run_model_utils import *
from src.run_sensitivity_tools import run_full_sensitivity_analysis

if __name__ == "__main__":
    # Model parameters
    num_agents = 10  # Number of scientists in the network
    network_type = "cycle"  # Options: 'cycle', 'wheel', 'complete'
    
    # Theory payoff parameters
    old_theory_payoff = 0.5  # Payoff for working with the old theory
    new_theory_payoffs = (0.4, 0.6)  # Payoffs for working with the new theory when either (old theory true, new theory true)
    true_theory = "new"  # Which theory is actually true (old or new)
    belief_strength_range = (0.5, 2.0)  # Belief strength range scientists have which will affect their resistance to change their beliefs
    
    # Number of simulations to run
    num_simulations = 2000
    show_final_state = False  # True if you want to see the final state of the simulation (to check convergence)
    
    # Parameters for the animation 
    # Runs until convergence but here you can say if you want to use animation or not
    use_animation = False
    max_steps = 1000   
    animation_params = {
        'num_frames': 30,
        'interval': 500,
        'steps_per_frame': 1
    }
    
    # Choose what analysis to run
    run_regular_simulations = True  
    run_sensitivity = True
    
    if run_regular_simulations:
        print("\n=== Running Regular Simulations ===")
        # Run the simulations with the above chosen parameters
        run_simulations_until_convergence(
            num_simulations=num_simulations,
            num_agents=num_agents,
            network_type=network_type,
            old_theory_payoff=old_theory_payoff,
            new_theory_payoffs=new_theory_payoffs,
            true_theory=true_theory,
            belief_strength_range=belief_strength_range,
            use_animation=use_animation,
            max_steps=max_steps,
            animation_params=animation_params,
            show_final_state=show_final_state
        )
    
    if run_sensitivity:
        print("\n=== Running Sensitivity Analysis ===")
        # Run sensitivity analysis
        run_full_sensitivity_analysis(
            num_trajectories=10,  # Number of trajectories for Morris analysis
            run_single=True,      # Run analysis for each network type
            run_comparison=True   # Run comparison across network types
        )
