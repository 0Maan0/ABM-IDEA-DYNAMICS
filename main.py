from datetime import datetime
import random
from src.run_model_utils import *
from src.run_sensitivity_tools import run_full_sensitivity_analysis
from src.plot_sensitivity_results import (
    plot_single_analysis,
    plot_network_comparison,
    plot_all_metrics,
    plot_all_comparisons
)
from src.scientists import ScientistAgent
from src.influential_scientists import SuperScientistAgent

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
    num_simulations = 1
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
    
    # Choose what code to run
    run_regular_simulations = True  
    run_sensitivity = False #True
    create_plots = True
    
    # Sensitivity analysis parameters
    num_trajectories = 4 # Will generate about 715 * 7 = 5005 parameter combinations (we should do 5k)
    
    if run_regular_simulations:
        print("\n=== Running Regular Simulations ===")
        # Run the simulations with the above chosen parameters
        run_simulations_until_convergence(
            agent_class=SuperScientistAgent,  # Choice: ScientistAgent or SuperScientistAgent
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
            num_trajectories=num_trajectories,
            run_single=True,      # Run analysis for each network type
            run_comparison=True   # Run comparison across network types
        )
        
    if create_plots:
        print("\n=== Creating Sensitivity Analysis Plots ===")
        # Create plots for each network type
        for net_type in ['cycle', 'wheel', 'complete']:
            plot_all_metrics(network_type=net_type, num_trajectories=num_trajectories)
            
        # Create comparison plots across network types
        plot_all_comparisons(num_trajectories=num_trajectories)
        
        print("All plots have been saved to the analysis_plots directory!")

    
