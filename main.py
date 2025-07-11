"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the main code to run the
ABM-IDEA-DYNAMICS project. It includes functions to run regular simulations,
sensitivity analysis, and create visualizations of the results.
"""

from src.run_model_utils import *
from src.run_sensitivity_tools import run_full_sensitivity_analysis
from src.plot_sensitivity_results import (
    plot_single_analysis,
    plot_network_comparison,
    plot_all_metrics,
    plot_all_comparisons,
    generate_simple_summary
)
from src.super_scientist import SuperScientistAgent
from src.scientists import ScientistAgent
from src.single_run_analysis import analyze_and_plot_results



if __name__ == "__main__":
    # Model parameters
    num_agents = 10  # Number of scientists in the network
    network_type = "cycle"  # Options: 'cycle', 'wheel', 'complete'
    agent_class = ScientistAgent  # Choose the agent type (SuperScientistAgent or ScientistAgent)

    # Theory payoff parameters
    old_theory_payoff = 0.5  # Payoff for working with the old theory

    # Payoffs for working with the new theory when either (old theory true, new theory true)
    new_theory_payoffs = (0.4, 0.6)
    true_theory = "new"  # Which theory is actually true (old or new)

    # Belief strength range scientists have which will affect their resistance to change their
    # beliefs
    belief_strength_range = (0.5, 2.0)

    # Number of simulations to run
    num_simulations = 1

    # Parameters for the animation
    max_steps = 1000
    animation_params = {
        'num_frames': max_steps,
        'interval': 500,
        'steps_per_frame': 1
    }

    # Choose what code to run
    run_regular_simulations = False  # True if you want to run the regular simulations
    # if true choose if you want to make an animation of 1 normakl simulation:
    use_animation = False  # True if you want to create an animation of a single simulation
    run_sensitivity_analysis = False
    create_sensitivity_plots = True

    # Sensitivity analysis parameters
    num_trajectories = 715  # Will generate about 715 * 7 = 5005 parameter combinations (we should do 5k)

    if run_regular_simulations:
        print("\n=== Running Regular Simulations ===")
        # Run the simulations with the above chosen parameters
        run_simulations_until_convergence(
            agent_class=agent_class,
            num_simulations=num_simulations,
            num_agents=num_agents,
            network_type=network_type,
            old_theory_payoff=old_theory_payoff,
            new_theory_payoffs=new_theory_payoffs,
            true_theory=true_theory,
            belief_strength_range=belief_strength_range,
            use_animation=use_animation,
            max_steps=max_steps,
            animation_params=animation_params
        )

        if use_animation is False:
            # Analyze and plot the results
            print("\n=== Analyzing Simulation Results ===")
            analyze_and_plot_results(
                network_type=network_type,
                num_agents=num_agents,
                num_simulations=num_simulations,
                belief_range=belief_strength_range,
                agent_type=agent_class.__name__,
            )

    if run_sensitivity_analysis:
        print("\n=== Running Sensitivity Analysis ===")
        # Run sensitivity analysis
        run_full_sensitivity_analysis(
            num_trajectories=num_trajectories,
            run_single=True,      # Run analysis for each network type
            run_comparison=True   # Run comparison across network types
        )

    if create_sensitivity_plots:
        print("\n=== Creating Sensitivity Analysis Plots ===")
        # Create plots for each network type
        for net_type in ['cycle', 'wheel', 'complete', 'bipartite', 'cliques']:
            plot_all_metrics(network_type=net_type, num_trajectories=num_trajectories)

        # Create comparison plots across network types
        # plot_all_comparisons(num_trajectories=num_trajectories)
        summary = generate_simple_summary(num_trajectories=715)

        print("All plots have been saved to the analysis_plots directory!")
        print( summary)
