"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the utility functions used to run th model in
main.py. it contains functions to create and run the model, animate the model,
save results to CSV, and run multiple simulations in parallel. It also includes
functions to run the model until convergence and to create animations of the
model's evolution.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
import seaborn as sns
import networkx as nx
from src.network import ScienceNetworkModel
from src.scientists import ScientistAgent
import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

colors = sns.color_palette("Set2", 8)


def animate_model(model, num_frames=200, interval=500, steps_per_frame=1, max_steps=None):
    """
    Function to make an animation of the evolution of the model.

    Args:
        model (ScienceNetworkModel): The model to animate.
        num_frames (int): Number of frames in the animation.
        interval (int): Interval between frames in milliseconds.
        steps_per_frame (int): Number of steps to take in the model per frame.
        max_steps (int, optional): Maximum number of steps to run the model before stopping the animation.

    Returns:
        str: Path to the saved animation video file.
    """

    def get_colors():
        """
        Helper function to get the colors for the agents based on their current choice.

        Returns:
            list: List of colors corresponding to the agents' current choices.
        """
        return [colors[1] if agent.current_choice == "old" else colors[0] for agent in model.schedule.agents]

    fig, ax = plt.subplots(figsize=(6, 6))
    G = model.network
    pos = nx.spring_layout(G, seed=69)
    legend_elements = [
        Patch(facecolor=colors[1], edgecolor='k', label='Working with Old Theory'),
        Patch(facecolor=colors[0], edgecolor='k', label='Working with New Theory')
    ]

    should_stop = [False]  # Apparently use list cause its muteable

    def update(frame):
        """
        Update function for the animation. It steps the model and redraws the graph.

        Args:
            frame (int): Current frame number.
        """
        if should_stop[0]:
            return

        for _ in range(steps_per_frame):
            model.step()
            if model.converged or (max_steps and model.step_count >= max_steps):
                should_stop[0] = True
                break

        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=get_colors(), node_size=500, ax=ax)
        ax.set_title(f"Step {model.step_count}")
        ax.legend(handles=legend_elements, loc='upper right')

    if max_steps:
        actual_frames = min(num_frames, max_steps // steps_per_frame)
    else:
        actual_frames = num_frames

    anim = animation.FuncAnimation(fig, update, frames=actual_frames, interval=interval, repeat=False)

    os.makedirs("figures/animations", exist_ok=True)
    video_path = f"figures/animations/model_evolution_{model.network_type}_{model.num_agents}_{model.agent_class.__name__}.mkv"
    anim.save(video_path, writer="ffmpeg", dpi=300)
    plt.close()
    return video_path


def run_simulation_until_convergence(model, max_steps=1000):
    """
    This function runs the simulation until there is a convergence of all the
    scientists believing the same theory.

    Args:
        model (ScienceNetworkModel): The model to run.
        max_steps (int): Maximum number of steps to run the simulation.

    Returns:
        dict: Convergence information including whether the simulation converged,
              the step at which it converged, and the theory to which it converged.
    """
    while not model.converged and model.step_count < max_steps:
        model.step()

    conv_info = model.get_convergence_info()
    if conv_info['converged']:
        print(f"The simulation converged at step {conv_info['step']} to the {conv_info['theory']}!\n")
    else:
        print(f"Simulation did not converge within {max_steps} steps\n")

    return conv_info


def create_and_run_model(
                         agent_class=ScientistAgent,  # Choice: ScientistAgent or SuperScientistAgent
                         num_agents=10,
                         network_type="cycle",
                         old_theory_payoff=0.5,
                         new_theory_payoffs=(0.4, 0.6),
                         true_theory="new",
                         belief_strength_range=(0.5, 2.0),
                         use_animation=False,
                         max_steps=1000,
                         animation_params=None,
                         noise="off",
                         noise_std=0.0):
    """
    Function to create and run the ScienceNetworkModel with specified parameters.

    Args:
        all the same args that are used in the ScienceNetworkModel
        agent_class (class); (see src/scientists.py for arguments)

    Returns:
        If use_animation is True, returns the animation object.
        If use_animation is False, returns a dictionary with convergence information.
        (See ScienceNetworkModel.get_convergence_info() for details)
    """

    model = ScienceNetworkModel(
        agent_class=agent_class,
        num_agents=num_agents,
        network_type=network_type,
        old_theory_payoff=old_theory_payoff,
        new_theory_payoffs=new_theory_payoffs,
        true_theory=true_theory,
        belief_strength_range=belief_strength_range,
        noise=noise,
        noise_std=noise_std
    )

    if use_animation:
        default_params = {
            'num_frames': 30,
            'interval': 500,
            'steps_per_frame': 1
        }
        if animation_params:
            default_params.update(animation_params)

        return animate_model(model, max_steps=max_steps, **default_params)
    else:
        conv_info = run_simulation_until_convergence(model, max_steps)
        return conv_info


def save_results_as_csv(results, filename="test_results.csv"):
    """
    Just to save the results as csv

    Args:
        results (list): List of dictionaries containing simulation results.
        filename (str): Name of the output CSV file.
    """
    df = pd.DataFrame(results)

    os.makedirs("simulation_results", exist_ok=True)
    filepath = f"simulation_results/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def _run_single_simulation(args):
    """Helper function for parallel processing

    Args:
        args (tuple): Tuple containing the simulation ID and parameters.
                      The first element is the simulation ID (int),
                      and the second element is a dictionary of parameters.

    Returns:
        dict: A dictionary containing the results of the simulation."""
    sim_id, params = args

    # Create and run the model
    model = ScienceNetworkModel(
        agent_class=params['agent_class'],
        num_agents=params['num_agents'],
        network_type=params['network_type'],
        old_theory_payoff=params['old_theory_payoff'],
        new_theory_payoffs=params['new_theory_payoffs'],
        true_theory=params['true_theory'],
        belief_strength_range=params['belief_strength_range'],
        noise=params['noise']
    )

    # Run until convergence
    while not model.converged and model.step_count < params['max_steps']:
        model.step()

    conv_info = model.get_convergence_info()

    # Add simulation data
    result = {
        'simulation_id': sim_id + 1,
        'agent_class': params['agent_class'].__name__,
        'network_type': params['network_type'],
        'num_agents': params['num_agents'],
        'old_theory_payoff': params['old_theory_payoff'],
        'new_theory_payoff_if_old_true': params['new_theory_payoffs'][0],
        'new_theory_payoff_if_new_true': params['new_theory_payoffs'][1],
        'true_theory': params['true_theory'],
        'belief_strength_range': params['belief_strength_range']
    }
    result.update(conv_info)

    return result


def run_simulations_until_convergence(num_simulations=100, num_agents=10, network_type="cycle",
                                      old_theory_payoff=0.5, new_theory_payoffs=(0.4, 0.6),
                                      true_theory="new", belief_strength_range=(0.5, 2.0),
                                      use_animation=False, max_steps=1000, noise="off", noise_std=0.0,
                                      animation_params=None,
                                      agent_class=ScientistAgent, custom_graph=None):
    """
    The most important function of this module that you can call on in the
    main.py to run multiple simulations.

    Args:
        All the same args that are used in the ScienceNetworkModel
        agent_class (class): The class of the agent to use in the model.

    Returns:
        If use_animation is True, returns the animation object.
        If use_animation is False, returns a list of dictionaries with convergence information
        for each simulation.
        (See ScienceNetworkModel.get_convergence_info() for details)
    """
    # Can't parallelize if using animation
    if use_animation:
        print(f"==> Running 1  simulation with {num_agents} agents on a {network_type} network :) <==\n")
        # Create and run the model
        animation = create_and_run_model(
            agent_class=agent_class,
            num_agents=num_agents,
            network_type=network_type if custom_graph is None else custom_graph,
            old_theory_payoff=old_theory_payoff,
            new_theory_payoffs=new_theory_payoffs,
            true_theory=true_theory,
            belief_strength_range=belief_strength_range,
            use_animation=use_animation,
            max_steps=max_steps,
            animation_params=animation_params,
            noise=noise,
            noise_std=noise_std
        )
        return animation
    else:
        # Prepare parameters for parallel processing
        params = {
            "agent_class": agent_class,
            "num_agents": num_agents,
            "network_type": network_type if custom_graph is None else custom_graph,
            "old_theory_payoff": old_theory_payoff,
            "new_theory_payoffs": new_theory_payoffs,
            "true_theory": true_theory,
            "belief_strength_range": belief_strength_range,
            "max_steps": max_steps,
            "noise": noise
        }

        # Parallelisation
        args_list = [(i, params) for i in range(num_simulations)]
        num_processes = min(cpu_count(), num_simulations)
        chunk_size = max(1, num_simulations // (num_processes * 4))

        print(f"\nRunning {num_simulations} simulations using {num_processes} CPU cores (chunk size: {chunk_size})...")

        # Run simulations in parallel with progress bar
        with Pool(processes=num_processes) as pool:
            all_results = list(tqdm(
                pool.imap(_run_single_simulation, args_list, chunksize=chunk_size),
                total=num_simulations,
                desc="Running simulations"
            ))

    os.makedirs(f"simulation_results/{network_type}", exist_ok=True)
    csv_filename = f"{network_type}/{num_agents}agents_{agent_class.__name__}_{num_simulations}sims_{belief_strength_range}.csv"
    save_results_as_csv(all_results, csv_filename)

    return all_results
