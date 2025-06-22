"""
This module contains the functions we call in the main.py
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import networkx as nx
from src.network import ScienceNetworkModel
import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

colors = sns.color_palette("Set2", 8)

def animate_model(model, num_frames=200, interval=500, steps_per_frame=1, max_steps=None):
    """
    Function to make an animation of the evolution of the model.
    """
    def get_colors():
        return [colors[1] if agent.current_choice == "old" else colors[0] for agent in model.schedule.agents]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    G = model.network
    pos = nx.spring_layout(G, seed=69)  
    legend_elements = [
        Patch(facecolor=colors[1], edgecolor='k', label='Working with Old Theory'),
        Patch(facecolor=colors[0], edgecolor='k', label='Working with New Theory')
    ]
    
    should_stop = [False] # Apparently use list cause its muteable
    
    def update(frame):
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
    plt.show()

def run_simulation_until_convergence(model, max_steps=1000):
    """
    This function runs the simulation until there is a convergence of all the scientists believing the same theory
    """
    while not model.converged and model.step_count < max_steps:
        model.step()
    
    conv_info = model.get_convergence_info()
    if conv_info['converged']:
        print(f"The simulation converged at step {conv_info['step']} to the {conv_info['theory']}!\n")
    else:
        print(f"Simulation did not converge within {max_steps} steps\n") 
    
    return conv_info

def show_final_state(model):
    # just to show the final state of the model so 
    # we dont need to model all iterations but you can check convergence
    animate_model(model, num_frames=1, interval=1000, steps_per_frame=1)

def create_and_run_model(
    num_agents=10, 
    network_type="cycle",
    old_theory_payoff=0.5, 
    new_theory_payoffs=(0.4, 0.6),
    true_theory="new", 
    belief_strength_range=(0.5, 2.0),
    use_animation=False,
    max_steps=1000,
    animation_params=None,
    show_final_state=False):

    model = ScienceNetworkModel(
        num_agents=num_agents,
        network_type=network_type,
        old_theory_payoff=old_theory_payoff,
        new_theory_payoffs=new_theory_payoffs,
        true_theory=true_theory,
        belief_strength_range=belief_strength_range
    )
    
    if use_animation:
        default_params = {
            'num_frames': 30,
            'interval': 500,
            'steps_per_frame': 1
        }
        if animation_params:
            default_params.update(animation_params)
        
        animate_model(model, max_steps=max_steps, **default_params)
    else:
        conv_info = run_simulation_until_convergence(model, max_steps)
        if show_final_state:
            show_final_state(model)
        return conv_info

def save_results_as_csv(results, filename="test_results.csv"):
    """
    Just to save the results as csv
    """
    df = pd.DataFrame(results)
    
    os.makedirs("simulation_results", exist_ok=True)
    filepath = f"simulation_results/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def _run_single_simulation(args):
    """Helper function for parallel processing"""
    sim_id, params = args
    
    # Create and run the model
    model = ScienceNetworkModel(
        num_agents=params['num_agents'],
        network_type=params['network_type'],
        old_theory_payoff=params['old_theory_payoff'],
        new_theory_payoffs=params['new_theory_payoffs'],
        true_theory=params['true_theory'],
        belief_strength_range=params['belief_strength_range']
    )
    
    # Run until convergence
    while not model.converged and model.step_count < params['max_steps']:
        model.step()
    
    # Get convergence info
    conv_info = model.get_convergence_info()
    
    # Add simulation metadata
    result = {
        'simulation_id': sim_id + 1,
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
                                     use_animation=False, max_steps=1000, animation_params=None, show_final_state=False):
    """
    The most important function of this module that you can call on in the main.py to run multiple simulations
    """
    # Can't parallelize if using animation or showing final state
    if use_animation or show_final_state:
        all_results = []
        for i in range(num_simulations):
            print(f"==> Running simulation {i + 1} with {num_agents} agents on a {network_type} network :) <==\n")
            result = {}
            result['simulation_id'] = i + 1
            result['network_type'] = network_type
            
            # Create and run the model
            conv_info = create_and_run_model(
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
            
            result.update(conv_info)
            result['num_agents'] = num_agents
            result['old_theory_payoff'] = old_theory_payoff
            result['new_theory_payoff_if_old_true'] = new_theory_payoffs[0]
            result['new_theory_payoff_if_new_true'] = new_theory_payoffs[1]
            result['true_theory'] = true_theory
            result['belief_strength_range'] = belief_strength_range
            all_results.append(result)
    else:
        # Prepare parameters for parallel processing
        params = {
            "num_agents": num_agents,
            "network_type": network_type,
            "old_theory_payoff": old_theory_payoff,
            "new_theory_payoffs": new_theory_payoffs,
            "true_theory": true_theory,
            "belief_strength_range": belief_strength_range,
            "max_steps": max_steps
        }
        
        # Create arguments for parallel processing
        args_list = [(i, params) for i in range(num_simulations)]
        
        # Calculate optimal number of processes and chunk size
        num_processes = min(cpu_count(), num_simulations)
        chunk_size = max(1, num_simulations // (num_processes * 4))  # Adjust chunk size based on number of simulations
        
        print(f"\nRunning {num_simulations} simulations using {num_processes} CPU cores (chunk size: {chunk_size})...")
        
        # Run simulations in parallel with progress bar
        with Pool(processes=num_processes) as pool:
            all_results = list(tqdm(
                pool.imap(_run_single_simulation, args_list, chunksize=chunk_size),
                total=num_simulations,
                desc="Running simulations"
            ))
    
    # Save results
    os.makedirs(f"simulation_results/{network_type}", exist_ok=True)
    csv_filename = f"{network_type}/{num_agents}agents_{num_simulations}sims_{belief_strength_range}.csv"
    save_results_as_csv(all_results, csv_filename)
    
    return all_results

def plot_belief_evolution(model_history):
    # Maybe here we can make some function to plot how the beliefs of the agents evolved over time
    # maybe they go up and down or more linear?
    pass

def plot_network_statistics(model):
    #here we can call some function statistics
    # id say write a seperate class for them but call on them in a nice manner here
    # this way we can keep the main clean
    pass


