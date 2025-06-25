"""
This module reproduces Figures 2 and 3 from Zollman's paper:
'The Communication Structure of Epistemic Communities'
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
from src.run_model_utils import run_simulations_until_convergence
from src.scientists import ScientistAgent
from src.influential_scientists import SuperScientistAgent

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def run_zollman_experiment(agent_class = ScientistAgent, network_sizes=[2, 4, 6, 8, 10, 12], num_simulations=10000):
    """
    Runs the experiment from Zollman's paper with the same parameters.
    Returns results for plotting Figures 2 and 3.
    """
    results = {}
    
    for size in network_sizes:
        print(f"\n=== Running simulations for network size {size} ===")
        results[size] = {}
        
        for network_type in ['cycle', 'wheel', 'complete']:
            print(f"\nNetwork type: {network_type}")
            
            # Run simulations with Zollman's parameters
            sim_results = run_simulations_until_convergence(
                agent_class=agent_class, 
                num_simulations=num_simulations,
                num_agents=size,
                network_type=network_type,
                old_theory_payoff=0.5,
                new_theory_payoffs=(0.4, 0.6),  # New theory is actually better
                true_theory="new",
                belief_strength_range=(0.5, 2.0),
                max_steps=2000  
            )
            
            df = pd.DataFrame(sim_results)
            # Count only runs that reached correct beliefs
            correct_runs = df[df['theory'] == 'Correct Theory']
            success_rate = len(correct_runs) / num_simulations
            avg_steps = correct_runs['step'].mean() if not correct_runs.empty else 0
            
            results[size][network_type] = {
                'success_rate': success_rate,
                'avg_steps': avg_steps
            }
            
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average steps to success: {avg_steps:.1f} steps")
            
            # Verify all runs converged
            if len(df) != num_simulations:
                print(f"Warning: {num_simulations - len(df)} runs did not converge!")
    
    results_df = pd.DataFrame([
        {
            'size': size,
            'network_type': net_type,
            'success_rate': data['success_rate'],
            'avg_steps': data['avg_steps']
        }
        for size in results
        for net_type, data in results[size].items()
    ])
    
    os.makedirs("analysis_results", exist_ok=True)
    results_df.to_csv(f"analysis_results/zollman_results_{agent_class.__name__}_{num_simulations}sims.csv", index=False)
    print(f"\nNumerical results saved to analysis_results/zollman_results_{agent_class.__name__}_{num_simulations}sims.csv")
    
    return results

def load_results(num_simulations, agent_class=ScientistAgent):
    """
    Load results from saved CSV file.
    """
    filepath = f"analysis_results/zollman_results_{agent_class.__name__}_{num_simulations}sims.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Convert to the same format as run_zollman_experiment returns
    results = {}
    for size in df['size'].unique():
        results[size] = {}
        size_data = df[df['size'] == size]
        for _, row in size_data.iterrows():
            results[size][row['network_type']] = {
                'success_rate': row['success_rate'],
                'avg_steps': row['avg_steps']
            }
    
    return results

def plot_zollman_figures(num_simulations, agent_class=ScientistAgent, save_dir="analysis_plots/zollman_reproduction"):
    """
    Recreates Figures 2 and 3 from Zollman's paper with the same styling.
    Reads data from saved results file.
    """
    # Load results from file
    results = load_results(num_simulations, agent_class)
    
    sizes = sorted(results.keys())
    network_types = ['cycle', 'wheel', 'complete']
    markers = ['+', 'x', '*']
    colors = ['red', 'green', 'blue']
    
    # Figure 2: Learning Results
    plt.figure(figsize=(8, 6))
    for i, network_type in enumerate(network_types):
        success_rates = [results[size][network_type]['success_rate'] for size in sizes]
        plt.plot(sizes, success_rates, 
                marker=markers[i], 
                color=colors[i],
                label=network_type.capitalize(),
                linestyle='-',
                markersize=8)
    
    plt.xlabel('Size')
    plt.ylabel('Probability of Successful Learning')
    plt.legend()
    plt.grid(True)
    plt.xlim(2, 12)
    plt.ylim(0.6, 1.0)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/learning_results_{agent_class.__name__}_{num_simulations}sims.pdf", 
                format='pdf', bbox_inches='tight')
    
    # Figure 3: Speed Results
    plt.figure(figsize=(8, 6))
    for i, network_type in enumerate(network_types):
        steps = [results[size][network_type]['avg_steps'] for size in sizes]
        plt.plot(sizes, steps, 
                marker=markers[i], 
                color=colors[i],
                label=network_type.capitalize(),
                linestyle='-',
                markersize=8)
    
    plt.xlabel('Size')
    plt.ylabel('Average Steps to Success')
    plt.legend()
    plt.grid(True)
    plt.xlim(2, 12)
    plt.ylim(0, 1200)
    
    plt.savefig(f"{save_dir}/speed_results_{agent_class.__name__}_{num_simulations}sims.pdf", 
                format='pdf', bbox_inches='tight')
    
    print(f"\nPlots saved to {save_dir}/")

if __name__ == "__main__":
    num_simulations = 10000  # Set number of simulations
    run_again = True  # Set to False if you just want to plot
    agent_class = ScientistAgent  # Choose between ScientistAgent or SuperScientistAgent
    results_file = f"analysis_results/zollman_results_{agent_class.__name__}_{num_simulations}sims"
    if run_again:
        print("=== Running Zollman's (2011) experiment ===")
        run_zollman_experiment(
            agent_class=agent_class,  
            network_sizes=[2, 4, 6, 8, 10, 12],
            num_simulations=num_simulations
        )
    
    print("\n=== Creating plots ===")
    plot_zollman_figures(num_simulations, agent_class=agent_class) 