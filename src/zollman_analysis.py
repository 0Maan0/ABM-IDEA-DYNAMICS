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

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def run_zollman_experiment(network_sizes=[2, 4, 6, 8, 10, 12], num_simulations=10000):
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
                num_simulations=num_simulations,
                num_agents=size,
                network_type=network_type,
                old_theory_payoff=0.5,
                new_theory_payoffs=(0.4, 0.6),  # New theory is actually better
                true_theory="new",
                belief_strength_range=(0.5, 2.0),
                max_steps=2000  # Increased to ensure convergence
            )
            
            # Calculate metrics
            df = pd.DataFrame(sim_results)
            # Count only runs that reached correct beliefs
            correct_runs = df[df['theory'] == 'Correct Theory']
            success_rate = len(correct_runs) / num_simulations
            avg_time = correct_runs['step'].mean() if not correct_runs.empty else 0
            
            results[size][network_type] = {
                'success_rate': success_rate,
                'avg_time': avg_time
            }
            
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average time to success: {avg_time:.1f} steps")
            
            # Verify all runs converged
            if len(df) != num_simulations:
                print(f"Warning: {num_simulations - len(df)} runs did not converge!")
    
    return results

def plot_zollman_figures(results, save_dir="analysis_plots"):
    """
    Recreates Figures 2 and 3 from Zollman's paper with the same styling.
    """
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_dir}/learning_results_{timestamp}.pdf", 
                format='pdf', bbox_inches='tight')
    
    # Figure 3: Speed Results
    plt.figure(figsize=(8, 6))
    for i, network_type in enumerate(network_types):
        times = [results[size][network_type]['avg_time'] for size in sizes]
        plt.plot(sizes, times, 
                marker=markers[i], 
                color=colors[i],
                label=network_type.capitalize(),
                linestyle='-',
                markersize=8)
    
    plt.xlabel('Size')
    plt.ylabel('Average Time to Success')
    plt.legend()
    plt.grid(True)
    plt.xlim(2, 12)
    plt.ylim(0, 1200)
    
    plt.savefig(f"{save_dir}/speed_results_{timestamp}.pdf", 
                format='pdf', bbox_inches='tight')
    
    print(f"\nPlots saved to {save_dir}/")
    return timestamp

if __name__ == "__main__":
    # Run experiment
    print("=== Running Zollman's (2011) experiment ===")
    results = run_zollman_experiment(
        network_sizes=[2, 4, 6, 8, 10, 12],  
        num_simulations=1000  
    )
    
    # Create plots
    print("\n=== Creating plots ===")
    timestamp = plot_zollman_figures(results)
    
    # Save numerical results
    results_df = pd.DataFrame([
        {
            'size': size,
            'network_type': net_type,
            'success_rate': data['success_rate'],
            'avg_time': data['avg_time']
        }
        for size in results
        for net_type, data in results[size].items()
    ])
    
    os.makedirs("analysis_results", exist_ok=True)
    results_df.to_csv(f"analysis_results/zollman_results_{timestamp}.csv", index=False)
    print(f"\nNumerical results saved to analysis_results/zollman_results_{timestamp}.csv") 