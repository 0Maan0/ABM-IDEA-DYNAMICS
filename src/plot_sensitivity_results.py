"""
This module contains functions for creating plots from saved sensitivity analysis results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

colors = sns.color_palette("Set2", 8)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def load_sensitivity_results(network_type, metric, timestamp=None, agent_type=None):
    """
    Load sensitivity analysis results from CSV files.
    If timestamp is None, loads the most recent results.
    """
    if agent_type:
        results_dir = f"analysis_results/{network_type}"#/{agent_type}"
    else:
        results_dir = f"analysis_results/{network_type}"
    
    if timestamp is None:
        files = [f for f in os.listdir(results_dir) if f.startswith(metric)]
        if not files:
            raise FileNotFoundError(f"No results found for {metric} in {results_dir}")
        filename = sorted(files)[-1] 
    else:
        filename = f"{metric}_{timestamp}.csv"
    
    filepath = os.path.join(results_dir, filename)
    return pd.read_csv(filepath)

def plot_single_analysis(network_type, metric, timestamp=None, save=True, num_trajectories=None):
    """
    Create plots for a single sensitivity analysis result
    """
    agent_types = [ 'SuperScientistAgent'] #ScientistAgent # add when needed
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, agent_type in enumerate(agent_types):
        try:
            results_df = load_sensitivity_results(network_type, metric, timestamp)
            results_df = results_df.sort_values('mu_star', ascending=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # mu_star
            ax1.barh(range(len(results_df)), results_df['mu_star'], 
                    alpha=0.7, color=colors[0])
            ax1.set_yticks(range(len(results_df)))
            ax1.set_yticklabels(results_df['parameter'])
            ax1.set_xlabel('Mean Absolute Elementary Effect')
            ax1.set_title('Parameter Importance')
            
            # sigma
            ax2.barh(range(len(results_df)), results_df['sigma'],
                    alpha=0.7, color=colors[1])
            ax2.set_yticks(range(len(results_df)))
            ax2.set_yticklabels(results_df['parameter'])
            ax2.set_xlabel('Standard Deviation')
            ax2.set_title('Parameter Interactions/Non-linearity')
        except FileNotFoundError:
            print(f"Results not found for {network_type} - {metric} - {agent_type}")
            
    plt.suptitle(f'Sensitivity Analysis - {network_type.capitalize()} Network - {metric}')
    plt.tight_layout()
    
    if save:
        save_dir = f"analysis_plots/{network_type}"
        os.makedirs(save_dir, exist_ok=True)
        traj_str = f"_{num_trajectories}traj" if num_trajectories is not None else ""
        plt.savefig(f"{save_dir}/{network_type}_{metric}_sensitivity{traj_str}.pdf",
                   format='pdf', bbox_inches='tight')
    
    return fig

def plot_network_comparison(metric, network_types=['cycle', 'wheel', 'complete'], 
                          timestamp=None, save=True, num_trajectories=None):
    """
    Create comparison plots across different network types
    """
    agent_types = [ 'SuperScientistAgent'] # 'ScientistAgent' add  when needed
    all_results = []


    for network in network_types:
        for agent_type in agent_types:
            try:
                df = load_sensitivity_results(network, metric, timestamp)
                df['network_type'] = network
                df['agent_type'] = agent_type
                all_results.append(df)
            except FileNotFoundError:
                print(f"Results not found for {network} - {metric} - {agent_type}")
    if not all_results:
        raise FileNotFoundError(f"No results found for metric {metric} across networks {network_types}")
    
    combined_df = pd.concat(all_results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # mu_star comparison 
    palette = {net: colors[i] for i, net in enumerate(network_types)}
    # agent_palette = {'ScientistAgent': colors[0], 'SuperScientistAgent': colors[1]}
    sns.barplot(data=combined_df, x='mu_star', y='parameter', 
                hue='network_type', ax=ax1, palette=palette)
    ax1.set_xlabel('Mean Absolute Elementary Effect')
    ax1.set_title('Parameter Importance Across Network Types')
    
    # sigma comparison 
    sns.barplot(data=combined_df, x='sigma', y='parameter',
                hue='network_type', ax=ax2, palette=palette)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Parameter Interactions Across Network Types')
    
    plt.suptitle(f'Complete Comparison - {metric}')
    plt.tight_layout()
    
    if save:
        save_dir = "analysis_plots/comparison"
        os.makedirs(save_dir, exist_ok=True)
        traj_str = f"_{num_trajectories}traj" if num_trajectories is not None else ""
        plt.savefig(f"{save_dir}/network_comparison_{metric}{traj_str}.pdf",
                   format='pdf', bbox_inches='tight')
    
    return fig

def plot_all_metrics(network_type, timestamp=None, num_trajectories=None):
    """
    Create plots for all available metrics for a given network type
    """
    results_dir = f"analysis_results/{network_type}"
    metrics = set(f.split('_')[0] for f in os.listdir(results_dir))
    
    for metric in metrics:
        plot_single_analysis(network_type, metric, timestamp, num_trajectories=num_trajectories)
        
def plot_all_comparisons(timestamp=None, num_trajectories=None):
    """
    Create comparison plots for all available metrics across all network types
    """
    network_types = ['cycle', 'wheel', 'complete']
    all_metrics = set()
    
    # Collect metrics from all network directories
    for network in network_types:
        results_dir = f"analysis_results/{network}"
        if os.path.exists(results_dir):
            metrics = set(f.split('_')[0] for f in os.listdir(results_dir))
            all_metrics.update(metrics)
    
    if not all_metrics:
        raise FileNotFoundError("No metrics found in any network directory")
    
    for metric in all_metrics:
        plot_network_comparison(metric, timestamp=timestamp, num_trajectories=num_trajectories) 