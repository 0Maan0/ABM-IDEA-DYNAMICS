"""
This module contains functions for plotting sensitivity analysis results from saved CSV files.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_sensitivity_results(network_type, metric, timestamp=None):
    """
    Load sensitivity analysis results from CSV file.
    If timestamp is None, loads the most recent results for the given network and metric.
    """
    results_dir = f"analysis_results/{network_type}"
    
    if timestamp is None:
        # Find the most recent file for this metric
        files = [f for f in os.listdir(results_dir) if f.startswith(metric)]
        if not files:
            raise FileNotFoundError(f"No results found for {metric} in {results_dir}")
        filename = sorted(files)[-1]  # Get the most recent file
    else:
        filename = f"{metric}_{timestamp}.csv"
    
    filepath = os.path.join(results_dir, filename)
    return pd.read_csv(filepath)

def plot_sensitivity_results(results_df, network_type, metric, save=True):
    """
    Create sensitivity analysis plots from results DataFrame
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort parameters by importance (mu_star)
    results_df = results_df.sort_values('mu_star', ascending=True)
    
    # Plot mu_star with confidence intervals
    ax1.barh(range(len(results_df)), results_df['mu_star'], xerr=results_df['mu_star_conf'],
             capsize=5, alpha=0.7)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['parameter'])
    ax1.set_xlabel('Mean Absolute Elementary Effect')
    ax1.set_title('Parameter Importance')
    
    # Plot sigma with confidence intervals
    ax2.barh(range(len(results_df)), results_df['sigma'], xerr=results_df['sigma_conf'],
             capsize=5, alpha=0.7)
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['parameter'])
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Parameter Interactions/Non-linearity')
    
    plt.suptitle(f'Sensitivity Analysis - {network_type.capitalize()} Network - {metric}')
    plt.tight_layout()
    
    if save:
        # Save plot as PDF
        save_dir = f"analysis_plots/{network_type}"
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = f"{save_dir}/{metric}_sensitivity.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {pdf_path}")
    
    return fig

def plot_network_comparison(network_types=['cycle', 'wheel', 'complete'], 
                          metric='convergence_time', timestamps=None):
    """
    Create comparison plot of sensitivity results across different network types
    """
    # Set figure style for publication quality
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Load results for each network type
    all_results = []
    for network in network_types:
        df = load_sensitivity_results(network, metric, 
                                    timestamps[network] if timestamps else None)
        df['network_type'] = network
        all_results.append(df)
    
    combined_df = pd.concat(all_results)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot mu_star comparison
    sns.barplot(data=combined_df, x='mu_star', y='parameter', hue='network_type', ax=ax1)
    ax1.set_xlabel('Mean Absolute Elementary Effect')
    ax1.set_title('Parameter Importance Across Network Types')
    
    # Plot sigma comparison
    sns.barplot(data=combined_df, x='sigma', y='parameter', hue='network_type', ax=ax2)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Parameter Interactions Across Network Types')
    
    plt.suptitle(f'Network Comparison - {metric}')
    plt.tight_layout()
    
    # Save plot as PDF
    save_dir = "analysis_plots/comparison"
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = f"{save_dir}/{metric}_network_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Comparison plot saved to {pdf_path}")
    
    return fig

def plot_all_metrics(network_type, timestamp=None):
    """
    Create plots for all available metrics for a given network type
    """
    # Set figure style for publication quality
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    results_dir = f"analysis_results/{network_type}"
    metrics = set(f.split('_')[0] for f in os.listdir(results_dir))
    
    for metric in metrics:
        df = load_sensitivity_results(network_type, metric, timestamp)
        plot_sensitivity_results(df, network_type, metric) 