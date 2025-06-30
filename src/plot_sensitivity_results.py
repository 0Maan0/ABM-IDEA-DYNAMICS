"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the code to plot the results of sensitivity analysis
on the parameters of the model. It includes functions to load results from CSV files,
create plots for single analyses, compare results across different network types,
and plot all metrics for a given network type. The plots include bar charts for
parameter importance (mu_star) and parameter interactions (sigma).
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
fontsize = 12
ticksize = 10

def load_sensitivity_results(network_type, metric, timestamp=None, agent_type=None):
    """
    Load sensitivity analysis results from CSV files.
    If timestamp is None, loads the most recent results.

    Args:
        network_type (str): Type of network (e.g., 'cycle', 'wheel', 'complete').
        metric (str): Metric to load (e.g., 'mu_star', 'sigma').
        timestamp (str, optional): Specific timestamp to load results for.
        agent_type (str, optional): Type of agent used in the analysis.

    Returns:
        pd.DataFrame: DataFrame containing the sensitivity analysis results.
    Raises:
        FileNotFoundError: If no results are found for the specified metric and network type.
    """
    if agent_type:
        results_dir = f"analysis_results/{network_type}"  # {agent_type}"
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

    Args:
        network_type (str): Type of network (e.g., 'cycle', 'wheel', 'complete').
        metric (str): Metric to plot (e.g., 'mu_star', 'sigma').
        timestamp (str, optional): Specific timestamp to load results for.
        save (bool): Whether to save the plot as a PDF.
        num_trajectories (int, optional): Number of trajectories used in the analysis.

    Returns:
        fig: Matplotlib figure object containing the plots.
    Raises:
        FileNotFoundError: If no results are found for the specified metric and network type.
    """
    agent_types = ['SuperScientistAgent']   # ScientistAgent, add when needed

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
            ax1.set_yticklabels(results_df['parameter'], fontsize=ticksize)
            ax1.set_xlabel('Mean Absolute Elementary Effect', fontsize=fontsize)
            ax1.set_title('Parameter Importance', fontsize=fontsize)

            # sigma
            ax2.barh(range(len(results_df)), results_df['sigma'],
                     alpha=0.7, color=colors[1])
            ax2.set_yticks(range(len(results_df)))
            ax2.set_yticklabels(results_df['parameter'],fontsize=ticksize)
            ax2.set_xlabel('Standard Deviation')
            ax2.set_title('Parameter Interactions/Non-linearity',fontsize=fontsize)
        except FileNotFoundError:
            print(f"Results not found for {network_type} - {metric} - {agent_type}")

    plt.suptitle(f'Sensitivity Analysis - {network_type.capitalize()} Network - {metric} of {agent_types[0]}', fontsize=ticksize)
    plt.tight_layout()

    if save:
        save_dir = f"analysis_plots/{network_type}"
        os.makedirs(save_dir, exist_ok=True)
        traj_str = f"_{num_trajectories}traj" if num_trajectories is not None else ""
        plt.savefig(f"{save_dir}/{network_type}_{metric}_sensitivity{traj_str}_{agent_types[0]}.pdf",
                    format='pdf', bbox_inches='tight')
        plt.close()
    return fig


def plot_network_comparison(metric, network_types=['cycle', 'wheel', 'complete', 'bipartite', 'cliques'],
                            timestamp=None, save=True, num_trajectories=None):
    """
    Create comparison plots across different network types

    Args:
        metric (str): Metric to compare (e.g., 'mu_star', 'sigma').
        network_types (list): List of network types to compare.
        timestamp (str, optional): Specific timestamp to load results for.
        save (bool): Whether to save the plot as a PDF.
        num_trajectories (int, optional): Number of trajectories used in the analysis.

    Returns:
        fig: Matplotlib figure object containing the comparison plots.
    Raises:
        FileNotFoundError: If no results are found for the specified metric across the network
        types.
    """
    agent_types = ['SuperScientistAgent']  # 'ScientistAgent' add  when needed
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
    ax1.set_xlabel('Mean Absolute Elementary Effect',fontsize=fontsize)
    ax1.set_title('Parameter Importance Across Network Types',fontsize=fontsize)

    # sigma comparison
    sns.barplot(data=combined_df, x='sigma', y='parameter',
                hue='network_type', ax=ax2, palette=palette)
    ax2.set_xlabel('Standard Deviation',fontsize=fontsize)
    ax2.set_title('Parameter Interactions Across Network Types',fontsize=fontsize)

    plt.suptitle(f'Complete Comparison - {metric} of {agent_types[0]}')
    plt.tight_layout()

    if save:
        save_dir = "analysis_plots/comparison"
        os.makedirs(save_dir, exist_ok=True)
        traj_str = f"_{num_trajectories}traj" if num_trajectories is not None else ""
        plt.savefig(f"{save_dir}/network_comparison_{metric}{traj_str}_{agent_type[0]}.pdf",
                    format='pdf', bbox_inches='tight')
        plt.close()

    return fig


def plot_all_metrics(network_type, timestamp=None, num_trajectories=None):
    """
    Create plots for all available metrics for a given network type.

    Args:
        network_type (str): Type of network (e.g., 'cycle', 'wheel',
        'complete', 'bipartite', 'cliques').
        timestamp (str, optional): Specific timestamp to load results for.
        num_trajectories (int, optional): Number of trajectories used in the analysis.
    """
    results_dir = f"analysis_results/{network_type}"
    metrics = set(f.split('_')[0] for f in os.listdir(results_dir))

    for metric in metrics:
        plot_single_analysis(network_type, metric, timestamp, num_trajectories=num_trajectories)


def plot_all_comparisons(timestamp=None, num_trajectories=None):
    """
    Create comparison plots for all available metrics across all network types.

    Args:
        timestamp (str, optional): Specific timestamp to load results for.
        num_trajectories (int, optional): Number of trajectories used in the analysis.

    Raises:
        FileNotFoundError: If no metrics are found in any network directory.
    """
    network_types = ['cycle', 'wheel', 'complete', 'bipartite', 'cliques']
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

def create_summary_table(network_types=['cycle', 'wheel', 'complete', 'bipartite', 'cliques'], 
                        metrics=['convergence_time', 'correct_theory_rate', 'old_theory_rate'],
                        timestamp=None, num_trajectories=None):
    """
    Create a simple summary table showing top parameters for each network-metric combination
    """
    summary_data = []
    
    for network_type in network_types:
        for metric in metrics:
            try:
                # Use your existing function
                results_df = load_sensitivity_results(network_type, metric, timestamp)
                
                # Filter out belief strength parameters
                filtered_df = results_df[
                    ~results_df['parameter'].str.contains('belief_strength', case=False, na=False)
                ]
                
                # Get top 3 parameters by mu_star from filtered results
                top_3 = filtered_df.nlargest(3, 'mu_star')
                
                # Format the data with both mu_star and sigma
                top_params = []
                for i, (_, row) in enumerate(top_3.iterrows()):
                    # Handle both string and float sigma values
                    try:
                        sigma_val = float(row['sigma'])
                        top_params.append(f"{row['parameter']} ({row['mu_star']:.3f}, {sigma_val:.3f})")
                    except (ValueError, TypeError):
                        # If sigma is a string or can't be converted, use as-is
                        top_params.append(f"{row['parameter']} ({row['mu_star']:.3f}, {row['sigma']})")
                
                # Pad with empty strings if less than 3 parameters
                while len(top_params) < 3:
                    top_params.append("-")
                
                summary_data.append({
                    'Network': network_type.capitalize(),
                    'Metric': metric.replace('_', ' ').title(),
                    '1st Most Important': top_params[0],
                    '2nd Most Important': top_params[1],
                    '3rd Most Important': top_params[2]
                })
                
            except FileNotFoundError:
                print(f"Results not found for {network_type} - {metric}")
                summary_data.append({
                    'Network': network_type.capitalize(),
                    'Metric': metric.replace('_', ' ').title(),
                    '1st Most Important': "No data",
                    '2nd Most Important': "No data",
                    '3rd Most Important': "No data"
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def save_summary_table_latex(summary_df, filename="sensitivity_summary_table.tex"):
    """
    Save the summary table as LaTeX with updated caption for mu* and sigma values
    """
    latex_code = summary_df.to_latex(
        index=False,
        caption="Top three most important parameters by network type and metric ($\\mu^*$, $\\sigma$ values in parentheses)",
        label="tab:sensitivity_summary",
        escape=False,
        column_format='llp{3cm}p{3cm}p{3cm}'  # Better column formatting for long parameter names
    )
    
    os.makedirs("analysis_results", exist_ok=True)
    filepath = f"analysis_results/{filename}"
    
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX table saved to: {filepath}")
    return filepath

# Alternative version with improved LaTeX formatting
def save_summary_table_latex_improved(summary_df, filename="sensitivity_summary_table.tex"):
    """
    Save the summary table as properly formatted LaTeX with mu* and sigma values
    """
    # Create custom LaTeX table with better formatting
    latex_code = """\\begin{table*}[htbp]
\\centering
\\caption{Top three most important parameters by network type and metric ($\\mu^*$, $\\sigma$ values in parentheses)}
\\label{tab:sensitivity_summary}
\\footnotesize
\\begin{tabular}{llp{2.8cm}p{2.8cm}p{2.8cm}}
\\toprule
\\textbf{Network} & \\textbf{Metric} & \\textbf{1st Most Important} & \\textbf{2nd Most Important} & \\textbf{3rd Most Important} \\\\
\\midrule
"""
    
    # Add data rows with proper grouping
    current_network = ""
    for _, row in summary_df.iterrows():
        # Add separator between networks
        if current_network != "" and current_network != row['Network']:
            latex_code += "\\midrule\n"
        current_network = row['Network']
        
        # Format parameter names (escape underscores)
        col1 = row['1st Most Important'].replace('_', '\\_')
        col2 = row['2nd Most Important'].replace('_', '\\_')
        col3 = row['3rd Most Important'].replace('_', '\\_')
        
        latex_code += f"{row['Network']} & {row['Metric']} & {col1} & {col2} & {col3} \\\\\n"
    
    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table*}"""
    
    os.makedirs("analysis_results", exist_ok=True)
    filepath = f"analysis_results/{filename}"
    
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX table saved to: {filepath}")
    return filepath

# Simple usage with your existing code:
def generate_simple_summary(num_trajectories=None):
    """
    Generate and save a simple summary table using your existing functions
    """
    # Create the summary table
    summary_table = create_summary_table(num_trajectories=num_trajectories)
    
    # Display it
    print("Sensitivity Analysis Summary:")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    
    # Save as LaTeX (using the improved version)
    latex_file = save_summary_table_latex_improved(summary_table)
    
    # Also save as CSV for reference
    csv_file = "analysis_results/sensitivity_summary_table.csv"
    summary_table.to_csv(csv_file, index=False)
    print(f"CSV table saved to: {csv_file}")
    
    return summary_table