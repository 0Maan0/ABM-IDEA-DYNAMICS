"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the code to analyze and visualize the results
of single-run simulations of the ABM-IDEA-DYNAMICS model. It includes functions
to load simulation results, analyze convergence statistics, and create
visualizations using both Matplotlib and Plotly. The analysis focuses on
convergence rates, final consensus theories, and steps to convergence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px


def load_simulation_results(network_type: str, num_agents: int,
                            num_simulations: int, belief_range: tuple,
                            agent_type: str = "SuperScientistAgent"):
    """
    Load simulation results from the CSV file.

    Args:
        network_type (str): Type of network used in the simulation (e.g.,
        'cycle', 'wheel', 'comlete').
        num_agents (int): Number of agents in the simulation.
        num_simulations (int): Number of simulations run.
        belief_range (tuple): Range of beliefs used in the simulation (e.g.,
        (0.0, 1.0)).
        agent_type (str): Type of agent used in the simulation (default is
        "SuperScientistAgent").

    Returns:
        pd.DataFrame: DataFrame containing the simulation results.
    Raises:
        FileNotFoundError: If the results file does not exist.
    """
    filename = f"{num_agents}agents_{agent_type}_{num_simulations}sims_{belief_range}.csv"
    filepath = Path("simulation_results") / network_type / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    return pd.read_csv(filepath)


def analyze_convergence_stats(results_df):
    """
    Analyze convergence statistics from the simulation results.

    Args:
        results_df (pd.DataFrame): DataFrame containing the simulation results.

    Returns:
        dict: Dictionary containing convergence statistics.
    """
    total_runs = len(results_df)

    # Handle convergence stats
    converged_mask = results_df['converged'] == True
    converged_runs = converged_mask.sum()
    convergence_rate = converged_runs / total_runs * 100

    # Handle theory stats - looking for "Correct Theory" in the theory column
    correct_theory_mask = results_df['theory'] == 'Correct Theory'
    correct_theory_runs = correct_theory_mask.sum()
    correct_theory_rate = correct_theory_runs / total_runs * 100

    # Calculate steps statistics only for converged runs
    converged_steps = results_df[converged_mask]['step']
    avg_steps = converged_steps.mean() if len(converged_steps) > 0 else 0
    median_steps = converged_steps.median() if len(converged_steps) > 0 else 0

    return {
        'total_runs': total_runs,
        'converged_runs': converged_runs,
        'correct_theory_runs': correct_theory_runs,
        'convergence_rate': convergence_rate,
        'correct_theory_rate': correct_theory_rate,
        'avg_steps': avg_steps,
        'median_steps': median_steps
    }


def plot_convergence_analysis(results_df, network_type: str, num_agents: int):
    """
    Create visualization plots for the simulation results.

    Args:
        results_df (pd.DataFrame): DataFrame containing the simulation results.
        network_type (str): Type of network used in the simulation (e.g.,
        'cycle', 'wheel', 'complete').
        num_agents (int): Number of agents in the simulation.
    """
    # Set up the plotting style
    plt.style.use('seaborn')

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Distribution of steps to convergence
    sns.histplot(
        data=results_df[results_df['converged']],
        x='steps_to_convergence',
        bins=30,
        ax=ax1
    )
    ax1.set_title(f'Distribution of Steps to Convergence\n({network_type.capitalize()} Network, {num_agents} Agents)')
    ax1.set_xlabel('Steps to Convergence')
    ax1.set_ylabel('Count')

    # Plot 2: Final consensus distribution
    consensus_counts = results_df['theory'].value_counts()
    colors = ['#2ecc71' if theory == 'Correct Theory' else '#e74c3c' for theory in consensus_counts.index]
    consensus_counts.plot(
        kind='bar',
        ax=ax2,
        color=colors
    )
    ax2.set_title(f'Distribution of Final Consensus\n({network_type.capitalize()} Network, {num_agents} Agents)')
    ax2.set_xlabel('Final Consensus Theory')
    ax2.set_ylabel('Count')

    # Adjust layout and save
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = Path('analysis_plots/single_run')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(save_dir / f'convergence_analysis_{network_type}_{num_agents}agents.pdf')
    plt.close()


def print_analysis_summary(stats):
    """
    Print a summary of the analysis results.

    Args:
        stats (dict): Dictionary containing convergence statistics.

    Returns:
        None: Prints the summary to the console.
    """
    print("\n=== Simulation Analysis Summary ===")
    print(f"Total number of runs: {stats['total_runs']}")
    print(f"Number of converged runs: {stats['converged_runs']} ({stats['convergence_rate']:.1f}%)")
    print(f"Runs converged to correct theory: {stats['correct_theory_runs']} ({stats['correct_theory_rate']:.1f}%)")
    print(f"Average steps to convergence: {stats['avg_steps']:.1f}")
    print(f"Median steps to convergence: {stats['median_steps']:.1f}")


def analyze_and_plot_results(network_type: str, num_agents: int,
                             num_simulations: int, belief_range: tuple,
                             agent_type: str = "SuperScientistAgent"):
    """
    Main function to analyze and plot simulation results.

    Args:
        network_type (str): Type of network used in the simulation (e.g.,
        'cycle', 'wheel', 'complete').
        num_agents (int): Number of agents in the simulation.
        num_simulations (int): Number of simulations run.
        belief_range (tuple): Range of beliefs used in the simulation (e.g.,
        (0.0, 1.0)).
        agent_type (str): Type of agent used in the simulation (default is
        "SuperScientistAgent").

    Returns:
        dict: Dictionary containing convergence statistics.
    Raises:
        FileNotFoundError: If the results file does not exist.
    """
    # Load results
    results_df = load_simulation_results(network_type, num_agents,
                                         num_simulations, belief_range,
                                         agent_type)

    # Analyze results
    stats = analyze_convergence_stats(results_df)

    # Create plots
    plot_convergence_analysis(results_df, network_type, num_agents)

    # Print summary
    print_analysis_summary(stats)

    return stats


def plot_convergence_analysis_plotly(results_df, network_type: str,
                                     num_agents: int):
    """
    Create Plotly visualizations for the simulation results.

    Args:
        results_df (pd.DataFrame): DataFrame containing the simulation results.
        network_type (str): Type of network used in the simulation (e.g.,
        'cycle', 'wheel', 'complete').
        num_agents (int): Number of agents in the simulation.

    Returns:
        tuple: Tuple containing two Plotly figures:
            - fig_steps: Histogram of steps to convergence for converged runs.
            - fig_consensus: Bar chart of final consensus distribution.
    """
    # Plot 1: Distribution of steps to convergence for converged runs
    converged_df = results_df[results_df['converged'] == True]
    fig_steps = px.histogram(
        converged_df,
        x='step',
        nbins=30,
        title=f'Distribution of Steps to Convergence\n({network_type.capitalize()} Network, {num_agents} Agents)'
    )
    fig_steps.update_layout(
        xaxis_title="Steps to Convergence",
        yaxis_title="Count",
        title_x=0.5  # Center the title
    )

    # Plot 2: Final consensus distribution
    # Fill None values with "Not Converged" for better visualization
    theory_counts = results_df['theory'].fillna('Not Converged').value_counts()

    # Define colors for each possible outcome
    colors = {
        'Correct Theory': '#2ecc71',    # Green
        'Old Theory': '#e74c3c',        # Red
        'Incorrect Theory': '#e74c3c',  # Red
        'Not Converged': '#95a5a6'      # Gray
    }

    fig_consensus = px.bar(
        x=theory_counts.index,
        y=theory_counts.values,
        title=f'Distribution of Final Consensus\n({network_type.capitalize()} Network, {num_agents} Agents)',
        color=theory_counts.index,
        color_discrete_map=colors
    )
    fig_consensus.update_layout(
        xaxis_title="Final Consensus Theory",
        yaxis_title="Count",
        showlegend=False,
        title_x=0.5  # Center the title
    )

    return fig_steps, fig_consensus
