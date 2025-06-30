"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the utility functions to run sensitivity
analysis on the parameters of the model. It includes functions to run
sensitivity analysis for a single network type, compare sensitivity analysis
across different network types, and run a full sensitivity analysis with options
for single network and comparison analyses.
"""

from datetime import datetime
from src.sensitivity_analysis import SensitivityAnalyzer
from src.scientists import ScientistAgent
from src.super_scientist import SuperScientistAgent


def run_sensitivity_analysis(network_type="cycle", num_trajectories=10):
    """Run sensitivity analysis for a single network type and all metrics

    Args:
        network_type (str): Type of network to analyze (e.g., 'bipartite', 'cliques').
        num_trajectories (int): Number of trajectories to run for the analysis.

    Returns:
        dict: Dictionary containing sensitivity analysis results for each metric.
        str: Timestamp of the analysis run.
    """
    analyzer = SensitivityAnalyzer()
    metrics = ['convergence_time', 'correct_theory_rate', 'old_theory_rate']
    agent_types = [SuperScientistAgent]  # You can add ScientistAgent when needed

    print(f"\n=== Running sensitivity analysis for {network_type} network ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    for agent_class in agent_types:
        agent_name = agent_class.__name__
        print(f"\nUsing agent class: {agent_name}")
        results[agent_name] = {}

        # Run analysis for each metric
        for metric in metrics:
            print(f"\nAnalyzing metric: {metric}")
            Si, _ = analyzer.morris_analysis(
                num_trajectories=num_trajectories,
                network_type=network_type,
                output_metric=metric,
                agent_class=agent_class
            )
            results[agent_name][metric] = Si
            analyzer.print_sensitivity_results(Si)

    return results, timestamp


def run_network_sensitivity_comparison(num_trajectories=10):
    """Compare sensitivity analysis across different network types.

    Args:
        num_trajectories (int): Number of trajectories to run for the analysis.

    Returns:
        dict: Dictionary containing sensitivity analysis results for each metric
        and network type.
        str: Timestamp of the analysis runs
    """
    analyzer = SensitivityAnalyzer()
    metrics = ['convergence_time', 'correct_theory_rate', 'old_theory_rate']
    network_types = ['cycle', 'wheel', 'complete']

    print("\n=== Running network comparison analysis ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        metric_results = {}

        for network_type in network_types:
            print(f"\nRunning analysis for {network_type} network")
            Si, _ = analyzer.morris_analysis(
                num_trajectories=num_trajectories,
                network_type=network_type,
                output_metric=metric
            )
            metric_results[network_type] = Si

            # Print detailed results
            print(f"\nResults for {network_type} network:")
            analyzer.print_sensitivity_results(Si)

        all_results[metric] = metric_results

    return all_results, timestamp


def run_full_sensitivity_analysis(num_trajectories=10, run_single=True, run_comparison=True):
    """
    Run complete sensitivity analysis with options for single network and
    comparison analyses.

    Args:
        num_trajectories (int): Number of trajectories to run for the analysis.
        run_single (bool): Whether to run sensitivity analysis for each network type individually.
        run_comparison (bool): Whether to compare sensitivity analysis across network types.
    """
    if run_single:
        # Run analysis for each network type individually
        single_results = {}
        for network_type in ['bipartite', 'cliques']:
            results, _ = run_sensitivity_analysis(network_type, num_trajectories)
            single_results[network_type] = results

    if run_comparison:
        # Compare across network types for both agent types
        comparison_results = {}
        for agent_class in [SuperScientistAgent]:  # add ScientistAgent if needed
            agent_name = agent_class.__name__
            comparison_results[agent_name] = {}

            analyzer = SensitivityAnalyzer()
            for metric in ['convergence_time', 'correct_theory_rate', 'old_theory_rate']:
                metric_results = {}
                for network_type in ['bipartite', 'cliques']:
                    Si, _ = analyzer.morris_analysis(
                        num_trajectories=num_trajectories,
                        network_type=network_type,
                        output_metric=metric,
                        agent_class=agent_class
                    )
                    metric_results[network_type] = Si
                comparison_results[agent_name][metric] = metric_results

    print("\nAnalysis complete! Results saved to analysis_results directory.")
