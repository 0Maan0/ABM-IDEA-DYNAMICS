"""
This module contains functions for running sensitivity analysis on the model.
"""
import os
import matplotlib.pyplot as plt
from src.sensitivity_analysis import SensitivityAnalyzer

def run_sensitivity_analysis(network_type="cycle", num_trajectories=10):
    """Run sensitivity analysis for a single network type and all metrics"""
    analyzer = SensitivityAnalyzer()
    metrics = ['convergence_time', 'correct_theory_rate', 'old_theory_rate']
    
    print(f"\n=== Running sensitivity analysis for {network_type} network ===")
    
    # Create output directory for plots
    os.makedirs("analysis_plots/sensitivity", exist_ok=True)
    
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        Si = analyzer.morris_analysis(
            num_trajectories=num_trajectories,
            network_type=network_type,
            output_metric=metric
        )
        
        # Plot and save results
        fig = analyzer.plot_morris_results(Si, title=f"Sensitivity Analysis - {network_type.capitalize()} Network - {metric}")
        plt.savefig(f"analysis_plots/sensitivity/{network_type}_{metric}.png")
        plt.close(fig)
        
        # Print detailed results
        analyzer.print_sensitivity_results(Si)

def run_network_sensitivity_comparison(num_trajectories=10):
    """Compare sensitivity analysis across different network types"""
    analyzer = SensitivityAnalyzer()
    metrics = ['convergence_time', 'correct_theory_rate', 'old_theory_rate']
    
    print("\n=== Running network comparison analysis ===")
    
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        results = analyzer.compare_networks(
            num_trajectories=num_trajectories,
            output_metric=metric
        )
        
        # Print detailed results for each network
        for network_type, Si in results.items():
            print(f"\nResults for {network_type} network:")
            analyzer.print_sensitivity_results(Si)

def run_full_sensitivity_analysis(num_trajectories=10, run_single=True, run_comparison=True):
    """Run complete sensitivity analysis with options for single network and comparison analyses"""
    # Create output directory
    os.makedirs("analysis_plots/sensitivity", exist_ok=True)
    
    if run_single:
        # Run analysis for each network type individually
        for network_type in ['cycle', 'wheel', 'complete']:
            run_sensitivity_analysis(network_type, num_trajectories)
    
    if run_comparison:
        # Run comparison across all network types
        run_network_sensitivity_comparison(num_trajectories)
    
    print("\nAnalysis complete! Check the analysis_plots/sensitivity directory for results.") 