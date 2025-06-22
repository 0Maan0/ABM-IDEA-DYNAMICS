"""
This module contains functions for running sensitivity analysis on the model.
"""
import os
from datetime import datetime
from src.sensitivity_analysis import SensitivityAnalyzer

def run_sensitivity_analysis(network_type="cycle", num_trajectories=10):
    """Run sensitivity analysis for a single network type and all metrics"""
    analyzer = SensitivityAnalyzer()
    metrics = ['convergence_time', 'correct_theory_rate', 'old_theory_rate']
    
    print(f"\n=== Running sensitivity analysis for {network_type} network ===")
    
    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        Si, _ = analyzer.morris_analysis(
            num_trajectories=num_trajectories,
            network_type=network_type,
            output_metric=metric
        )
        results[metric] = Si
        
        # Print detailed results
        analyzer.print_sensitivity_results(Si)
    
    return results, timestamp

def run_network_sensitivity_comparison(num_trajectories=10):
    """Compare sensitivity analysis across different network types"""
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
    """Run complete sensitivity analysis with options for single network and comparison analyses"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if run_single:
        # Run analysis for each network type individually
        single_results = {}
        for network_type in ['cycle', 'wheel', 'complete']:
            results, _ = run_sensitivity_analysis(network_type, num_trajectories)
            single_results[network_type] = results
    
    if run_comparison:
        # Run comparison across all network types
        comparison_results, _ = run_network_sensitivity_comparison(num_trajectories)
    
    print("\nAnalysis complete! Results have been saved to analysis_results directory.") 