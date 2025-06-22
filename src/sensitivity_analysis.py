"""
This module performs sensitivity analysis on the ABM model.
"""
import numpy as np
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from src.network import ScienceNetworkModel
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from datetime import datetime

class SensitivityAnalyzer:
    def __init__(self):
        self.problem = {
            'num_vars': 6,
            'names': [
                'num_agents',
                'old_theory_payoff',
                'new_theory_payoff_old_true',
                'new_theory_payoff_new_true',
                'belief_strength_min',
                'belief_strength_max'
            ],
            'bounds': [
                [5, 50],       # num_agents
                [0.1, 0.9],    # old_theory_payoff
                [0.1, 0.9],    # new_theory_payoff when old theory is true
                [0.1, 0.9],    # new_theory_payoff when new theory is true
                [0.1, 2.0],    # belief_strength_range min
                [0.5, 4.0]     # belief_strength_range max
            ]
        }
        
        # Create directories for results if they don't exist
        os.makedirs("analysis_results", exist_ok=True)
        os.makedirs("analysis_plots", exist_ok=True)
    
    def _run_single_instance(self, args):
        """Helper function for parallel processing"""
        try:
            params, network_type, num_steps, num_replicates = args
            
            # Store results from multiple replicates
            results = []
            
            for rep in range(num_replicates):
                # Create and run model
                model = ScienceNetworkModel(
                    num_agents=int(params[0]),
                    network_type=network_type,
                    old_theory_payoff=params[1],
                    new_theory_payoffs=(params[2], params[3]),
                    true_theory="new",  # Fixed for sensitivity analysis
                    belief_strength_range=(params[4], max(params[4] + 0.1, params[5]))
                )
                
                # Run until convergence or max steps
                for _ in range(num_steps):
                    if model.converged:
                        break
                    model.step()
                
                # Get convergence info
                conv_info = model.get_convergence_info()
                results.append(conv_info)
            
            # Calculate metrics
            df = pd.DataFrame(results)
            convergence_time = df[df['converged']]['step'].mean()
            if np.isnan(convergence_time):
                convergence_time = num_steps
                
            correct_theory_rate = (df['theory'] == 'Correct Theory').mean()
            old_theory_rate = (df['theory'] == 'Old Theory').mean()
            
            return {
                'convergence_time': convergence_time,
                'correct_theory_rate': correct_theory_rate,
                'old_theory_rate': old_theory_rate
            }
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            return {
                'convergence_time': num_steps,
                'correct_theory_rate': 0,
                'old_theory_rate': 0
            }
    
    def save_sensitivity_results(self, Si, network_type, output_metric, timestamp):
        """Save sensitivity analysis results to CSV files"""
        # Create results directory if it doesn't exist
        results_dir = f"analysis_results/{network_type}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare results as DataFrames
        metrics_df = pd.DataFrame({
            'parameter': Si['names'],
            'mu_star': Si['mu_star'],
            'mu': Si['mu'],
            'sigma': Si['sigma'],
            'mu_star_conf': Si['mu_star_conf'],
            'mu_conf': Si['mu_conf'],
            'sigma_conf': Si['sigma_conf']
        })
        
        # Save to CSV
        filename = f"{results_dir}/{output_metric}_{timestamp}.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"Saved sensitivity results to {filename}")
        
        return filename
    
    def morris_analysis(self, num_trajectories=10, network_type="cycle", output_metric='convergence_time'):
        """
        Perform Morris sensitivity analysis and save results
        """
        print(f"Starting Morris analysis for {network_type} network...")
        print(f"Analyzing output metric: {output_metric}")
        
        # Generate timestamp for this analysis run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate Morris samples
        param_values = morris.sample(self.problem, num_trajectories)
        print(f"Generated {len(param_values)} parameter combinations")
        
        # Prepare arguments for parallel processing
        args_list = [(params, network_type, 500, 2) for params in param_values]  # Reduced steps and replicates
        
        # Use fewer processes and larger chunks for better efficiency
        num_processes = min(8, cpu_count() - 1)  # Leave one core free
        chunk_size = max(2, len(param_values) // (num_processes * 2))
        
        print(f"Running simulations using {num_processes} CPU cores (chunk size: {chunk_size})...")
        
        # Run simulations in parallel with progress bar
        results = []
        with Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(self._run_single_instance, args_list, chunksize=chunk_size),
                total=len(param_values),
                desc="Running simulations"
            ):
                results.append(result)
        
        # Extract the requested metric from results
        Y = [result[output_metric] for result in results]
        
        # Analyze results
        Si = morris_analyze.analyze(self.problem, param_values, np.array(Y))
        
        # Save results
        results_file = self.save_sensitivity_results(Si, network_type, output_metric, timestamp)
        print(f"Analysis complete. Results saved to: {results_file}")
        
        return Si, timestamp
    
    def compare_networks(self, num_trajectories=10, output_metric='convergence_time'):
        """
        Compare sensitivity across different network types
        """
        networks = ['cycle', 'wheel', 'complete']
        results = {}
        timestamps = {}
        
        for network in networks:
            print(f"\n--- Analyzing {network} network ---")
            Si, timestamp = self.morris_analysis(num_trajectories, network, output_metric)
            results[network] = Si
            timestamps[network] = timestamp
        
        return results, timestamps
    
    def plot_morris_results(self, Si, title="Morris Sensitivity Analysis"):
        """
        Plot Morris sensitivity analysis results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot mu_star (mean of absolute elementary effects)
        ax1.barh(range(len(Si['names'])), Si['mu_star'])
        ax1.set_yticks(range(len(Si['names'])))
        ax1.set_yticklabels(Si['names'])
        ax1.set_xlabel('Mean Absolute Elementary Effect')
        ax1.set_title('Parameter Importance')
        
        # Plot sigma (standard deviation of elementary effects)
        ax2.barh(range(len(Si['names'])), Si['sigma'])
        ax2.set_yticks(range(len(Si['names'])))
        ax2.set_yticklabels(Si['names'])
        ax2.set_xlabel('Standard Deviation')
        ax2.set_title('Parameter Interactions/Non-linearity')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def print_sensitivity_results(self, Si):
        """
        Print formatted sensitivity analysis results
        """
        print("\n" + "="*50)
        print("MORRIS SENSITIVITY ANALYSIS RESULTS")
        print("="*50)
        
        print(f"{'Parameter':<25} {'μ*':<10} {'σ':<10} {'μ':<10}")
        print("-" * 55)
        
        for i, name in enumerate(Si['names']):
            print(f"{name:<25} {Si['mu_star'][i]:<10.3f} {Si['sigma'][i]:<10.3f} {Si['mu'][i]:<10.3f}")
        
        print("\nInterpretation:")
        print("- μ* (mu_star): Overall parameter importance")
        print("- σ (sigma): Indicates non-linear effects or interactions")
        print("- μ (mu): Mean elementary effect (can be negative)")