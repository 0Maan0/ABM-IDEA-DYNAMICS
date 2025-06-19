import numpy as np
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from src.network import ScienceNetworkModel
import matplotlib.pyplot as plt

class SensitivityAnalyzer:
    def __init__(self):
        self.problem = {
            'num_vars': 5,
            'names': ['num_agents', 'prior_old_strength', 'prior_new_strength', 'true_prob_old', 'true_prob_new'],
            'bounds': [
                [5, 50],      # num_agents
                [0.1, 0.9],   # prior_old_strength (how strong initial old beliefs are)
                [0.1, 0.9],   # prior_new_strength (how strong initial new beliefs are)
                [0.1, 0.9],   # true_prob_old (actual probability old theory is correct)
                [0.1, 0.9]    # true_prob_new (actual probability new theory is correct)
            ]
        }
    
    def run_model_instance(self, params, network_type="cycle", num_steps=50, num_replicates=5):
        """
        Run a single instance of the model with given parameters
        
        Returns:
        - Dictionary with outcome metrics
        """
        num_agents, prior_old_strength, prior_new_strength, true_prob_old, true_prob_new = params
        num_agents = int(num_agents)
        
        # Store results from multiple replicates to account for stochasticity
        convergence_times = []
        final_consensuses = []
        adoption_rates = []
        
        for rep in range(num_replicates):
            model = ScienceNetworkModel(
                num_agents=num_agents,
                network_type=network_type,
                true_probs=(true_prob_old, true_prob_new)
            )
            
            # Modify initial beliefs based on sensitivity parameters
            self._set_initial_beliefs(model, prior_old_strength, prior_new_strength)
            
            # Run simulation
            convergence_time, final_consensus, adoption_rate = self._run_simulation(model, num_steps)
            
            convergence_times.append(convergence_time)
            final_consensuses.append(final_consensus)
            adoption_rates.append(adoption_rate)
        
        return {
            'convergence_time': np.mean(convergence_times),
            'final_consensus': np.mean(final_consensuses),  # 0=old theory, 1=new theory
            'adoption_rate': np.mean(adoption_rates),
            'convergence_std': np.std(convergence_times)
        }
    
    def _set_initial_beliefs(self, model, prior_old_strength, prior_new_strength):
        """
        Set initial beliefs based on sensitivity analysis parameters
        """
        # Randomly select one agent to have strong belief in new theory
        original_agent = np.random.randint(0, model.num_agents)
        
        for i, agent in enumerate(model.schedule.agents):
            if i == original_agent:
                agent.belief_old = 1 - prior_new_strength
                agent.belief_new = prior_new_strength
            else:
                agent.belief_old = prior_old_strength
                agent.belief_new = 1 - prior_old_strength
            
            # Update current choice based on beliefs
            agent.current_choice = 0 if agent.belief_old > agent.belief_new else 1
    
    def _run_simulation(self, model, num_steps):
        """
        Run the simulation and return outcome metrics
        """
        initial_choices = [agent.current_choice for agent in model.schedule.agents]
        
        for step in range(num_steps):
            model.step()
            
            # Check for convergence
            current_choices = [agent.current_choice for agent in model.schedule.agents]
            if len(set(current_choices)) == 1:  # All agents agree
                convergence_time = step + 1
                final_consensus = current_choices[0]
                adoption_rate = sum(current_choices) / len(current_choices)
                return convergence_time, final_consensus, adoption_rate
        
        # If no convergence, return final state
        final_choices = [agent.current_choice for agent in model.schedule.agents]
        final_consensus = 1 if sum(final_choices) > len(final_choices) / 2 else 0
        adoption_rate = sum(final_choices) / len(final_choices)
        
        return num_steps, final_consensus, adoption_rate
    
    def morris_analysis(self, num_trajectories=10, network_type="cycle", output_metric='convergence_time'):
        """
        Perform Morris sensitivity analysis
        
        Parameters:
        - num_trajectories: Number of Morris trajectories to generate
        - network_type: Type of network ('cycle', 'wheel', 'complete')
        - output_metric: Which output to analyze ('convergence_time', 'adoption_rate', etc.)
        
        Returns:
        - Dictionary with Morris sensitivity indices
        """
        print(f"Starting Morris analysis for {network_type} network...")
        print(f"Analyzing output metric: {output_metric}")
        
        # Morris sample
        param_values = morris.sample(self.problem, num_trajectories)
        print(f"Generated {len(param_values)} parameter combinations")
        
        # Run model for each parameter combination
        Y = []
        for i, params in enumerate(param_values):
            if i % 10 == 0:
                print(f"Running simulation {i+1}/{len(param_values)}")
            
            result = self.run_model_instance(params, network_type)
            Y.append(result[output_metric])
        
        Si = morris_analyze.analyze(self.problem, param_values, np.array(Y))
        
        return Si
    
    def compare_networks(self, num_trajectories=10, output_metric='convergence_time'):
        """
        Compare sensitivity across different network types
        """
        networks = ['cycle', 'wheel', 'complete']
        results = {}
        
        for network in networks:
            print(f"\n--- Analyzing {network} network ---")
            results[network] = self.morris_analysis(num_trajectories, network, output_metric)
        
        return results
    
    def plot_morris_results(self, Si, title="Morris Sensitivity Analysis"):
        """
        Plot Morris sensitivity analysis results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot mu_star (mean of absolute elementary effects)
        ax1.barh(range(len(Si['names'])), Si['mu_star'])
        ax1.set_yticks(range(len(Si['names'])))
        ax1.set_yticklabels(Si['names'])
        ax1.set_xlabel('μ* (Mean Absolute Elementary Effect)')
        ax1.set_title('Parameter Importance')
        
        # Plot sigma (standard deviation of elementary effects)
        ax2.barh(range(len(Si['names'])), Si['sigma'])
        ax2.set_yticks(range(len(Si['names'])))
        ax2.set_yticklabels(Si['names'])
        ax2.set_xlabel('σ (Standard Deviation)')
        ax2.set_title('Parameter Interactions/Non-linearity')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_sensitivity_results(self, Si):
        """
        Print formatted sensitivity analysis results
        """
        print("\n" + "="*50)
        print("MORRIS SENSITIVITY ANALYSIS RESULTS")
        print("="*50)
        
        print(f"{'Parameter':<20} {'μ*':<10} {'σ':<10} {'μ':<10}")
        print("-" * 50)
        
        for i, name in enumerate(Si['names']):
            print(f"{name:<20} {Si['mu_star'][i]:<10.3f} {Si['sigma'][i]:<10.3f} {Si['mu'][i]:<10.3f}")
        
        print("\nInterpretation:")
        print("- μ* (mu_star): Overall parameter importance")
        print("- σ (sigma): Indicates non-linear effects or interactions")
        print("- μ (mu): Mean elementary effect (can be negative)")