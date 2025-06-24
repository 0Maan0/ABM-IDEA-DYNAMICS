""" This module tests the effects of noise on the convergence. """

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from src.run_model_utils import run_simulations_until_convergence

def run_noise_experiment(
    network_sizes=[2, 4, 6, 8, 10, 12],
    noise_levels=[0.0],
    num_simulations=1000,
    max_steps=2000
):
    """
    Runs simulations for different noise levels, with network size fixed.
    Returns dictionary with results.
    """
    results = {}

    for network_type in ['cycle', 'wheel', 'complete']:
        results[network_type] = {}

        for noise_std in noise_levels:
            print(f"=== Running simulations for network {network_type}"
                  f" and noise with std = {noise_std} ===\n")
            results[network_type][noise_std] = {}

            for size in network_sizes:
                print(f"Network: {network_type}")

                sim_results = run_simulations_until_convergence(
                    num_simulations=num_simulations,
                    num_agents=size,
                    network_type=network_type,
                    old_theory_payoff=0.5,
                    new_theory_payoffs=(0.4, 0.6),
                    true_theory="new",
                    belief_strength_range=(0.5, 2.0),
                    max_steps=max_steps,
                    noise=noise_std
                )

                df = pd.DataFrame(sim_results)
                correct = df[df['theory'] == 'Correct Theory']
                success_rate = len(correct) / num_simulations
                avg_steps = correct['step'].mean() if not correct.empty else 0

                # ✅ Correct nested structure
                results[network_type][noise_std][size] = {
                    'success_rate': success_rate,
                    'avg_steps': avg_steps
                }

                print(f"Success rate: {success_rate:.2%}")
                print(f"Average steps to success: {avg_steps:.1f} steps")
                
                # Verify all runs converged
                if len(df) != num_simulations:
                    print(f"Warning: {num_simulations - len(df)}" 
                          " runs did not converge!")
                
    return results



def save_noise_results_as_csv(results, filename="noise_experiment_results.csv"):
    """ Saves the results in a csv file. """
    results_df = []

    for noise, network_data in results.items():
        for net, data in network_data.items():
            results_df.append({
                "noise": noise,
                "network_type": net,
                "success_rate": data["success_rate"],
                "avg_steps": data["avg_steps"]
            })
        
    df = pd.DataFrame(results_df)
    os.makedirs("analysis_results", exist_ok=True)
    filepath = f"analysis_results/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Numerical results saved to Results saved to {filepath}")


def plot_success_vs_noise(
        results, save_path="analysis_plots/noise_success_plot.pdf"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))

    network_types = ['cycle', 'wheel', 'complete']
    colors = {'cycle': 'red', 'wheel': 'green', 'complete': 'blue'}

    for network in network_types:
        x = []
        y = []
        for noise in sorted(results.keys()):
            if network in results[noise]:
                x.append(noise)
                y.append(results[noise][network]["success_rate"])
        plt.plot(
            x, y, marker='o', color=colors[network], label=network.capitalize())

    plt.xlabel("Noise Level (Gaussian scale)")
    plt.ylabel("Probability of Successful Learning")
    plt.title("Impact of Noise on Learning Success")
    plt.grid(True)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")


def plot_success_vs_size_per_network(
        results, save_dir="analysis_plots/noise_vs_size"):
    """
    Plots success rate vs network size for each network type.
    Each plot contains multiple noise levels.
    """
    os.makedirs(save_dir, exist_ok=True)

    # (Change later) Gives every noise level a different color
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(next(iter(results.values())))))

    for network_type in results:
        plt.figure(figsize=(8, 6))
        noise_levels = sorted(results[network_type].keys())
        sizes = sorted(next(iter(results[network_type].values())).keys())

        for i, noise in enumerate(noise_levels):
            success_rates = [
                results[network_type][noise][size]['success_rate'] 
                for size in sizes
                ]
            
            plt.plot(
                sizes, 
                success_rates, 
                marker='o', 
                label=f"Noise = {noise}", 
                color=colors[i]
                )

        plt.title(
            f"Learning Success vs. Network Size ({network_type.capitalize()})")
        plt.xlabel("Network Size")
        plt.ylabel("Probability of Successful Learning")
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.savefig(f"{save_dir}/success_vs_size_{network_type}.pdf", 
                    format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_dir}/success_vs_size_{network_type}.pdf")


if __name__ == "__main__":
    noise_levels = [0.0, 0.2, 0.5, 0.7, 0.9]
    network_sizes = [2, 4, 6, 8, 10, 12]
    num_simulations = 1000

    results = run_noise_experiment(
        network_sizes=network_sizes,
        noise_levels=noise_levels,
        num_simulations=num_simulations
    )

    save_noise_results_as_csv(results, filename="...")
    plot_success_vs_size_per_network(results, 
                                     save_dir="analysis_plots/noise_vs_size")