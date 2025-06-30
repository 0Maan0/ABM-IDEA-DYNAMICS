"""
University: University of Amsterdam
Course: Agent Based Modelling
Authors: Margarita Petrova; Pjotr Piet; Maan Scipio; Fred Loth;
UvaNetID's: 15794717; 12714933; 15899039; 12016926

Description: This file contains the code for running noise experiments
to test the impact of noise on the learning success of agents in different
network types and sizes. It includes functions to run simulations,
plot results, and save the results to CSV files.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from src.run_model_utils import run_simulations_until_convergence
from src.super_scientist import SuperScientistAgent


def run_noise_experiment(
    network_sizes=[4, 6, 8, 10, 12],
    noise_levels=[0.0, 0.1, 0.2, 0.3],
    num_simulations=1000,
    max_steps=2000
):
    """
    Runs simulations for different noise levels, with network size fixed.

    Args:
        network_sizes (list): List of network sizes to test.
        noise_levels (list): List of noise standard deviations to test.
        num_simulations (int): Number of simulations to run for each configuration.
        max_steps (int): Maximum number of steps for each simulation.

    Returns:
        results (dict): Dictionary containing success rates and average steps
                        for each network type, noise level, and size.
    """
    results = {}

    # network_types = ScienceNetworkModel.network_types
    network_types = ["cycle", "wheel", "complete"]  # For now, only these networks
    for network_type in network_types:
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
                    noise="on",
                    noise_std=noise_std,
                    agent_class=SuperScientistAgent  # Choice: ScientistAgent OR SuperScientistAgent
                )

                df = pd.DataFrame(sim_results)
                correct = df[df['theory'] == 'Correct Theory']
                success_rate = len(correct) / num_simulations
                avg_steps = correct['step'].mean() if not correct.empty else 0

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


def plot_success_vs_noise(
        results, save_path="analysis_plots/noise_success_plot.pdf"):
    """
    Plots the success rate of learning against noise levels for different network types.
    Each network type is represented by a different color.

    Args:
        results (dict): Dictionary containing success rates for each network type
        and noise level.
        save_path (str): Path to save the plot.
    """
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

    Args:
        results (dict): Dictionary containing success rates for each network type,
                        noise level, and size.
        save_dir (str): Directory to save the plots.
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


def save_noise_results_as_csv(results, num_simulations, filename_prefix="noise_results"):
    """
    Saves noise experiment results to CSV in a format.
    Columns: network_type, noise_std, size, success_rate, avg_steps

    Args:
        results (dict): Dictionary containing success rates and average steps
                        for each network type, noise level, and size.
        num_simulations (int): Number of simulations run for each configuration.
        filename_prefix (str): Prefix for the output CSV file name.
    """
    rows = []

    for network_type, noise_dict in results.items():
        for noise_std, size_dict in noise_dict.items():
            for size, metrics in size_dict.items():
                rows.append({
                    'network_type': network_type,
                    'noise_std': noise_std,
                    'size': size,
                    'success_rate': metrics['success_rate'],
                    'avg_steps': metrics['avg_steps']
                })

    df = pd.DataFrame(rows)

    # Make sure directory exists
    os.makedirs("analysis_results", exist_ok=True)

    # Construct filename
    filename = f"analysis_results/{filename_prefix}_{num_simulations}sims.csv"
    df.to_csv(filename, index=False)
    print(f"\nNoise experiment results saved to {filename}")


if __name__ == "__main__":
    noise_levels = [0.3, 0.7]
    network_sizes = [2, 4]
    num_simulations = 100

    results = run_noise_experiment(
        network_sizes=network_sizes,
        noise_levels=noise_levels,
        num_simulations=num_simulations
    )

    # This does not
    save_noise_results_as_csv(results, num_simulations)
    plot_success_vs_size_per_network(results,
                                     save_dir="analysis_plots/noise_vs_size")
