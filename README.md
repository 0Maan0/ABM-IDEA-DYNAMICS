# ABM-IDEA-DYNAMICS

A Python-based Agent-Based Model for simulating how scientific theories spread through networks of researchers. 

## About the paper this model is based on (https://www.researchgate.net/publication/228973111_The_Communication_Structure_of_Epistemic_Communities):
"We will, unbeknownst to our agents, set the world in φ2, where the new
methodology is better. We will then assign our agents random beliefs uniformly drawn from the interior of the probability space and allow them to
pursue the action they think best. They will then receive some return (a
"payoff") that is randomly drawn from a distribution for that action. The
agents will then update their beliefs about the state of the world based on
their results and the results of those to which they are connected. A population of agents is considered finished learning if one of two conditions are
met. First, a population has finished learning if every agent takes action A1,
in this case no new information can arrive which will convince our agents to
change strategies. (Remember that the payoff for action A1 is the same in
both states, so it is uninformative.) Alternatively the network has finished
learning if every agent comes to believe that they are in φ2 with probability
greater than 0.9999. Although it is possible that some unfortunate sequence
of results could drag these agents away, it is unlikely enough to be ignored."

## Getting Started

To run the model, just execute main.py and change the parameters to get what you want:

```bash
python main.py
```

## What's in Each File

- **main.py**: Here you can set up your simulation parameters and run the model
- **src/scientists.py**: Contains the ScientistAgent class that defines how individual scientists behave and update their beliefs
- **src/network.py**: The main model (ScienceNetworkModel) that handles the simulation and network structure
- **src/run_model_utils.py**: Helper functions that make it easy to run simulations and analyze results
- **src/run_sensitivity_tools.py**: Tools for running sensitivity analysis on model parameters
- **src/plot_sensitivity_results.py**: Functions for visualizing sensitivity analysis results
- **src/zollman_analysis.py**: Reproduces figures from Zollman's original paper

## Networks

You can choose from three different networks: **"cycle"**, **"wheel"**, **"complete"**

## Set up 

In main.py, you can adjust the following important parameters:

```python
num_agents = 10                    # How many scientists you have in your network
network_type = "cycle"            # Which network structure to use
old_theory_payoff = 0.5          # Payoff for working with the old theory
new_theory_payoffs = (0.4, 0.6)  # Payoffs for new theory when (old theory true, new theory true)
true_theory = "new"              # Which theory is actually true
belief_strength_range = (0.5, 2.0) # Range affecting how resistant scientists are to changing beliefs
num_simulations = 2000           # How many times to run the simulation
```

## Features

- **Multiple Network Types**: Run simulations on cycle, wheel, or complete networks
- **Sensitivity Analysis**: Analyze how different parameters affect model outcomes
- **Visualization Tools**: Generate plots to understand simulation results
- **Parallel Processing**: Efficient simulation running using multiple CPU cores
- **Zollman Paper Reproduction**: Tools to reproduce figures from the original paper


