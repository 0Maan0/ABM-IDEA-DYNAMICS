# ABM-IDEA-DYNAMICS

A Python-based Agent-Based Model for simulating how scientific theories spread through networks of researchers. 

## About the paper this model is based on (https://www.researchgate.net/publication/228973111_The_Communication_Structure_of_Epistemic_Communities):

There is an old established theory and a new theory is introduced (only one of them can be correct). Each scientist forms a random belief about both theories which is uniformly drawn from a probability distribution. After choosing a theory they will experiment and report their results and adjust their opinion on the results of their experiments and the experiments of their neighbours. After this the process repeats and each agent chooses a theory again. This continues until either every agent believes the same theory with a probability above 0.9999 or that in the case that every take the action to experiment on the old theory where no new information can arrive. The scientists are connected through different types of networks. 

## Getting Started

To run the model, just execute main.py and change the parameters to get what you want:

```bash
python main.py
```

The model can also be used with the UI:
```bash
streamlit run src/UI/streamlit_ui.py
```

To obtain the analysis and plots from the zollman paper, run:
```bash
python -m src.zollman_analysis
```
To obtain the analysis of the noise parameter, run:
```bash
python -m src.noise_tests
```
## What's in Each File

- **main.py**: Here you can set up your simulation parameters and run the model
- **src/scientists.py**: Contains the ScientistAgent class that defines how individual scientists behave and update their beliefs
- **src/super_scientist.py**: Contains an enhanced version of the scientist agent with additional capabilities
- **src/network.py**: The main model (ScienceNetworkModel) that handles the simulation and network structure
- **src/run_model_utils.py**: Helper functions that make it easy to run simulations and analyze results
- **src/run_sensitivity_tools.py**: Tools for running sensitivity analysis on model parameters
- **src/sensitivity_analysis.py**: Core functionality for performing sensitivity analysis on the model
- **src/plot_sensitivity_results.py**: Functions for visualizing sensitivity analysis results
- **src/single_run_analysis.py**: Tools for analyzing individual simulation runs in detail
- **src/noise_tests.py**: Tests and analysis related to noise parameters in the model
- **src/zollman_analysis.py**: Reproduces figures from Zollman's original paper
- **src/UI/streamlit_ui.py**: Interactive web-based user interface for configuring and running simulations

## Networks

You can choose from the following different networks: **"cycle"**, **"wheel"**, **"complete"**, **"bipartite"**, **"cliques"**

Within the UI you can create your own **"custom"** network. 

## Set up 

In main.py, you can adjust the following parameters:

```python
# Model parameters
num_agents = 10                    # Number of scientists in the network
network_type = "cycle"            # Options: 'cycle', 'wheel', 'complete', 'bipartite', 'cliques'
agent_class = ScientistAgent      # Choose agent type: ScientistAgent or SuperScientistAgent

# Theory parameters
old_theory_payoff = 0.5          # Payoff for working with the old theory
new_theory_payoffs = (0.4, 0.6)  # Payoffs for new theory when (old theory true, new theory true)
true_theory = "new"              # Which theory is actually true
belief_strength_range = (0.5, 2.0) # Range affecting how resistant scientists are to changing beliefs

# Simulation settings
num_simulations = 2000           # How many times to run the simulation
use_animation = True             # Whether to animate one simulation run
max_steps = 1000                 # Maximum steps per simulation

# Analysis options
run_regular_simulations = True   # Run standard simulations with above parameters
run_sensitivity_analysis = False  # Run sensitivity analysis
create_sensitivity_plots = False  # Create plots from sensitivity analysis
num_trajectories = 715           # Number of trajectories for sensitivity analysis
```

## Features

- **Multiple Network Types**: Run simulations on cycle, wheel, or complete networks
- **Sensitivity Analysis**: Analyze how different parameters affect model outcomes
- **Visualization Tools**: Generate plots to understand simulation results
- **Parallel Processing**: Efficient simulation running using multiple CPU cores
- **Zollman Paper Reproduction**: Tools to reproduce figures from the original paper
