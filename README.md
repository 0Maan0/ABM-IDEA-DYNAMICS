# ABM-IDEA-DYNAMICS

A Python-based Agent-Based Model for simulating how scientific theories spread through networks of researchers. 

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

## Networks

You can choose from three different networks: **"cycle"**, **"wheel"**, **"complete"**

## Set up 

In main.py, you can adjust the following important parameters:

```python
num_agents = 10                    # How many scientists you have in your network
network_type = "cycle"            # Which network structure to use
true_probs = (0.2, 0.8)          # The actual probabilities that each theory is correct
num_simulations = 100             # How many times to run the simulation
```

## Priors

The priors we still need to define but its the way that the believes of each scientist are distributed at the start of the simulation.


