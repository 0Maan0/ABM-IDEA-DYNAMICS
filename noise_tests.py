""" This module tests the effects of noise on the convergence. """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from src.run_model_utils import run_simulations_until_convergence

def run_noise_experiment(
    network_size=10,
    noise_levels=[0.0],
    num_simulations=500,
    max_steps=2000):
    """ Runs the code to analyse the effect of noise on convergence 
    towards the 'True' Theory."""
    
    pass

if __name__ == "__main__":
    noise_levels = np.linspace(0.0, 0.1, num=6)
    num_simulations=1000

    results = run_noise_experiment(
        network_size=10,
        noise_levels=noise_levels,
        num_simulations=num_simulations
    )