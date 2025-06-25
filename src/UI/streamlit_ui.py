import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from src.run_model_utils import run_simulations_until_convergence
from src.run_sensitivity_tools import run_full_sensitivity_analysis
from src.plot_sensitivity_results import plot_all_metrics, plot_all_comparisons
import plotly.graph_objects as go
import base64

def create_network_visualization(G, height=400):
    """Create an interactive network visualization"""
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", 
                 font_color="black")
    
    # Copy the graph to avoid modifying the original
    G_copy = G.copy()
    
    # Add nodes and edges
    for node in G_copy.nodes():
        net.add_node(node, label=f"Agent {node}")
    for edge in G_copy.edges():
        net.add_edge(edge[0], edge[1])
    
    # Generate the HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height)

def custom_network_creator(num_agents):
    """Create a custom network using an adjacency matrix"""
    st.subheader("Custom Network Creator")
    
    # Initialize or get the adjacency matrix from session state
    if 'adj_matrix' not in st.session_state or \
       st.session_state.adj_matrix.shape[0] != num_agents:
        st.session_state.adj_matrix = np.zeros((num_agents, num_agents))
    
    # Create a DataFrame for the adjacency matrix
    df = pd.DataFrame(
        st.session_state.adj_matrix,
        columns=[f"Agent {i}" for i in range(num_agents)],
        index=[f"Agent {i}" for i in range(num_agents)]
    )
    
    # Display the adjacency matrix as an editable table
    st.write("Adjacency Matrix (1 for connection, 0 for no connection):")
    edited_df = st.data_editor(df)
    
    # Update the adjacency matrix in session state
    st.session_state.adj_matrix = edited_df.values
    
    # Create network from adjacency matrix
    G = nx.from_numpy_array(st.session_state.adj_matrix)
    
    # Visualize the network
    st.write("Network Preview:")
    create_network_visualization(G)
    
    return G if not st.session_state.adj_matrix.size == 0 else None

def main():
    st.set_page_config(page_title="Opinion Dynamics Model", layout="wide")
    st.title("Opinion Dynamics Model Configuration")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        # Network Parameters
        st.subheader("Network Parameters")
        num_agents = st.number_input("Number of Agents", min_value=2, max_value=100, value=10)
        network_type = st.selectbox(
            "Network Type",
            ["cycle", "wheel", "complete", "custom"]
        )
        
        # Theory Parameters
        st.subheader("Theory Parameters")
        old_theory_payoff = st.slider("Old Theory Payoff", 0.0, 1.0, 0.5)
        new_theory_payoff_low = st.slider("New Theory Payoff (Old True)", 0.0, 1.0, 0.4)
        new_theory_payoff_high = st.slider("New Theory Payoff (New True)", 0.0, 1.0, 0.6)
        true_theory = st.selectbox("True Theory", ["old", "new"], index=1)
        
        belief_strength_min = st.slider("Belief Strength (Min)", 0.0, 10.0, 0.5)
        belief_strength_max = st.slider("Belief Strength (Max)", 0.0, 10.0, 2.0)
        
        # Simulation Parameters
        st.subheader("Simulation Parameters")
        num_simulations = st.number_input("Number of Simulations", 1, 10000, 2000)
        use_animation = st.checkbox("Show Animation")
        max_steps = st.number_input("Maximum Steps", 1, 10000, 1000)
        
        # Analysis Options
        st.subheader("Analysis Options")
        run_regular = st.checkbox("Run Regular Simulations", True)
        run_sensitivity = st.checkbox("Run Sensitivity Analysis")
        
        # Only show these options if sensitivity analysis is selected
        if run_sensitivity:
            create_plots = st.checkbox("Create Analysis Plots", True)
            num_trajectories = st.number_input("Number of Trajectories", 1, 5000, 715)
        else:
            create_plots = False
            num_trajectories = 715  

    # Main area
    custom_network = None
    if network_type == "custom":
        custom_network = custom_network_creator(num_agents)
    else:
        # Show default network preview
        G = None
        if network_type == "cycle":
            G = nx.cycle_graph(num_agents)
        elif network_type == "wheel":
            G = nx.wheel_graph(num_agents)
        elif network_type == "complete":
            G = nx.complete_graph(num_agents)
            
        if G:
            st.write("Network Preview:")
            create_network_visualization(G)
    
    # Run button
    if st.button("Run Model"):
        with st.spinner("Running simulation..."):
            params = {
                'num_simulations': num_simulations,
                'num_agents': num_agents,
                'network_type': custom_network if network_type == "custom" else network_type,
                'old_theory_payoff': old_theory_payoff,
                'new_theory_payoffs': (new_theory_payoff_low, new_theory_payoff_high),
                'true_theory': true_theory,
                'belief_strength_range': (belief_strength_min, belief_strength_max),
                'use_animation': use_animation,
                'max_steps': max_steps,
                'animation_params': {
                    'num_frames': 30,
                    'interval': 500,
                    'steps_per_frame': 1
                },
            }
            
            if run_regular:
                st.write("=== Running Regular Simulations ===")
                run_simulations_until_convergence(**params)
            
            if run_sensitivity:
                st.write("=== Running Sensitivity Analysis ===")
                run_full_sensitivity_analysis(
                    num_trajectories=num_trajectories,
                    run_single=True,
                    run_comparison=True
                )
                
            if create_plots:
                st.write("=== Creating Sensitivity Analysis Plots ===")
                for net_type in ['cycle', 'wheel', 'complete']:
                    plot_all_metrics(network_type=net_type, 
                                   num_trajectories=num_trajectories)
                
                plot_all_comparisons(num_trajectories=num_trajectories)
                st.success("All plots have been saved to the analysis_plots directory!")

if __name__ == "__main__":
    main() 