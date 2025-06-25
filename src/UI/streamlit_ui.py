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
from streamlit_elements import elements, mui
import json

def create_network_visualization(G, height=400):
    """Create an interactive network visualization"""
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", 
                 font_color="black")
    
    # Copy the graph to avoid modifying the original
    G_copy = G.copy()
    
    # Add nodes and edges
    for node in G_copy.nodes():
        net.add_node(node, label=f"Scientist {node+1}")
    for edge in G_copy.edges():
        net.add_edge(edge[0], edge[1])
    
    # Generate the HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height)

def custom_network_creator(num_agents):
    """Create a custom network using adjacency matrix"""
    st.subheader("Custom Network Creator")
    
    # Create an empty adjacency matrix with boolean values
    adj_matrix = np.zeros((num_agents, num_agents), dtype=bool)
    
    # Create a DataFrame for the adjacency matrix with labeled rows and columns
    df = pd.DataFrame(
        adj_matrix,
        columns=[f"Scientist {i+1}" for i in range(num_agents)],
        index=[f"Scientist {i+1}" for i in range(num_agents)]
    )
    
    # Create a mask for diagonal elements
    for i in range(num_agents):
        df.iloc[i, i] = None  # Set diagonal to None to make it uneditable
    
    # Display the editable adjacency matrix
    st.write("Click cells to connect/disconnect scientists:")
    edited_df = st.data_editor(
        df,
        disabled=["index"],  # Disable editing of row labels
        column_config={
            col: st.column_config.CheckboxColumn(col, default=False)
            for col in df.columns
        },
        use_container_width=True
    )
    
    # Convert back to numpy array and to float for NetworkX
    # Replace any None values with 0 before conversion
    adj_matrix = edited_df.fillna(0).values.astype(float)
    
    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Display the network
    st.write("Network Preview:")
    create_network_visualization(G)
    
    return G

def main():
    st.set_page_config(page_title="Opinion Dynamics Model", layout="wide")
    st.title("Opinion Dynamics Model Configuration")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        # Network Parameters
        st.subheader("Network Parameters")
        num_agents = st.number_input("Number of Scientists", min_value=2, max_value=100, value=10)
        network_type = st.selectbox(
            "Network Type",
            ["cycle", "wheel", "complete", "custom"],
            index=0
        )
        
        # Theory Parameters
        st.subheader("Theory Parameters")
        old_theory_payoff = st.number_input(
            "Old Theory Payoff",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        new_theory_payoff_low = st.number_input(
            "New Theory Payoff (Old True)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1
        )
        new_theory_payoff_high = st.number_input(
            "New Theory Payoff (New True)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        true_theory = st.selectbox(
            "True Theory",
            ["old", "new"],
            index=1
        )
        
        belief_strength_low = st.number_input(
            "Belief Strength (Min)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1
        )
        belief_strength_high = st.number_input(
            "Belief Strength (Max)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
        
        # Simulation Parameters
        st.subheader("Simulation Parameters")
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=1,
            max_value=10000,
            value=2000
        )
        
        show_final_state = st.checkbox("Show Final State")
        use_animation = st.checkbox("Use Animation")
        
        max_steps = st.number_input(
            "Maximum Steps",
            min_value=1,
            max_value=10000,
            value=1000
        )
        
        # Analysis Options
        st.subheader("Analysis Options")
        run_regular = st.checkbox("Run Regular Simulations", value=True)
        run_sensitivity = st.checkbox("Run Sensitivity Analysis")
        
        if run_sensitivity:
            create_plots = st.checkbox("Create Plots", value=True)
            num_trajectories = st.number_input(
                "Number of Trajectories",
                min_value=1,
                max_value=5000,
                value=715
            )

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
                'belief_strength_range': (belief_strength_low, belief_strength_high),
                'use_animation': use_animation,
                'max_steps': max_steps,
                'animation_params': {
                    'num_frames': 30,
                    'interval': 500,
                    'steps_per_frame': 1
                },
                'show_final_state': show_final_state,
                'custom_network': custom_network
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