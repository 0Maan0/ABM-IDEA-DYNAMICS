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

from src.scientists import ScientistAgent
from src.super_scientist import SuperScientistAgent
import plotly.graph_objects as go
import plotly.express as px
import base64
from streamlit_elements import elements, mui
import json
from pathlib import Path
import seaborn as sns
from src.zollman_analysis import run_zollman_experiment, plot_zollman_figures, load_results

# Set the plotting style
sns_colors = sns.color_palette("Set2", 8)
# Convert seaborn colors (0-1 range) to hex for Plotly
colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in sns_colors]

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
    
    # Set upper triangle and diagonal to None to make them uneditable
    for i in range(num_agents):
        for j in range(num_agents):
            if i <= j:  # This covers both diagonal and upper triangle
                df.iloc[i, j] = None
    
    # Display the editable adjacency matrix
    st.write("Click cells to connect/disconnect scientists:")
    st.write("Note: The network must be fully connected (no isolated scientists or groups).")
    st.write("Note: No scientist in the network should be self-connected.")
    
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
    adj_matrix = edited_df.fillna(0).values.astype(float)
    
    # Since we only edited the lower triangle, we need to make the matrix symmetric
    adj_matrix = adj_matrix + adj_matrix.T
    
    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Check if the network is connected
    is_connected = nx.is_connected(G)
    
    # Display the network
    st.write("Network Preview:")
    create_network_visualization(G)
    
    return G if is_connected else None

def plot_convergence_analysis_plotly(results_df, network_type: str, num_agents: int):
    """
    Create Plotly visualizations for the simulation results.
    """
    # Plot 1: Distribution of steps to convergence
    fig_steps = px.histogram(
        results_df[results_df['converged']],
        x='step',
        nbins=30,
        title=f'Distribution of Steps to Convergence ({network_type.capitalize()} Network, {num_agents} Agents)'
    )
    fig_steps.update_layout(
        xaxis_title="Steps to Convergence",
        yaxis_title="Count"
    )
    
    # Plot 2: Final consensus distribution
    consensus_counts = results_df['theory'].value_counts()
    colors = ['#2ecc71' if theory == 'new' else '#e74c3c' for theory in consensus_counts.index]
    fig_consensus = px.bar(
        x=consensus_counts.index,
        y=consensus_counts.values,
        title=f'Distribution of Final Consensus ({network_type.capitalize()} Network, {num_agents} Agents)',
        color=consensus_counts.index,
        color_discrete_map={'new': '#2ecc71', 'old': '#e74c3c'}
    )
    fig_consensus.update_layout(
        xaxis_title="Final Consensus Theory",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig_steps, fig_consensus

def main():
    st.set_page_config(page_title="Opinion Dynamics Model", layout="wide")
    st.title("Opinion Dynamics Model Configuration")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        # Analysis Type Selection (moved to top)
        st.subheader("Analysis Type")
        analysis_type = st.radio(
            "Choose Analysis",
            ["Regular Simulations", "Make Animation", "Zollman Analysis"],
            help="Choose the type of analysis to run"
        )
        
        # Add Zollman mode selection right after analysis type if Zollman is selected
        zollman_mode = None
        if analysis_type == "Zollman Analysis":
            zollman_mode = st.radio(
                "Zollman Analysis Mode",
                ["Run new simulations", "Plot existing results"],
                help="Choose whether to run new simulations or plot existing results"
            )
        
        # Network Parameters
        st.subheader("Network Parameters")
        
        # Only show number of scientists if not doing Zollman analysis
        if analysis_type != "Zollman Analysis":
            num_agents = st.number_input("Number of Scientists", min_value=2, max_value=50, value=6)
        else:
            num_agents = 6  # Default value, won't be used
        
        # Only show network type selection if not doing Zollman analysis
        if analysis_type != "Zollman Analysis":
            network_options = ["cycle", "wheel", "complete", "bipartite", "cliques", "custom"]
            network_type = st.selectbox(
                "Network Type",
                network_options,
                index=0
            )
        else:
            network_type = "cycle"  # Default value, won't be used
        
        # Simulation Parameters (moved up)
        st.subheader("Simulation Parameters")
        
        # Set simulation parameters based on analysis type
        if analysis_type == "Make Animation":
            use_animation = True
            num_simulations = 1
        else:
            use_animation = False
            num_simulations = st.number_input(
                "Number of Simulations",
                min_value=1,
                max_value=10000,
                value=2000
            )
        
        # Agent Parameters
        st.subheader("Agent Parameters")
        agent_type = st.selectbox(
            "Agent Type",
            ["ScientistAgent", "SuperScientistAgent"],
            index=0,
            help="Choose between regular Scientists or Super Scientists with enhanced learning capabilities"
        )
        
        if analysis_type != "Zollman Analysis":
            use_noise = st.checkbox(
                "Add Noise",
                value=False,
                help="Enable noise in the agents' observations"
            )
            
            if use_noise:
                noise_value = st.number_input(
                    "Noise Value",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Standard deviation of the Gaussian noise added to observations"
                )
            else:
                noise_value = 0.0
            
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
        else:
            # Set default values for Zollman analysis
            use_noise = False
            noise_value = 0.0
            old_theory_payoff = 0.5
            new_theory_payoff_low = 0.4
            new_theory_payoff_high = 0.6
            true_theory = "new"
            belief_strength_low = 0.5
            belief_strength_high = 2.0

    # Main area
    custom_network = None
    if network_type == "custom":
        custom_network = custom_network_creator(num_agents)
        if custom_network is None:
            st.error("Cannot run simulation with a disconnected network. Please modify the network to ensure all scientists are connected.")
            st.stop()
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
                'network_type': network_type,
                'old_theory_payoff': old_theory_payoff,
                'new_theory_payoffs': (new_theory_payoff_low, new_theory_payoff_high),
                'true_theory': true_theory,
                'belief_strength_range': (belief_strength_low, belief_strength_high),
                'use_animation': use_animation,
                'animation_params': {
                    'num_frames': 30,
                    'interval': 500,
                    'steps_per_frame': 1
                },
                'noise': "on" if use_noise else "off",
                'noise_std': noise_value if use_noise else 0.0
            }
            
            if analysis_type == "Regular Simulations":
                # Handle custom network
                if network_type == "custom" and custom_network is not None:
                    params['custom_graph'] = custom_network
                
                # Run simulations
                results = run_simulations_until_convergence(
                    agent_class=ScientistAgent if agent_type == "ScientistAgent" else SuperScientistAgent,
                    **params
                )

                # Load and analyze results
                try:
                    # Convert results to DataFrame if it's a list
                    results_df = pd.DataFrame(results) if isinstance(results, list) else results
                    
                    # Calculate statistics
                    total_runs = len(results_df)
                    converged_mask = results_df['converged'] == True
                    converged_runs = converged_mask.sum()
                    convergence_rate = converged_runs / total_runs * 100
                    
                    correct_theory_mask = results_df['theory'] == 'Correct Theory'
                    correct_theory_runs = correct_theory_mask.sum()
                    correct_theory_rate = correct_theory_runs / total_runs * 100
                    
                    converged_steps = results_df[converged_mask]['step']
                    avg_steps = converged_steps.mean() if len(converged_steps) > 0 else 0
                    
                    # Display statistics in a nice format
                    st.subheader("Simulation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Runs", total_runs)
                        st.metric("Converged Runs", f"{converged_runs} ({convergence_rate:.1f}%)")
                    
                    with col2:
                        st.metric("Correct Theory Runs", f"{correct_theory_runs} ({correct_theory_rate:.1f}%)")
                        st.metric("Average Steps to Convergence", f"{avg_steps:.1f}")
                    
                    # Create and display plots
                    st.subheader("Analysis Plots")
                    
                    # Plot 1: Steps to convergence
                    fig_steps = px.histogram(
                        results_df[converged_mask],
                        x='step',
                        nbins=30,
                        title=f'Distribution of Steps to Convergence<br>({network_type.capitalize()} Network, {num_agents} Agents)',
                        color_discrete_sequence=[colors[2]]  # Use third color from Set2 (green)
                    )
                    fig_steps.update_layout(
                        xaxis_title="Steps to Convergence",
                        yaxis_title="Count",
                        title={
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        margin=dict(t=80),  # Add more margin at the top for the title
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial")
                    )
                    
                    # Plot 2: Theory distribution
                    theory_counts = results_df['theory'].fillna('Not Converged').value_counts()
                    colors_dict = {
                        'Correct Theory': colors[2],    # Green
                        'Old Theory': colors[1],        # Red/Orange
                        'Incorrect Theory': colors[1],  # Same as Old Theory
                        'Not Converged': colors[7]      # Gray
                    }
                    
                    fig_consensus = px.bar(
                        x=theory_counts.index,
                        y=theory_counts.values,
                        title=f'Distribution of Final Consensus<br>({network_type.capitalize()} Network, {num_agents} Agents)',
                        color=theory_counts.index,
                        color_discrete_map=colors_dict
                    )
                    fig_consensus.update_layout(
                        xaxis_title="Final Consensus Theory",
                        yaxis_title="Count",
                        showlegend=False,
                        title={
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        margin=dict(t=80),  # Add more margin at the top for the title
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial")
                    )
                    
                    # Display plots
                    col3, col4 = st.columns(2)
                    with col3:
                        st.plotly_chart(fig_steps, use_container_width=True)
                    with col4:
                        st.plotly_chart(fig_consensus, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred while analyzing results: {str(e)}")
            
            elif analysis_type == "Make Animation":
                # Run single simulation with animation
                params['use_animation'] = True
                params['num_simulations'] = 1
                
                # Run simulation and get video path
                video_path = run_simulations_until_convergence(
                    agent_class=ScientistAgent if agent_type == "ScientistAgent" else SuperScientistAgent,
                    **params
                )
                
                # Display the generated video in a smaller size
                if os.path.exists(video_path):
                    st.subheader("Simulation Animation")
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        st.video(video_path)
                else:
                    st.error("Animation file was not generated successfully.")
            
            elif analysis_type == "Zollman Analysis":
                st.write("=== Running Zollman Analysis ===")
                
                # Check if results already exist
                agent_class = ScientistAgent if agent_type == "ScientistAgent" else SuperScientistAgent
                
                if zollman_mode == "Run new simulations":
                    print(f"Running Zollman experiment")
                    results = run_zollman_experiment(
                        agent_class=agent_class,
                        num_simulations=num_simulations,
                        network_sizes=[4, 6, 8, 10, 12]
                    )
                else:
                    # Try to load existing results
                    results = load_results(agent_class=agent_class, num_simulations=num_simulations)
                    if results is None:
                        st.error(f"No existing results found for {agent_class.__name__} with {num_simulations} simulations. Please run new simulations first.")
                        st.stop()
                    print("Loading existing results")
                
                # Create and display plots side by side
                st.write("=== Zollman Analysis Plots ===")
                learning_fig, speed_fig = plot_zollman_figures(
                    num_simulations, 
                    agent_class=agent_class
                )
                
                # Display plots side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Learning Results:")
                    st.pyplot(learning_fig, use_container_width=True)
                with col2:
                    st.write("Speed Results:")
                    st.pyplot(speed_fig, use_container_width=True)

if __name__ == "__main__":
    main() 