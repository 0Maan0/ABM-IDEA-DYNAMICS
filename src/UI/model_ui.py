import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt
from ..run_model_utils import run_simulations_until_convergence
from ..run_sensitivity_tools import run_full_sensitivity_analysis
from ..plot_sensitivity_results import plot_all_metrics, plot_all_comparisons

class ModelConfigUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Opinion Dynamics Model Configuration")
        self.setMinimumWidth(600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Network Parameters Group
        network_group = QGroupBox("Network Parameters")
        network_layout = QFormLayout()
        
        # Number of agents
        self.num_agents_spin = QSpinBox()
        self.num_agents_spin.setRange(2, 100)
        self.num_agents_spin.setValue(10)
        network_layout.addRow("Number of Agents:", self.num_agents_spin)
        
        # Network type
        self.network_type_combo = QComboBox()
        self.network_type_combo.addItems(['cycle', 'wheel', 'complete'])
        network_layout.addRow("Network Type:", self.network_type_combo)
        
        network_group.setLayout(network_layout)
        main_layout.addWidget(network_group)
        
        # Theory Parameters Group
        theory_group = QGroupBox("Theory Parameters")
        theory_layout = QFormLayout()
        
        # Old theory payoff
        self.old_theory_payoff_spin = QDoubleSpinBox()
        self.old_theory_payoff_spin.setRange(0, 1)
        self.old_theory_payoff_spin.setSingleStep(0.1)
        self.old_theory_payoff_spin.setValue(0.5)
        theory_layout.addRow("Old Theory Payoff:", self.old_theory_payoff_spin)
        
        # New theory payoffs
        self.new_theory_payoff_low = QDoubleSpinBox()
        self.new_theory_payoff_low.setRange(0, 1)
        self.new_theory_payoff_low.setSingleStep(0.1)
        self.new_theory_payoff_low.setValue(0.4)
        theory_layout.addRow("New Theory Payoff (Old True):", self.new_theory_payoff_low)
        
        self.new_theory_payoff_high = QDoubleSpinBox()
        self.new_theory_payoff_high.setRange(0, 1)
        self.new_theory_payoff_high.setSingleStep(0.1)
        self.new_theory_payoff_high.setValue(0.6)
        theory_layout.addRow("New Theory Payoff (New True):", self.new_theory_payoff_high)
        
        # True theory
        self.true_theory_combo = QComboBox()
        self.true_theory_combo.addItems(['old', 'new'])
        self.true_theory_combo.setCurrentText('new')
        theory_layout.addRow("True Theory:", self.true_theory_combo)
        
        # Belief strength range
        self.belief_strength_low = QDoubleSpinBox()
        self.belief_strength_low.setRange(0, 10)
        self.belief_strength_low.setSingleStep(0.1)
        self.belief_strength_low.setValue(0.5)
        theory_layout.addRow("Belief Strength (Min):", self.belief_strength_low)
        
        self.belief_strength_high = QDoubleSpinBox()
        self.belief_strength_high.setRange(0, 10)
        self.belief_strength_high.setSingleStep(0.1)
        self.belief_strength_high.setValue(2.0)
        theory_layout.addRow("Belief Strength (Max):", self.belief_strength_high)
        
        theory_group.setLayout(theory_layout)
        main_layout.addWidget(theory_group)
        
        # Simulation Parameters Group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout()
        
        # Number of simulations
        self.num_sims_spin = QSpinBox()
        self.num_sims_spin.setRange(1, 10000)
        self.num_sims_spin.setValue(2000)
        sim_layout.addRow("Number of Simulations:", self.num_sims_spin)
        
        # Show final state
        self.show_final_state = QCheckBox()
        sim_layout.addRow("Show Final State:", self.show_final_state)
        
        # Animation parameters
        self.use_animation = QCheckBox()
        sim_layout.addRow("Use Animation:", self.use_animation)
        
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 10000)
        self.max_steps_spin.setValue(1000)
        sim_layout.addRow("Maximum Steps:", self.max_steps_spin)
        
        sim_group.setLayout(sim_layout)
        main_layout.addWidget(sim_group)
        
        # Analysis Options Group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout()
        
        self.run_regular = QCheckBox()
        self.run_regular.setChecked(True)
        analysis_layout.addRow("Run Regular Simulations:", self.run_regular)
        
        self.run_sensitivity = QCheckBox()
        analysis_layout.addRow("Run Sensitivity Analysis:", self.run_sensitivity)
        
        self.create_plots = QCheckBox()
        self.create_plots.setChecked(True)
        analysis_layout.addRow("Create Plots:", self.create_plots)
        
        self.num_trajectories = QSpinBox()
        self.num_trajectories.setRange(1, 10000)
        self.num_trajectories.setValue(715)
        analysis_layout.addRow("Number of Trajectories:", self.num_trajectories)
        
        analysis_group.setLayout(analysis_layout)
        main_layout.addWidget(analysis_group)
        
        # Run button
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)
        main_layout.addWidget(self.run_button)
        
    def get_parameters(self):
        """Collect all parameters from the UI"""
        return {
            # Network parameters
            'num_agents': self.num_agents_spin.value(),
            'network_type': self.network_type_combo.currentText(),
            
            # Theory parameters
            'old_theory_payoff': self.old_theory_payoff_spin.value(),
            'new_theory_payoffs': (self.new_theory_payoff_low.value(), 
                                 self.new_theory_payoff_high.value()),
            'true_theory': self.true_theory_combo.currentText(),
            'belief_strength_range': (self.belief_strength_low.value(),
                                    self.belief_strength_high.value()),
            
            # Simulation parameters
            'num_simulations': self.num_sims_spin.value(),
            'show_final_state': self.show_final_state.isChecked(),
            'use_animation': self.use_animation.isChecked(),
            'max_steps': self.max_steps_spin.value(),
            'animation_params': {
                'num_frames': 30,
                'interval': 500,
                'steps_per_frame': 1
            },
            
            # Analysis options
            'run_regular_simulations': self.run_regular.isChecked(),
            'run_sensitivity': self.run_sensitivity.isChecked(),
            'create_plots': self.create_plots.isChecked(),
            'num_trajectories': self.num_trajectories.value()
        }
    
    def run_model(self):
        """Get parameters and run the model"""
        params = self.get_parameters()
        
        if params['run_regular_simulations']:
            print("\n=== Running Regular Simulations ===")
            run_simulations_until_convergence(
                num_simulations=params['num_simulations'],
                num_agents=params['num_agents'],
                network_type=params['network_type'],
                old_theory_payoff=params['old_theory_payoff'],
                new_theory_payoffs=params['new_theory_payoffs'],
                true_theory=params['true_theory'],
                belief_strength_range=params['belief_strength_range'],
                use_animation=params['use_animation'],
                max_steps=params['max_steps'],
                animation_params=params['animation_params'],
                show_final_state=params['show_final_state']
            )
        
        if params['run_sensitivity']:
            print("\n=== Running Sensitivity Analysis ===")
            run_full_sensitivity_analysis(
                num_trajectories=params['num_trajectories'],
                run_single=True,
                run_comparison=True
            )
            
        if params['create_plots']:
            print("\n=== Creating Sensitivity Analysis Plots ===")
            for net_type in ['cycle', 'wheel', 'complete']:
                plot_all_metrics(network_type=net_type, 
                               num_trajectories=params['num_trajectories'])
            
            plot_all_comparisons(num_trajectories=params['num_trajectories'])
            print("All plots have been saved to the analysis_plots directory!")

def main():
    app = QApplication(sys.argv)
    window = ModelConfigUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
