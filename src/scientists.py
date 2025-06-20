from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import networkx as nx
import random
import numpy as np

class ScientistAgent(Agent):
    """
    An agent representing a scientist in Zollman's (2007) model 
    (https://www.researchgate.net/publication/228973111_The_Communication_Structure_of_Epistemic_Communities).
    So every scientist has a belief about which theory (old or new) is true. Each step they choose which theory 
    they want to believe/work with based on expected utility. Then their beieves are updated,
    based on experimental results.
    """
    def __init__(self, unique_id, model, initial_belief=0.5, belief_strength=1.0):
        super().__init__(unique_id, model)
        
        self.belief_in_new_theory = initial_belief  # Belief that the new theory is true
        self.belief_strength = belief_strength # Belief strength parameter ==> if this value is higher, the scientist is more resistant to change
    
        self.current_choice = self._choose_theory() # The current theory choice ("old" or "new")
        
        # Experimental results when working with old or new theory
        self.old_theory_results = [] 
        self.new_theory_results = []  

    def _choose_theory(self):
        """To choose between the theories ==> which theory to believe/work
            ==> its based on expected utility (remember game theory :) )"""
        # Always the same payoff for the old theory
        old_theory_utility = self.model.old_theory_payoff
        
        # New theory utility looks at which theory is true 
        new_theory_utility = (
            self.belief_in_new_theory * self.model.new_theory_payoffs[1] +  # Payoff if new theory is true
            (1 - self.belief_in_new_theory) * self.model.new_theory_payoffs[0]  # Payoff if new theory is false
        )
        
        return "new" if new_theory_utility > old_theory_utility else "old"

    def update_belief(self, result, theory_choice):
        """
        Paper mentioned bayesian update based on experimental result: so try this:
        P(New Theory|Result) = P(Result|New Theory) * P(New Theory) / P(Result)
        """
        if theory_choice == "old":
            # if you work with the old theory, you don't update your belief
            return
            
        # Current belief in new theory
        prior = self.belief_in_new_theory
        
        # Likelihood of result under each theory ==> see paper for example
        p_result_if_new = self._likelihood(result, self.model.new_theory_payoffs[1])
        p_result_if_old = self._likelihood(result, self.model.new_theory_payoffs[0])
        
        # Bayes update
        bayes_update = (p_result_if_new * prior) / (p_result_if_new * prior + p_result_if_old * (1 - prior))
        
        self.belief_in_new_theory = bayes_update

    def _likelihood(self, result, mean):
        """Calculate likelihood of result given the mean 
         this is to determine: P(Result|Theory): """
        # Assuming a normal distribution for the likelihood
        return np.exp(-0.5 * ((result - mean) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))

    def incorporate_neighbor_evidence(self, neighbor):
        """Learn from neighbor's experimental results"""
        # Only learn from new theory results 
        for result in neighbor.new_theory_results:
            self.update_belief(result, "new")
            
    def step(self):
        # Get payoff from working with chosen theory
        true_mean = self.model.get_action_payoff(self.current_choice)
        result = random.gauss(true_mean, 0.1) # tbh i dont quite understand
        
        if self.current_choice == "old":
            self.old_theory_results.append(result)
        else:
            self.new_theory_results.append(result)
        
        self.update_belief(result, self.current_choice)
        
        # Learn from neighbors experimental results aka evidence
        neighbors = self.model.network.neighbors(self.unique_id)
        for n_id in neighbors:
            neighbor = self.model.schedule.agents[n_id]
            self.incorporate_neighbor_evidence(neighbor)
        
        # Choose theory for next step
        self.current_choice = self._choose_theory()
