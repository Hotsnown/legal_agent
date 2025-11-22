import uuid
import math
import copy
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

from system_state import SystemState, Task
# On importe le moteur d'inférence pour faire tourner les simulations (What-If)
from reasoning_engine import ReasoningEngine, PyroLegalModel

# ==============================================================================
# CONFIGURATION DES COÛTS (COST MODEL)
# ==============================================================================

class InteractionCost(Enum):
    """
    Modélise l'effort cognitif ou temporel pour l'utilisateur.
    Score = EIG / Cost.
    """
    LOW = 1.0    # Question Oui/Non, info de tête
    MEDIUM = 3.0 # Besoin de vérifier un email ou une date approximative
    HIGH = 10.0  # Besoin de retrouver un document, comptabilité, scan

class AcquisitionCandidate(object):
    """Une variable latente sur laquelle on peut poser une question."""
    def __init__(self, var_name: str, description: str, cost: InteractionCost, variable_key: str):
        self.var_name = var_name
        self.description = description
        self.cost = cost
        self.variable_key = variable_key # La clé dans le dictionnaire de contexte Pyro

# Liste des questions possibles (Knowledge Base d'acquisition)
CANDIDATES_POOL = [
    AcquisitionCandidate(
        "check_fault", 
        "Y a-t-il eu des reproches écrits avant la rupture ?", 
        InteractionCost.LOW, 
        "p_grave_fault"
    ),
    AcquisitionCandidate(
        "check_dependency_share", 
        "Quelle part (en %) de votre CA ce client représentait-il ?", 
        InteractionCost.MEDIUM, 
        "dep_alpha" # Influence la distribution Beta
    ),
    AcquisitionCandidate(
        "check_exclusivity", 
        "Aviez-vous une clause d'exclusivité ?", 
        InteractionCost.LOW, 
        "dep_beta" # Influence aussi la dépendance
    ),
    AcquisitionCandidate(
        "check_written_contract", 
        "Existe-t-il un contrat cadre signé ?", 
        InteractionCost.HIGH, # Chercher le PDF est coûteux
        "p_established"
    ),
    AcquisitionCandidate(
        "check_notice_duration",
        "Quel preavis en mois a ete effectivement accorde ?",
        InteractionCost.MEDIUM,
        "actual_notice_months"
    )

]

# ==============================================================================
# MOTEUR D'ACQUISITION (BOED IMPLEMENTATION)
# ==============================================================================

class AcquisitionEngine:
    def __init__(self, reasoning_engine: ReasoningEngine):
        self.reasoner = reasoning_engine
        self.model = PyroLegalModel() # Accès direct au modèle mathématique
        
        # Seuils d'arrêt
        self.ENTROPY_THRESHOLD = 0.4  # Si H < 0.4, on arrête, le verdict est clair
        self.MIN_EIG_GAIN = 0.01      # Si aucune question n'apporte plus de 0.01 bit d'info

    def optimize_next_step(self, state: SystemState) -> Optional[Task]:
        """
        Cerveau principal : 
        1. Calcule l'entropie actuelle.
        2. Si trop haute, simule toutes les questions possibles.
        3. Retourne la tâche de poser la meilleure question.
        """
        print(f"--- [ACQUISITION] Analyzing Uncertainty (Current Entropy: {state.entropy:.4f}) ---")
        
        # 1. Check Conditions d'Arrêt
        if state.entropy < self.ENTROPY_THRESHOLD:
            print("✅ Entropy is low enough. No further questions needed.")
            return None

        # 2. Récupération du Contexte Actuel (Priors)
        current_context = self.reasoner._extract_priors_from_graph(state)
        
        best_candidate = None
        best_score = -1.0
        
        # 3. Boucle d'Optimisation (EIG Calculation)
        print(f"   -> Evaluating {len(CANDIDATES_POOL)} candidates for Information Gain...")
        
        for candidate in CANDIDATES_POOL:
            # On ignore les questions déjà résolues (simulé ici par une vérif simple)
            # Dans la réalité, on vérifierait si le fait existe déjà dans le Graphe
            
            eig = self._calculate_eig(candidate, current_context)
            roi_score = eig / candidate.cost.value
            
            print(f"      - [{candidate.var_name}] EIG: {eig:.4f} / Cost: {candidate.cost.value} = Score: {roi_score:.4f}")
            
            if roi_score > best_score:
                best_score = roi_score
                best_candidate = candidate

        # 4. Décision
        if best_score < self.MIN_EIG_GAIN:
            print("🛑 Diminishing returns: No question provides enough info.")
            return None
            
        print(f"🚀 BEST ACTION SELECTED: {best_candidate.description} (Score: {best_score:.4f})")
        
        # Création de la tâche d'interaction
        return Task(
            action="ASK_USER",
            target_node_id=best_candidate.var_name,
            priority=100, # Urgent
            status="PENDING",
            # On stocke la question générée pour l'UI
            dependencies=[f"QUESTION: {best_candidate.description}"] 
        )

    def _calculate_eig(self, candidate: AcquisitionCandidate, base_context: Dict[str, Any]) -> float:
        """
        Calcule l'Expected Information Gain par simulation de Monte Carlo.
        EIG = H_current - E[H_posterior]
        """
        # 1. H_current (déjà connu ou recalculé)
        h_current = self._simulate_entropy(base_context)
        
        # 2. Simulation des Mondes (Hypothèse binaire simplifiée pour la démo)
        # On imagine que l'utilisateur répond OUI (High impact on variable) ou NON (Low impact)
        
        # Scenario A : Réponse OUI (ex: "Oui, il y a une faute grave")
        # On force la probabilité de la variable à 0.99 dans le contexte
        ctx_yes = base_context.copy()
        self._apply_simulation_override(ctx_yes, candidate.variable_key, is_positive_answer=True)
        h_yes = self._simulate_entropy(ctx_yes)
        
        # Scenario B : Réponse NON (ex: "Non, pas de faute")
        ctx_no = base_context.copy()
        self._apply_simulation_override(ctx_no, candidate.variable_key, is_positive_answer=False)
        h_no = self._simulate_entropy(ctx_no)
        
        # 3. Calcul de l'Espérance
        # P(Yes) est la probabilité a priori que la réponse soit oui (selon le modèle actuel)
        # Pour simplifier la démo sans moteur Pyro complet, on suppose P(Yes)=0.5 si incertain
        p_yes = base_context.get(candidate.variable_key, 0.5) 
        if isinstance(p_yes, (int, float)):
             pass # OK
        else:
             p_yes = 0.5 # Fallback pour les distributions complexes

        expected_posterior_entropy = (p_yes * h_yes) + ((1 - p_yes) * h_no)
        
        # EIG
        eig = h_current - expected_posterior_entropy
        return max(eig, 0.0) # Jamais négatif en théorie

    def _apply_simulation_override(self, context: Dict, key: str, is_positive_answer: bool):
        """Modifie le dictionnaire de contexte pour simuler une reponse."""
        if key == "p_grave_fault":
            context[key] = 0.99 if is_positive_answer else 0.01
        elif key == "p_established":
            context[key] = 0.99 if is_positive_answer else 0.01
        elif key == "dep_alpha":
            context["dep_alpha"] = 10.0 if is_positive_answer else 2.0
            context["dep_beta"] = 2.0 if is_positive_answer else 5.0
        elif key == "dep_beta":
            context["dep_alpha"] = 5.0
            context["dep_beta"] = 1.0 if is_positive_answer else 5.0
        elif key == "actual_notice_months":
            context[key] = 6.0 if is_positive_answer else 0.5

    def _simulate_entropy(self, context: Dict) -> float:
        """Appelle le moteur d'inference (Module 3) pour obtenir l'entropie."""
        try:
            return self.reasoner._heuristic_entropy(context)
        except Exception:
            p_fault = context.get("p_grave_fault", 0.1)
            p_est = context.get("p_established", 0.9)
            dep_alpha = context.get("dep_alpha", 2.0)
            dep_beta = context.get("dep_beta", 5.0)
            dep_mean = dep_alpha / (dep_alpha + dep_beta)
            uncertainty = 0.0
            uncertainty += 4 * (p_fault * (1 - p_fault))
            uncertainty += 4 * (p_est * (1 - p_est))
            uncertainty += 2 * (dep_mean * (1 - dep_mean))
            return min(uncertainty, 1.0)

