import os
import uuid
import time
from enum import Enum, auto
from typing import Optional, Dict, Any

# Importation des Modules précédents
from system_state import SystemState, Task, TaskStatus, Rule, RuleType
from regex_engine import RegexPerceptionEngine # Module 2 (Version Mock)
from reasoning_engine import ReasoningEngine   # Module 3
from active_learning_engine import AcquisitionEngine # Module 4
from domain_config import get_static_rules, get_domain_version

# ==============================================================================
# DÉFINITIONS DES ÉTATS ET SIGNAUX
# ==============================================================================

class WorkflowState(Enum):
    INIT = auto()        # Chargement
    PLANNING = auto()    # LLM décide quoi faire
    EXECUTING = auto()   # Perception ou Calcul
    EVALUATING = auto()  # Check Entropie & Stop Conditions
    INTERACTING = auto() # Attente Humaine
    VERDICT = auto()     # Fin

class OrchestratorSignal(Enum):
    CONTINUE = auto() # Passer à l'étape suivante immédiate
    WAIT = auto()     # Suspendre (ex: attente user)
    STOP = auto()     # Fin du process

# ==============================================================================
# L'ORCHESTRATEUR (THE BOSS)
# ==============================================================================

class LegalOrchestrator:
    """
    Le Système Nerveux Central.
    Gère les transitions d'états, la persistance et les appels aux sous-modules.
    """
    
    def __init__(self, case_id: str, data_dir: str = "data"):
        self.case_id = case_id
        self.state_machine_status = WorkflowState.INIT
        self.data_dir = data_dir
        # --- Initialisation des Organes ---
        self.system_state = SystemState(case_id=case_id)
        self.perception = RegexPerceptionEngine()
        self.reasoner = ReasoningEngine()
        self.acquirer = AcquisitionEngine(self.reasoner)
        self._data_loaded = False
        self.system_state.domain_version = get_domain_version()
        self._hydrate_static_rules()
        # --- Securite (Watchdogs) ---
        self.step_counter = 0
        self.MAX_STEPS = 20
        self.history_hashes = set()
        self._needs_replan = False
        self._replan_reason: Optional[str] = None
        self.replan_attempts = 0

    def run_step(self, user_input: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Exécute UN pas de la machine à états.
        Appelé en boucle ou via API endpoint.
        """
        self.step_counter += 1
        print(f"\n[STEP {self.step_counter}] Current State: {self.state_machine_status.name}")
        
        # 1. Gas Limit Check
        if self.step_counter > self.MAX_STEPS:
            print(" MAX BUDGET EXCEEDED. Forcing Verdict.")
            self.state_machine_status = WorkflowState.VERDICT

        # 2. State Machine Logic
        current = self.state_machine_status
        
        if current == WorkflowState.INIT:
            return self._handle_init()
            
        elif current == WorkflowState.PLANNING:
            return self._handle_planning(user_input)
            
        elif current == WorkflowState.EXECUTING:
            return self._handle_executing()
            
        elif current == WorkflowState.EVALUATING:
            return self._handle_evaluating()
            
        elif current == WorkflowState.INTERACTING:
            # Si on est ici, c'est qu'on attendait un input.
            # Si user_input est None, on reste en attente.
            if user_input:
                print(f" User Input Received: {user_input}")
                self._needs_replan = True
                self._replan_reason = "Reponse utilisateur"
                self.reasoner.request_replan("Signal utilisateur")
                # On injecte la réponse dans le graphe (Simulation simple ici)
                # Dans le vrai système : PerceptionEngine.parse_user_reply(user_input)
                # Ici on force juste une re-planification
                return self._transition(WorkflowState.PLANNING, msg="Input processed, replanning.")
            else:
                return {"status": "WAITING_FOR_USER", "question": self._get_pending_question()}

        elif current == WorkflowState.VERDICT:
            return self._handle_verdict()

        return {"error": "Unknown State"}

    # --- HANDLERS D'ÉTATS ---

    def _handle_init(self):
        """Reset et Preparation."""
        print("   -> Loading Ontology & Context...")
        if not self._data_loaded and os.path.isdir(self.data_dir):
            print(f"   -> Ingesting dataset from {self.data_dir}...")
            self.perception.ingest_directory(self.data_dir, self.system_state)
            self._data_loaded = True
        return self._transition(WorkflowState.PLANNING, msg="Initialized.")

    def _handle_planning(self, user_context=None):
        """Appel au Module 3.1 (Planner)."""
        # Replanification prioritaire si demandée par l'inférence ou l'utilisateur
        replan_reason = self.reasoner.consume_replan_request()
        if self._needs_replan and not replan_reason:
            replan_reason = self._replan_reason or "Replanification demandee"
        if replan_reason:
            self.replan_attempts += 1
            self._needs_replan = False
            self._replan_reason = None
            self.reasoner.replan_strategy(self.system_state, reason=replan_reason)
            self.system_state.replan_history.append(replan_reason)
            return self._transition(WorkflowState.EXECUTING, msg=f"Replanification: {replan_reason}")

        # Si la queue est vide, on demande au Planner de la remplir
        if not self.system_state.plan_queue:
            self.reasoner.plan_strategy(self.system_state)
        else:
            print("   -> Plan already active, skipping generation.")
            
        # Si après appel au planner, c'est toujours vide -> Verdict (plus rien à faire)
        if not self.system_state.plan_queue:
             return self._transition(WorkflowState.EVALUATING, msg="Plan empty.")
             
        return self._transition(WorkflowState.EXECUTING, msg="Plan ready.")

    def _handle_executing(self):
        """Dépile une tâche et l'exécute via le bon Worker."""
        if not self.system_state.plan_queue:
            return self._transition(WorkflowState.EVALUATING, msg="Queue drained.")

        current_task = self.system_state.plan_queue.pop(0) # FIFO
        print(f"    Executing Task: {current_task.action}")
        
        # Routing des tâches vers les modules
        if current_task.action == "CHECK_LIMITATION_PERIOD":
            # Simulation : Appel à une règle 'Hard'
            print("      -> Checking dates... (Simulated OK)")
            
        elif current_task.action == "QUALIFY_RELATIONSHIP":
            if not self._data_loaded and os.path.isdir(self.data_dir):
                print("      -> Loading evidence from data directory.")
                self.perception.ingest_directory(self.data_dir, self.system_state)
                self._data_loaded = True
            else:
                print("      -> Relationship evidence already ingested.")

        elif current_task.action == "COMPUTE_NOTICE_GAP":
            # Appel au Module 3.2 (Pyro Inference)
            # L'inférence est faite dans EVALUATING, ici on prépare juste les données si besoin
            pass

        elif current_task.action == "GENERATE_INTERPRETATION":
            added = self.reasoner.generate_interpretations(self.system_state)
            print(f"      -> Interpretations generated: {len(added)}")
            return self._transition(WorkflowState.EVALUATING, msg=f"Interpretation(s): {', '.join(added) if added else 'aucune'}")

        elif current_task.action == "APPLY_ALEXY_BALANCING":
            notice = self.reasoner.apply_alexy_balancing(self.system_state)
            print(f"      -> Alexy notice computed at {notice:.2f} months")
            return self._transition(WorkflowState.EVALUATING, msg=f"ALEXY notice calcule ({notice:.1f} mois)")
            
        elif current_task.action == "ASK_USER":
            # Tâche spéciale générée par l'Active Learning
            # On la remet en 'pending' ou on la traite ? 
            # En fait, l'Active Learning génère cette tâche pour forcer l'état INTERACTING
            # On stocke la question quelque part
            self.pending_question = current_task.dependencies[0] if current_task.dependencies else "Précisez ?"
            return self._transition(WorkflowState.INTERACTING, msg="Need user info.")

        elif current_task.action == "REPLAN_CONTEXT":
            self._needs_replan = True
            self._replan_reason = "Tache de replanification"
            return self._transition(WorkflowState.PLANNING, msg="Replanification forcee depuis le plan.")

        return self._transition(WorkflowState.EVALUATING, msg="Task done.")

    def _handle_evaluating(self):
        """Le Juge (Module 3.2 + 4)."""
        # 1. Lancer l'Inférence Pyro (Module 3)
        current_entropy = self.reasoner.run_inference(self.system_state)
        
        # 2. Check Convergence
        if current_entropy < 0.4: # Seuil arbitraire
            print("   [OK] Entropy Low enough. Verdict reached.")
            return self._transition(WorkflowState.VERDICT, msg="Confident.")
            
        # 3. Si entropie haute, a-t-on encore des tâches prévues ?
        if self.system_state.plan_queue:
            print("   -> Uncertainty remains, but tasks pending. Continue execution.")
            return self._transition(WorkflowState.EXECUTING, msg="Next task.")
            
        # 4. Si plus de tâches et entropie haute -> Replan + ACTIVE LEARNING (Module 4)
        if current_entropy > 0.55 and self.replan_attempts < 2:
            self._needs_replan = True
            self._replan_reason = "Entropie elevee apres plan"
            self.reasoner.request_replan("Entropie elevee, replanification")
            return self._transition(WorkflowState.PLANNING, msg="Replanification automatique (incertitude).")

        print("   -> Plan finished but uncertainty high. Calling Active Learning.")
        acquisition_task = self.acquirer.optimize_next_step(self.system_state)
        
        if acquisition_task:
            # On ajoute la tâche d'acquisition en priorité haute
            self.system_state.plan_queue.insert(0, acquisition_task)
            return self._transition(WorkflowState.EXECUTING, msg="Acquisition task added.")
        else:
            # Le module 4 a jeté l'éponge (coût trop haut ou gain trop faible)
            print("    Active Learning gave up. Forcing Verdict with doubt.")
            return self._transition(WorkflowState.VERDICT, msg="Exhausted.")

    def _handle_verdict(self):
        """Generation de la sortie finale."""
        priors = self.reasoner._extract_priors_from_graph(self.system_state)
        dep_mean = priors["dep_alpha"] / (priors["dep_alpha"] + priors["dep_beta"])
        expected_notice = priors.get("alexy_notice_months")
        if expected_notice is None:
            expected_notice = min(36.0, max(3.0, priors["seniority_years"] * (0.6 + dep_mean)))
        gap = expected_notice - priors["actual_notice_months"]
        decision = "INCERTAIN"
        if priors.get("p_grave_fault", 0.0) > 0.8:
            decision = "REJET (FAUTE GRAVE)"
        elif gap > 2:
            decision = "RUPTURE BRUTALE PROBABLE"
        confidence = max(0.0, min(1.0, 1.0 - self.system_state.entropy))
        self.replan_attempts = 0
        return {
            "status": "DONE",
            "final_state": "VERDICT",
            "decision": decision,
            "confidence_score": confidence,
            "explanation": f"Base sur {len(self.system_state.graph.nodes)} faits, anciennete {priors['seniority_years']:.1f} ans, ecart preavis {gap:.1f} mois (attendu {expected_notice:.1f})."
        }

    # --- UTILS ---

    def _transition(self, new_state: WorkflowState, msg: str) -> Dict:
        print(f"   >>> Transitioning to {new_state.name} ({msg})")
        self.state_machine_status = new_state
        # Ici : Sauvegarde DB (Checkpointing)
        return {"status": "RUNNING", "state": new_state.name, "description": msg}

    def _get_pending_question(self):
        return getattr(self, 'pending_question', "Question générique ?")

    def _hydrate_static_rules(self):
        static_rules = get_static_rules()
        if not static_rules:
            return
        existing_ids = {rule.rule_id for rule in self.system_state.rules_registry}
        for cfg in static_rules:
            if cfg.get("rule_id") in existing_ids:
                continue
            rule = Rule(
                rule_id=cfg.get("rule_id", f"STATIC_{len(existing_ids)}"),
                type=RuleType.STATIC,
                description=cfg.get("description", ""),
                logic_payload=cfg.get("logic_payload"),
            )
            self.system_state.rules_registry.append(rule)
            existing_ids.add(rule.rule_id)

# ==============================================================================
# SCÉNARIO DE TEST COMPLET (END-TO-END)
# ==============================================================================

if __name__ == "__main__":
    print("STARTING NEURO-SYMBOLIC LEGAL SYSTEM ENGINE...")
    
    # 1. Instanciation
    engine = LegalOrchestrator(case_id="FULL_DEMO_2025")
    
    # 2. Boucle d'exécution (Simulation d'un serveur API)
    # On boucle jusqu'à ce que le système demande une info ou finisse
    
    keep_running = True
    user_replied = False
    
    while keep_running:
        time.sleep(0.5) # Pour la lisibilité des logs
        
        # Simulation de l'input utilisateur au bon moment
        current_input = None
        if engine.state_machine_status == WorkflowState.INTERACTING and not user_replied:
            print("\n---  SIMULATING USER INTERACTION ---")
            print("User sees question: ", engine._get_pending_question())
            print("User types: 'Non, pas de faute.'")
            current_input = {"answer": "NO_FAULT"}
            user_replied = True # On ne répond qu'une fois pour la démo
        
        result = engine.run_step(user_input=current_input)
        
        if result.get("final_state") == "VERDICT":
            print("\n FINAL VERDICT REACHED:")
            print(result)
            keep_running = False
        
        if result.get("status") == "WAITING_FOR_USER" and user_replied:
            # Sécurité anti-boucle si la simulation input foire
            pass
