import os
import uuid
import time
import asyncio
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any

# Importation des Modules précédents
from system_state import (
    SystemState,
    Task,
    TaskStatus,
    Rule,
    RuleType,
    ExtendedCommercialRelationship,
    FinancialHistory,
)
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
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._stream_task: Optional[asyncio.Task] = None
        # --- Initialisation des Organes ---
        self.system_state = SystemState(case_id=case_id)
        self.perception = RegexPerceptionEngine(emit_event=self._publish_event)
        self.reasoner = ReasoningEngine(emit_event=self._publish_event)
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

    def _publish_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload or {},
        }
        try:
            self.event_queue.put_nowait(event)
        except Exception:
            pass
        return event

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

    # --- Legacy compatibility (used by /step endpoint) ---
    def _legacy_run_step(self, user_input: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Legacy step runner placeholder to keep existing endpoints alive.
        Streaming demo does not execute state-machine logic; this stub signals unavailability.
        """
        return {"status": "STREAM_MODE_ONLY", "state": self.state_machine_status.name}

    # ==================================================================
    # MODE STREAMING DEMO (100 MICRO-EVENEMENTS)
    # ==================================================================

    async def stream_analysis(self, user_input: Optional[Dict] = None):
        """
        Démo SSE : séquence linéaire (~100 micro-évènements) pour saturer l'UI.
        Aucun appel aux moteurs réels, tout est scénarisé pour la narration.
        """
        domain_version = self.system_state.domain_version or get_domain_version()

        def evt(event_type: str, payload: Optional[Dict[str, Any]] = None, delay: float = 0.08):
            event = {"type": event_type, "payload": payload or {}, "delay": delay}
            event["payload"].setdefault("case_id", self.case_id)
            return event

        data_files = []
        if os.path.isdir(self.data_dir):
            data_files = [f for f in sorted(os.listdir(self.data_dir)) if os.path.isfile(os.path.join(self.data_dir, f))]
        if not data_files:
            data_files = [
                "Kbis_RBoutin.pdf",
                "Kbis_TI.pdf",
                "Historique_CA_HT_1987_2012.csv",
                "Facture_RB-2009-0315.pdf",
                "Attestation_Mme_Vivion.docx",
                "Email_Franco_2008.msg",
            ]

        actor_boutin = {"id": "actor_boutin", "label": "R. BOUTIN S.A", "type": "ACTOR", "world_tag": "SHARED", "probability": 0.98}
        actor_ti = {"id": "actor_ti", "label": "TÔLERIE INDUSTRIELLE", "type": "ACTOR", "world_tag": "SHARED", "probability": 0.98}
        rupture_node = {"id": "reason_rupture", "label": "Rupture commerciale", "type": "EVENT", "world_tag": "SHARED", "probability": 0.6}
        impayes_node_id = "impayes_repetes"

        financial_points = [
            ("1999", 1.2), ("2002", 2.4), ("2004", 2.9), ("2006", 3.1), ("2008", 3.2),
            ("2009", 3.0), ("2010", 3.4), ("2011", 3.6), ("2012", 3.8), ("2013", 4.0),
        ]

        email_files = [
            "email_2008-05-29_franco.msg",
            "email_2009-03-12_relance.msg",
            "email_2010-07-02_conditions.doc",
            "email_2011-01-15_impayes.msg",
            "email_2011-03-22_reconduction.pdf",
            "email_2011-06-12_tolerance.msg",
            "email_2011-07-01_suspension.msg",
            "email_2011-07-15_compte_rendu.msg",
            "email_2011-08-05_escalade.msg",
            "email_2011-08-20_relance_finance.msg",
            "email_2011-09-01_avocat.msg",
            "email_2011-09-12_estoppel.msg",
            "email_2011-10-04_negociation.msg",
            "email_2011-11-19_rupture.msg",
            "email_2011-12-24_preavis.msg",
        ]

        events = []

        # ACTE 1 - Boot & Ingestion
        events += [
            evt("LOG", {"message": f"Initialisation du moteur Neuro-Symbolique v{domain_version}..."}, 0.05),
            evt("LOG", {"message": "Chargement de l'ontologie juridique (L.442-1 II)...", "phase": "BOOT"}, 0.05),
            evt("LOG", {"message": "Compilation des règles statiques (Rome I, L.442-1 II)...", "phase": "BOOT"}, 0.06),
            evt("PROBABILITY_UPDATE", {"probability": 0.0, "note": "Démarrage, entropie maximale"}, 0.04),
            evt("PROBABILITY_UPDATE", {"probability": 0.5, "note": "Calibration initiale"}, 0.04),
        ]

        for f in data_files:
            events.append(evt("LOG", {"message": f"Reading file: {f}"}, 0.05))
        events.append(evt("NODE_ADDED", actor_boutin, 0.06))
        events.append(evt("NODE_ADDED", actor_ti, 0.06))
        events.append(evt("EDGE_ADDED", {"source": actor_boutin["id"], "target": actor_ti["id"], "type": "RELATED"}, 0.06))
        events.append(evt("NODE_ADDED", rupture_node, 0.05))

        for i in range(6):
            events.append(evt("LOG", {"message": f"Indexation vecteur n°{i+1}..."}, 0.04))

        events.append(evt("LOG", {"message": "Lecture: Historique_CA_HT_1987_2012.csv"}, 0.08))
        for year, amount in financial_points:
            events.append(evt("NODE_ADDED", {"id": f"ca_{year}", "label": f"CA {year}: {amount} M€", "type": "FACT", "world_tag": "SHARED", "probability": 0.92}, 0.03))
        for year, _ in financial_points:
            events.append(evt("EDGE_ADDED", {"source": f"ca_{year}", "target": actor_boutin["id"], "type": "RELATED"}, 0.03))
        events.append(evt("LOG", {"message": "Détection d'une relation commerciale établie (>20 ans)."}, 0.12))
        events.append(evt("PROBABILITY_UPDATE", {"probability": 0.95, "note": "Relation établie consolidée"}, 0.12))

        # ACTE 2 - Clause NY faux positif
        events.append(evt("LOG", {"message": "Analyse des clauses contractuelles..."}, 0.1))
        events.append(evt("NODE_ADDED", {"id": "clause_ny", "label": "Clause Compétence : New York", "type": "FACT", "world_tag": "SHARED", "probability": 0.6}, 0.1))
        events.append(evt("ALERT", {"level": "HIGH", "message": "DEFEATER DETECTED: Juridiction Hors-UE potentielle.", "target": "clause_ny"}, 0.1))
        events.append(evt("LOG", {"message": "Vérification de l'applicabilité (Règlement Rome I)...", "phase": "CHECK"}, 1.5))
        events.append(evt("PROBABILITY_UPDATE", {"probability": 0.05, "note": "Clause NY rejetée"}, 0.08))
        events.append(evt("LOG", {"message": "Clause écartée : Inapplicable aux délits civils."}, 0.08))

        # ACTE 2 - Attaque
        events.append(evt("LOG", {"message": "Analyse des pièces de la défense (Mme Vivion)..."} ,0.12))
        events.append(evt("NARRATIVE_UPDATE", {"message": "Activation Multiverse : Monde B (Défense)"}, 0.08))
        events.append(evt("NODE_ADDED", {"id": impayes_node_id, "label": "Allégation : Impayés Répétés", "type": "FACT", "world_tag": "NARRATIVE_B", "probability": 0.62}, 0.1))
        events.append(evt("NODE_ADDED", {"id": "suspension_livraisons", "label": "Suspension des livraisons", "type": "EVENT", "world_tag": "NARRATIVE_B", "probability": 0.55}, 0.1))
        events.append(evt("EDGE_ADDED", {"source": impayes_node_id, "target": rupture_node["id"], "type": "CAUSES"}, 0.1))
        events.append(evt("PROBABILITY_UPDATE", {"probability": 0.45, "note": "Risque Faute Grave détecté"}, 0.12))
        events.append(evt("LOG", {"message": "Scénario de Faute Grave détecté. Risque de rejet de la demande."}, 0.14))

        # ACTE 3 - Recherche active
        events.append(evt("LOG", {"message": "Incohérence détectée. Recherche de preuves contradictoires..."}, 0.12))
        events.append(evt("LOG", {"message": "Scan profond des emails (2008-2011)...", "phase": "SCAN"}, 0.12))
        for email in email_files:
            events.append(evt("LOG", {"message": f"Parsing: {email}"}, 0.05))
        for i in range(5):
            events.append(evt("PROBABILITY_UPDATE", {"probability": 0.45 + i * 0.02, "note": "Oscillation pendant la fouille"}, 0.06))

        # Smoking Gun
        events.append(evt("NODE_ADDED", {"id": "email_tolerance", "label": "Email : Pas de souci pour le délai", "type": "EVIDENCE", "world_tag": "SHARED", "probability": 0.99}, 0.12))
        events.append(evt("LOG", {"message": "Analyse sémantique : Tolérance financière explicite."}, 0.12))
        events.append(evt("EDGE_ADDED", {"source": "email_tolerance", "target": impayes_node_id, "type": "CONTRADICTS"}, 0.1))
        events.append(evt("NARRATIVE_COLLAPSE", {"collapsed_id": impayes_node_id, "rescued_by": "email_tolerance", "message": "Preuve de tolérance neutralise la faute grave"}, 0.12))
        events.append(evt("LOG", {"message": "Narratif Défenseur invalidé (Principe de l'Estoppel)."}, 0.12))

        # Verdict buildup
        prob_track = [0.50, 0.65, 0.80, 0.88, 0.92]
        for p in prob_track:
            events.append(evt("PROBABILITY_UPDATE", {"probability": p, "note": "Raffinement du verdict"}, 0.15))
        events.append(evt("LOG", {"message": "Calcul du préjudice (Méthode Marge sur Coûts Variables)...", "phase": "DAMAGES"}, 0.12))
        events.append(evt("NODE_ADDED", {"id": "prejudice_estime", "label": "Préjudice Estimé : 1.8 M€", "type": "CONCEPT", "world_tag": "SHARED", "probability": 0.9}, 0.1))

        verdict_payload = {
            "decision": "RUPTURE BRUTALE ÉTABLIE",
            "confidence_score": 0.92,
            "explanation": "Relation de 22 ans rompue avec brutalité. L'argument de faute grave est contredit par la preuve de tolérance factuelle (Email 12/06/2011).",
            "case_id": self.case_id,
        }
        events.append(evt("VERDICT", verdict_payload, 0.0))

        # Padding to ~100 events avec des heartbeats si besoin
        while len(events) < 100:
            idx = len(events) + 1
            events.append(evt("LOG", {"message": f"Heartbeat #{idx}"}, 0.05))

        for event in events:
            yield event
            await asyncio.sleep(event.get("delay", 0.1))
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
        
        result = engine._legacy_run_step(user_input=current_input)
        
        if result.get("final_state") == "VERDICT":
            print("\n FINAL VERDICT REACHED:")
            print(result)
            keep_running = False
        
        if result.get("status") == "WAITING_FOR_USER" and user_replied:
            # Sécurité anti-boucle si la simulation input foire
            pass
