import os
import uuid
import time
import asyncio
import csv
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List

# Importation des Modules
from system_state import (
    SystemState,
    Task,
    TaskStatus,
    Rule,
    RuleType,
    ExtendedCommercialRelationship,
    FinancialHistory,
    GraphNode,
    NodeType,
    WorldTag,
    EdgeType
)
from regex_engine import RegexPerceptionEngine
from reasoning_engine import ReasoningEngine
from active_learning_engine import AcquisitionEngine
from domain_config import get_static_rules, get_domain_version

# ==============================================================================
# DÉFINITIONS DES ÉTATS ET SIGNAUX
# ==============================================================================

class WorkflowState(Enum):
    INIT = auto()
    PLANNING = auto()
    EXECUTING = auto()
    EVALUATING = auto()
    INTERACTING = auto()
    VERDICT = auto()

class OrchestratorSignal(Enum):
    CONTINUE = auto()
    WAIT = auto()
    STOP = auto()

# ==============================================================================
# L'ORCHESTRATEUR (THE BOSS) - V2 "ARCHITECTE"
# ==============================================================================

class LegalOrchestrator:
    """
    Le Système Nerveux Central.
    Version 2.1 : Intègre la persistance d'état pour la démo.
    """
    
    def __init__(self, case_id: str, data_dir: str = "data"):
        self.case_id = case_id
        self.state_machine_status = WorkflowState.INIT
        self.data_dir = data_dir
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.system_state = SystemState(case_id=case_id)
        
        # Simulation des moteurs pour l'architecture
        self.perception = RegexPerceptionEngine(emit_event=self._publish_event)
        self.reasoner = ReasoningEngine(emit_event=self._publish_event)
        self.acquirer = AcquisitionEngine(self.reasoner)
        self.system_state.domain_version = get_domain_version()

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

    def _get_pending_question(self):
        return getattr(self, 'pending_question', "Question générique ?")

    def _legacy_run_step(self, user_input: Optional[Dict] = None) -> Dict[str, Any]:
        return {"status": "STREAM_MODE_ONLY", "state": self.state_machine_status.name}

    # --- HELPERS DE PERSISTANCE (FIX POUR LE GRAPHE) ---

    def _persist_node(self, payload: Dict[str, Any]):
        """Enregistre le nœud dans le SystemState pour qu'il soit visible au refresh."""
        try:
            node_id = payload.get("id")
            if not node_id: return
            
            # Création propre du GraphNode
            node = GraphNode(
                node_id=node_id,
                label=payload.get("label", "Sans titre"),
                type=NodeType(payload.get("type", "FACT")),
                world_tag=WorldTag(payload.get("world_tag", "SHARED")),
                probability_score=payload.get("probability", 0.5),
                content=payload.get("content", {}), # Contenu générique
                description=payload.get("description", ""),
                grounding=payload.get("grounding", None)
            )
            self.system_state.graph.add_node(node)
        except Exception as e:
            print(f"[WARN] Failed to persist node {payload.get('id')}: {e}")

    def _persist_edge(self, payload: Dict[str, Any]):
        """Enregistre le lien dans le SystemState."""
        try:
            self.system_state.graph.add_edge(
                source=payload["source"],
                target=payload["target"],
                type=EdgeType(payload["type"])
            )
        except Exception as e:
            print(f"[WARN] Failed to persist edge: {e}")

    # ==================================================================
    # MODE STREAMING DEMO V2 (SCÉNARIO RICHE + PERSISTANCE)
    # ==================================================================

    async def stream_analysis(self, user_input: Optional[Dict] = None):
        """
        Scénario V2 : Relation établie -> Rupture Partielle -> Pesée Alexy -> Verdict.
        Exécute la logique ET persiste les données.
        """
        
        def evt(event_type: str, payload: Optional[Dict[str, Any]] = None, delay: float = 0.1):
            # 1. Persistance immédiate (Side Effect)
            if event_type == "NODE_ADDED":
                self._persist_node(payload)
            elif event_type == "EDGE_ADDED":
                self._persist_edge(payload)
            elif event_type == "PROBABILITY_UPDATE":
                self.system_state.entropy = 1.0 - payload.get("probability", 0.5)

            # 2. Construction de l'événement SSE
            event = {"type": event_type, "payload": payload or {}, "delay": delay}
            event["payload"].setdefault("case_id", self.case_id)
            return event

        # Données échantillon
        financial_history_snippet = [
            {"year": 1987, "amount": 0.64},
            {"year": 1995, "amount": 1.58},
            {"year": 2000, "amount": 2.11},
            {"year": 2005, "amount": 2.83},
            {"year": 2008, "amount": 3.22},
            {"year": 2011, "amount": 2.71},
            {"year": 2012, "amount": 2.26}
        ]

        # --- DÉBUT DU FLUX ---

        yield evt("LOG", {"message": "Démarrage de l'analyseur L.442-1 II (Rupture Brutale)..."}, 0.05)
        yield evt("LOG", {"message": "Chargement du contexte : 27 documents indexés."}, 0.05)

        # ---------------------------------------------------------
        # ACTE 1 : LE SOCLE (RELATION ÉTABLIE)
        # ---------------------------------------------------------
        yield evt("LOG", {"message": "PHASE 1 : Qualification de la Relation Commerciale...", "phase": "QUALIFICATION"}, 0.1)
        
        yield evt("LOG", {"message": "Ingestion : Historique_CA_HT_1987_2012.csv"}, 0.1)
        for pt in financial_history_snippet:
            yield evt("NODE_ADDED", {
                "id": f"ca_{pt['year']}", 
                "label": f"CA {pt['year']} : {pt['amount']} M€", 
                "type": "FACT", 
                "world_tag": "SHARED",
                "probability": 1.0
            }, 0.03)
        
        yield evt("PROBABILITY_UPDATE", {"probability": 0.2, "note": "Flux financiers détectés"}, 0.1)
        
        yield evt("LOG", {"message": "Analyse temporelle : Relation continue de 1987 à 2012 (25 ans)."}, 0.1)
        yield evt("NODE_ADDED", {
            "id": "relation_etablie", 
            "label": "Relation Commerciale Établie (>20 ans)", 
            "type": "CONCEPT", 
            "world_tag": "SHARED",
            "probability": 0.95,
            "description": "Flux ininterrompus et croissants."
        }, 0.1)
        
        # Création des liens temporels pour structurer le graphe
        for pt in financial_history_snippet:
             yield evt("EDGE_ADDED", {"source": f"ca_{pt['year']}", "target": "relation_etablie", "type": "PROVES"}, 0.02)

        yield evt("LOG", {"message": "Lecture : Attestation_Mme_VIVION_2012-01-31.docx"}, 0.1)
        yield evt("NODE_ADDED", {
            "id": "attest_vivion", 
            "label": "Preuve : Commandes régulières 2007-2011", 
            "type": "EVIDENCE", 
            "world_tag": "SHARED", 
            "probability": 0.9
        }, 0.1)
        yield evt("EDGE_ADDED", {"source": "attest_vivion", "target": "relation_etablie", "type": "PROVES"}, 0.1)
        
        yield evt("PROBABILITY_UPDATE", {"probability": 0.98, "note": "Socle juridique confirmé (L.442-1 II)"}, 0.1)

        # ---------------------------------------------------------
        # ACTE 2 : L'USAGE (LE FRANCO)
        # ---------------------------------------------------------
        yield evt("LOG", {"message": "PHASE 2 : Audit des Conditions Contractuelles...", "phase": "CONTRACT"}, 0.1)
        
        yield evt("LOG", {"message": "Scan : Email_2008-05-29_franco_de_port.docx"}, 0.1)
        yield evt("NODE_ADDED", {
            "id": "cond_franco_2008", 
            "label": "Accord : Franco de port (Mai 2008)", 
            "type": "FACT", 
            "world_tag": "SHARED", 
            "probability": 0.99,
            "grounding": {"source_doc_id": "Email_2008.docx", "page_number": 1, "text_span": "Franco dès 1 € net HT... tacitement reconduites"}
        }, 0.1)

        yield evt("LOG", {"message": "Vérification de l'exécution (Usage)..."}, 0.1)
        yield evt("LOG", {"message": "Scan : Facture_RB-2010-0709.docx"}, 0.05)
        yield evt("NODE_ADDED", {
            "id": "facture_2010", 
            "label": "Facture 2010 : Mention 'Franco maintenu'", 
            "type": "EVIDENCE", 
            "world_tag": "SHARED", 
            "probability": 1.0
        }, 0.1)
        yield evt("EDGE_ADDED", {"source": "facture_2010", "target": "cond_franco_2008", "type": "CONFIRMS"}, 0.1)
        
        yield evt("LOG", {"message": "Qualification Juridique : Le Franco est un avantage acquis (Usage)."}, 0.1)

        # ---------------------------------------------------------
        # ACTE 3 : LA RUPTURE PARTIELLE (LE BASCULEMENT)
        # ---------------------------------------------------------
        yield evt("LOG", {"message": "PHASE 3 : Détection d'Anomalies (Rupture)...", "phase": "SCAN"}, 0.1)
        
        yield evt("LOG", {"message": "Comparaison : Commande_1947_TFVM_comparatif_prix.docx"}, 0.1)
        yield evt("ALERT", {
            "level": "MEDIUM", 
            "message": "DÉTÉRIORATION DÉTECTÉE : Perte du Franco & Hausse Tarifs", 
            "target": "cond_franco_2008"
        }, 0.1)
        
        yield evt("NODE_ADDED", {
            "id": "modif_unilaterale", 
            "label": "Modification Unilatérale Substantielle", 
            "type": "EVENT", 
            "world_tag": "NARRATIVE_A", 
            "probability": 0.85,
            "description": "Suppression brutale de l'avantage logistique acquis."
        }, 0.1)
        
        yield evt("EDGE_ADDED", {"source": "modif_unilaterale", "target": "relation_etablie", "type": "BREAKS"}, 0.1)
        yield evt("LOG", {"message": "Application Règle : Modification substantielle imposée = RUPTURE PARTIELLE."}, 0.1)
        
        # Fausse piste défense (Impayés) évacuée rapidement
        yield evt("LOG", {"message": "Contrôle narratif défense (Impayés évoqués)..."}, 0.1)
        yield evt("LOG", {"message": "Rejeté : Tolérance établie par courriers antérieurs (Estoppel)."}, 0.1)

        # ---------------------------------------------------------
        # ACTE 4 : LA PESÉE (ALEXY)
        # ---------------------------------------------------------
        yield evt("LOG", {"message": "PHASE 4 : Calcul du Préavis Raisonnable (Moteur Alexy)...", "phase": "REASONING"}, 0.2)
        
        final_notice = 24
        
        yield evt("WEIGHT_UPDATE", {
            "factor": "Ancienneté (25 ans)", 
            "impact": "+18 mois", 
            "type": "BASE"
        }, 0.1)
        
        yield evt("LOG", {"message": "Détection Facteur Aggravant : Courrier_RBOUTIN_TI_2000..."}, 0.1)
        yield evt("WEIGHT_UPDATE", {
            "factor": "Exclusivité de fait (Non-prospection)", 
            "impact": "+4 mois", 
            "type": "AGGRAVATING"
        }, 0.1)
        
        yield evt("LOG", {"message": "Détection Facteur Aggravant : Dépendance Économique > 20%."}, 0.1)
        yield evt("WEIGHT_UPDATE", {
            "factor": "Dépendance Économique / Rupture Partielle", 
            "impact": "+2 mois", 
            "type": "AGGRAVATING"
        }, 0.1)
        
        yield evt("NODE_ADDED", {
            "id": "preavis_alexy", 
            "label": f"Préavis Raisonnable Calculé : {final_notice} mois", 
            "type": "RULE_APP", 
            "world_tag": "SHARED", 
            "probability": 0.95
        }, 0.1)
        yield evt("EDGE_ADDED", {"source": "relation_etablie", "target": "preavis_alexy", "type": "INTERPRETS"}, 0.1)
        
        yield evt("LOG", {"message": f"Préavis retenu : {final_notice} mois (vs 0 mois accordé)."}, 0.1)

        # ---------------------------------------------------------
        # ACTE 5 : LE VERDICT (QUANTIFICATION)
        # ---------------------------------------------------------
        yield evt("LOG", {"message": "PHASE 5 : Liquidation du Préjudice...", "phase": "DAMAGES"}, 0.1)
        
        yield evt("LOG", {"message": "Lecture Taux Marge : Attestation_EC_BOUTIN...docx"}, 0.1)
        margin_rate = 0.65
        avg_monthly_ca = 240000 # Approx du CSV (~2.9M / 12)
        total_damages = avg_monthly_ca * final_notice * margin_rate # ~3.7M
        
        yield evt("NODE_ADDED", {
            "id": "marge_brute", 
            "label": f"Taux Marge Certifié : {int(margin_rate*100)}%", 
            "type": "FACT", 
            "world_tag": "SHARED", 
            "probability": 1.0
        }, 0.1)

        yield evt("LOG", {"message": "Formule : (Moyenne CA) x (Delta Préavis) x (Taux Marge)"}, 0.1)
        
        verdict_payload = {
            "decision": "RUPTURE BRUTALE (TOTALE + PARTIELLE)",
            "confidence_score": 0.96,
            "explanation": (
                f"La suppression du Franco constituait une rupture partielle. "
                f"La rupture finale sans préavis est brutale. "
                f"Compte tenu de l'exclusivité et de la durée (25 ans), un préavis de {final_notice} mois était dû."
            ),
            "damages": f"{total_damages:,.0f} €".replace(",", " "),
            "case_id": self.case_id,
        }
        
        yield evt("VERDICT", verdict_payload, 0.1)
        
        yield evt("LOG", {"message": "Analyse terminée. Rapport généré."}, 0.1)

if __name__ == "__main__":
    # Test runner pour debug
    async def run():
        orc = LegalOrchestrator("TEST")
        async for e in orc.stream_analysis():
            print(e)
    asyncio.run(run())