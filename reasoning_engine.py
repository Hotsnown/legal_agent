import math
import torch
import networkx as nx
import uuid
from typing import List, Dict, Any, Tuple, Optional

# Nous importons Pyro pour la modÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©lisation probabiliste
# (Dans un environnement rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©el, pip install pyro-ppl torch)
try:
    import pyro
    import pyro.distributions as dist
except ImportError:
    print("[WARN] Pyro/Torch not installed. Running in simulated mode.")
    pyro = None
    dist = None
    torch = None

from system_state import (
    SystemState, GraphNode, NodeType, Task, TaskStatus,
    CommercialRelationship, RuptureContext, MisconductEvent,
    ExtendedCommercialRelationship, FinancialHistory, RelationshipAttestation,
    NonSolicitationCommitment, LogisticsCondition, UnilateralModification, DigitalMessage,
    RuleType, Rule, EdgeType, WorldTag, AlexyWeight, Grounding
)
from domain_config import get_planner_blueprint, get_interpretation_templates

# ==============================================================================
# 3.1. LE PLANIFICATEUR STRATÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â°GIQUE (LLM+P)
# ==============================================================================


class StrategicPlanner:
    """
    Responsable de la 'M?ta-Cognition'.
    Il ne r?sout pas le dossier, il d?cide DANS QUEL ORDRE le r?soudre.
    """

    def __init__(self):
        # Le graphe de t?ches (DAG)
        self.task_graph = nx.DiGraph()
        self.blueprint = get_planner_blueprint() or [
            {"id": "t_check_prescription", "action": "CHECK_LIMITATION_PERIOD", "priority": 100, "dependencies": []},
            {"id": "t_qualify_relationship", "action": "QUALIFY_RELATIONSHIP", "priority": 90, "dependencies": []},
            {"id": "t_generate_interpretation", "action": "GENERATE_INTERPRETATION", "priority": 85, "dependencies": ["t_qualify_relationship"]},
            {"id": "t_apply_alexy", "action": "APPLY_ALEXY_BALANCING", "priority": 82, "dependencies": ["t_generate_interpretation"]},
            {"id": "t_calc_notice", "action": "COMPUTE_NOTICE_GAP", "priority": 80, "dependencies": ["t_apply_alexy"]},
            {"id": "t_check_fault", "action": "VERIFY_GRAVE_FAULT", "priority": 75, "dependencies": []}
        ]

    def generate_plan(self, state: SystemState, replan_reason: str = "") -> List[Task]:
        """
        Simule l'appel au LLM qui genere le graphe de dependance.
        Dans la realite, c'est un prompt 'Text-to-Graph'.
        """
        print("--- [PLANNER] Generating Strategy DAG ---")
        self.task_graph.clear()

        tasks: Dict[str, Task] = {}
        for entry in self.blueprint:
            deps = list(entry.get("dependencies", []))
            task = Task(
                task_id=entry["id"],
                action=entry["action"],
                priority=entry.get("priority", 50),
                dependencies=deps,
            )
            tasks[task.task_id] = task
            self.task_graph.add_node(task.task_id, data=task)
        for entry in self.blueprint:
            for dep in entry.get("dependencies", []):
                self.task_graph.add_edge(dep, entry["id"])

        replan_task = None
        if replan_reason:
            replan_task = Task(
                task_id=f"t_replan_{uuid.uuid4().hex[:6]}",
                action="REPLAN_CONTEXT",
                priority=95,
            )

        if replan_task:
            self.task_graph.add_node(replan_task.task_id, data=replan_task)
            first_task_id = self.blueprint[0]["id"] if self.blueprint else None
            if first_task_id:
                self.task_graph.add_edge(replan_task.task_id, first_task_id)

        ordered_task_ids = list(nx.topological_sort(self.task_graph))

        final_plan = []
        for tid in ordered_task_ids:
            final_plan.append(self.task_graph.nodes[tid]['data'])

        print(f"[PLAN] Plan Generated: {[t.action for t in final_plan]}")
        if replan_reason:
            print(f"[PLAN] Replan reason: {replan_reason}")
        return final_plan
class PyroLegalModel:
    """
    Encapsule la logique juridique sous forme de rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©seau BayÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©sien.
    C'est ici que la "Loi" devient une distribution de probabilitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©.
    """
    
    @staticmethod
    def rupture_brutale_model(data_context: Dict[str, Any]):
        """
        Le modÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨le Pyro.
        data_context contient les paramÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨tres extraits du Graphe (Priors).
        """
        if pyro is None: return 0.0 # Fallback si pas de Torch

        # --- 1. VARIABLES LATENTES (INCERTITUDES FACTUELLES) ---
        
        # A. La relation est-elle juridiquement "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©tablie" ? 
        # Prior : Vient du score de confiance du module Perception (ou Regex)
        # Si Regex a trouvÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© 10 ans de CA, proba proche de 1.0.
        is_established = pyro.sample("is_established", dist.Bernoulli(data_context["p_established"]))
        
        # B. GravitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© de la faute (ExonÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ration)
        # Prior : Vient de l'analyse des emails de reproches.
        # Si pas de faute trouvÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e, proba faible (0.1).
        has_grave_fault = pyro.sample("has_grave_fault", dist.Bernoulli(data_context["p_grave_fault"]))
        
        # C. IntensitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© de la DÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pendance (Facteur d'Alexy)
        # Distribution Beta pour modÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©liser une probabilitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© continue entre 0 et 1
        dependency_intensity = pyro.sample("dependency_intensity", dist.Beta(data_context["dep_alpha"], data_context["dep_beta"]))

        # --- 2. LOGIQUE DÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â°TERMINISTE (HARD RULES) ---
        
        # Calcul du PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©avis DÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â» (ModÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨le simplifiÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© de la jurisprudence)
        # Formule : 1 mois par annÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e d'anciennetÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© * facteur dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pendance
        seniority = data_context["seniority_years"]
        target_notice = data_context.get("alexy_notice_months")
        if target_notice is None:
            target_notice = seniority * 1.0 * (1.0 + (dependency_intensity * 0.5)) # Bonus max 50% si dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pendant
        else:
            target_notice = torch.tensor(float(target_notice))
        
        actual_notice = data_context["actual_notice_months"]
        
        # Le "Gap" (BrutalitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© mathÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©matique)
        notice_gap = torch.clamp(target_notice - actual_notice, min=0.0)
        
        # --- 3. FONCTION DE JUGEMENT (SOFT RULES / ALEXY) ---
        
        # ProbabilitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© de condamnation (Sigmoid sur le Gap)
        # Plus le Gap est grand, plus la condamnation est certaine.
        # On soustrait un seuil de tolÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rance (ex: 0.5 mois d'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cart est tolÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©)
        raw_liability_score = torch.sigmoid(notice_gap - 0.5) 
        
        # Masquage Logique (Hard Rule application)
        # Si Pas de relation OU Faute Grave = Pas de condamnation (0.0)
        final_verdict = is_established * (1.0 - has_grave_fault) * raw_liability_score
        
        # Observation (Optionnel : si on avait la jurisprudence exacte)
        # pyro.sample("obs", dist.Bernoulli(final_verdict), obs=torch.tensor(1.0))
        
        return final_verdict

class ReasoningEngine:
    def __init__(self):
        self.planner = StrategicPlanner()
        self.model = PyroLegalModel()
        self.pending_replan_reason: Optional[str] = None
        self.interpretation_templates = {tpl.get("key"): tpl for tpl in get_interpretation_templates()}

    def plan_strategy(self, state: SystemState):
        """Appelle le Planner et remplit la queue du State."""
        plan = self.planner.generate_plan(state)
        state.plan_queue = plan



    def request_replan(self, reason: str):
        self.pending_replan_reason = reason

    def consume_replan_request(self) -> Optional[str]:
        reason = self.pending_replan_reason
        self.pending_replan_reason = None
        return reason

    def replan_strategy(self, state: SystemState, reason: str = ""):
        state.plan_queue = []
        plan = self.planner.generate_plan(state, replan_reason=reason)
        state.plan_queue = plan
        return plan


def generate_interpretations(self, state: SystemState) -> List[str]:
    """
    Cree des noeuds d'interpretation (regles dynamiques) bases sur les faits ingeres.
    """
    created: List[str] = []
    existing_rule_ids = {r.rule_id for r in state.rules_registry}
    already_noted = set(state.interpretations_applied)

    def edge_exists(src: str, tgt: str, etype: EdgeType) -> bool:
        return any(e.source_id == src and e.target_id == tgt and e.type == etype for e in state.graph.edges)

    def resolve_meta(key: str, default_rule: str, default_label: str, default_probability: float):
        meta = (self.interpretation_templates or {}).get(key, {}) if hasattr(self, "interpretation_templates") else {}
        label = meta.get("label", default_label)
        rule_id = meta.get("rule_id", default_rule)
        prob = meta.get("probability", default_probability)
        desc = meta.get("description", "")
        return label, rule_id, prob, desc

    competence_triggered = False

    for node_id, node in list(state.graph.nodes.items()):
        content = node.content
        grounding: Optional[Grounding] = node.grounding if hasattr(node, "grounding") else None

        if isinstance(content, LogisticsCondition) and content.is_tacitly_renewable:
            label, rule_id, probability, meta_desc = resolve_meta(
                "franco_tacite", "interp_franco_tacite", "Interpretation : Franco reconduit", 0.82
            )
            if rule_id not in existing_rule_ids:
                state.rules_registry.append(
                        Rule(
                            rule_id=rule_id,
                            type=RuleType.DYNAMIC,
                            description=meta_desc or "Le franco tacite impose un preavis aligne sur la duree initiale",
                            python_snippet="def compute(condition):\\n    if condition.get('is_tacitly_renewable'):\\n        return condition.get('initial_duration_months', 0)\\n    return 0",
                        )
                )
                existing_rule_ids.add(rule_id)
            target_node_id = next((nid for nid, n in state.graph.nodes.items() if n.label == label), None)
            if not target_node_id:
                interp_node = GraphNode(
                    type=NodeType.RULE_APPLICATION,
                    label=label,
                    world_tag=WorldTag.SHARED,
                    probability_score=probability,
                    content={"type": "interpretation", "rule_id": rule_id},
                )
                state.graph.add_node(interp_node)
                target_node_id = interp_node.node_id
            if not edge_exists(node_id, target_node_id, EdgeType.INTERPRETS):
                state.graph.add_edge(node_id, target_node_id, EdgeType.INTERPRETS)
            created.append(label)

        if isinstance(content, UnilateralModification) and content.severity >= 0.5:
            label, rule_id, probability, meta_desc = resolve_meta(
                "unilateral_modification",
                "interp_unilateral_drop",
                "Interpretation : modification unilaterale",
                0.8,
            )
            if rule_id not in existing_rule_ids:
                state.rules_registry.append(
                        Rule(
                            rule_id=rule_id,
                            type=RuleType.DYNAMIC,
                            description=meta_desc or "Une concession imposee sans preavis accroit la brutalite",
                            python_snippet="def compute(severity):\\n    return max(0.0, min(1.0, 0.4 + 0.5*severity))",
                        )
                )
                existing_rule_ids.add(rule_id)
            target_node_id = next((nid for nid, n in state.graph.nodes.items() if n.label == label), None)
            if not target_node_id:
                interp_node = GraphNode(
                    type=NodeType.RULE_APPLICATION,
                    label=label,
                    world_tag=WorldTag.SHARED,
                    probability_score=probability,
                    content={"type": "interpretation", "rule_id": rule_id},
                )
                state.graph.add_node(interp_node)
                target_node_id = interp_node.node_id
            if not edge_exists(node_id, target_node_id, EdgeType.INTERPRETS):
                state.graph.add_edge(node_id, target_node_id, EdgeType.INTERPRETS)
            created.append(label)

        if isinstance(content, DigitalMessage) and content.platform in ("WHATSAPP", "EMAIL", "SMS"):
            label, rule_id, probability, meta_desc = resolve_meta(
                "digital_written",
                "interp_digital_written",
                "Interpretation : Message digital = ecrit",
                0.78,
            )
            if rule_id not in existing_rule_ids:
                state.rules_registry.append(
                        Rule(
                            rule_id=rule_id,
                            type=RuleType.DYNAMIC,
                            description=meta_desc or "Un message digital est traite comme un ecrit pour la notification.",
                            python_snippet="def compute(msg):\\n    return msg.get('platform') in ['WHATSAPP','EMAIL','SMS']",
                        )
                )
                existing_rule_ids.add(rule_id)
            target_node_id = next((nid for nid, n in state.graph.nodes.items() if n.label == label), None)
            if not target_node_id:
                interp_node = GraphNode(
                    type=NodeType.RULE_APPLICATION,
                    label=label,
                    world_tag=WorldTag.SHARED,
                    probability_score=probability,
                    content={"type": "interpretation", "rule_id": rule_id},
                )
                state.graph.add_node(interp_node)
                target_node_id = interp_node.node_id
            if not edge_exists(node_id, target_node_id, EdgeType.INTERPRETS):
                state.graph.add_edge(node_id, target_node_id, EdgeType.INTERPRETS)
            created.append(label)

        ground_text = grounding.text_span.lower() if grounding and grounding.text_span else ""
        doc_hint = (grounding.source_doc_id or "").lower() if grounding else ""
        label_text = (node.label or "").lower()
        if not competence_triggered and ("new york" in ground_text or "competence" in label_text or "newyork" in doc_hint):
            competence_triggered = True
            label, rule_id, probability, meta_desc = resolve_meta(
                "clause_competence_ny",
                "interp_competence_ny",
                "Clause de competence internationale (NY)",
                0.7,
            )
            target_node_id = next((nid for nid, n in state.graph.nodes.items() if n.label == label), None)
            if not target_node_id:
                interp_node = GraphNode(
                    type=NodeType.RULE_APPLICATION,
                    label=label,
                    world_tag=WorldTag.SHARED,
                    probability_score=probability,
                    content={"type": "interpretation", "note": "Competence New York detectee", "rule_id": rule_id},
                )
                state.graph.add_node(interp_node)
                target_node_id = interp_node.node_id
            if not edge_exists(node_id, target_node_id, EdgeType.INTERPRETS):
                state.graph.add_edge(node_id, target_node_id, EdgeType.INTERPRETS)
            created.append(label)
            self.request_replan(meta_desc or "Clause de competence New York detectee")

    if created:
        state.interpretations_applied = list(sorted(already_noted.union(created)))
    return created
    def apply_alexy_balancing(self, state: SystemState) -> float:
        """
        Applique un balancier d'Alexy pour estimer le preavis raisonnable (multiverse A/B).
        """
        context = self._extract_priors_from_graph(state)
        dep_mean = context["dep_alpha"] / (context["dep_alpha"] + context["dep_beta"])
        weights: List[AlexyWeight] = []

        # Dependance economique
        weights.append(
            AlexyWeight(
                principle_name="Dependance eco",
                intensity=min(10.0, dep_mean * 10.0),
                abstract_weight=8.0,
                reliability=context.get("p_established", 0.6),
            )
        )

        # Anciennete/stabilite
        weights.append(
            AlexyWeight(
                principle_name="Stabilite de la relation",
                intensity=min(10.0, context.get("seniority_years", 1.0) * 1.2),
                abstract_weight=7.0,
                reliability=0.7,
            )
        )

        if context.get("exclusivity_seen"):
            weights.append(
                AlexyWeight(
                    principle_name="Exclusivite de fait",
                    intensity=8.0,
                    abstract_weight=7.0,
                    reliability=0.75,
                )
            )

        if any(isinstance(n.content, UnilateralModification) for n in state.graph.nodes.values()):
            weights.append(
                AlexyWeight(
                    principle_name="Protection de la confiance",
                    intensity=7.0,
                    abstract_weight=6.5,
                    reliability=0.7,
                )
            )

        combined_score = sum(w.score for w in weights) / max(1, len(weights))
        base_notice = min(36.0, max(3.0, context.get("seniority_years", 1.0) * (0.8 + dep_mean)))
        multiplier = 1.0 + min(0.75, combined_score / 100.0)
        notice = min(36.0, base_notice * multiplier)

        state.alexy_notice_months = notice
        state.alexy_weights = weights
        label = "Preavis raisonnable (Alexy)"

        target_node_id = next((nid for nid, n in state.graph.nodes.items() if n.label == label), None)
        if target_node_id:
            node = state.graph.nodes[target_node_id]
            node.probability_score = 0.9
            node.content = {
                "type": "alexy_notice",
                "notice_months": notice,
                "base_notice": base_notice,
                "multiplier": multiplier,
            }
        else:
            alexy_node = GraphNode(
                type=NodeType.RULE_APPLICATION,
                label=label,
                world_tag=WorldTag.SHARED,
                probability_score=0.9,
                content={
                    "type": "alexy_notice",
                    "notice_months": notice,
                    "base_notice": base_notice,
                    "multiplier": multiplier,
                },
            )
            state.graph.add_node(alexy_node)
            target_node_id = alexy_node.node_id

        def link_supporters(kind):
            for nid, n in state.graph.nodes.items():
                if isinstance(n.content, kind) and not any(
                    e.source_id == nid and e.target_id == target_node_id and e.type == EdgeType.INTERPRETS
                    for e in state.graph.edges
                ):
                    state.graph.add_edge(nid, target_node_id, EdgeType.INTERPRETS)

        link_supporters(ExtendedCommercialRelationship)
        link_supporters(LogisticsCondition)
        link_supporters(UnilateralModification)

        state.interpretations_applied = list(sorted(set(state.interpretations_applied + [f"Alexy: {notice:.1f} mois"])))
        return notice
    def run_inference(self, state: SystemState, num_samples=1000) -> float:
        """
        Ex?cute une Inf?rence de Monte Carlo sur l'?tat actuel du graphe.
        Retourne l'entropie du verdict.
        """
        print("--- [REASONING] Running Probabilistic Inference (Pyro) ---")
        # 1. Extraction des Donn?es
        context = self._extract_priors_from_graph(state)
        dep_mean = context["dep_alpha"] / (context["dep_alpha"] + context["dep_beta"])
        print(
            f"   -> Context extracted: Seniority={context['seniority_years']}y, "
            f"Notice={context['actual_notice_months']}m, Dependency~{dep_mean:.2f}"
        )
        if context.get("alexy_notice_months"):
            print(f"   -> Alexy override notice: {context['alexy_notice_months']:.2f}m")
        if pyro is None:
            print("   -> [SIMULATION] Pyro not available. Falling back to heuristic inference.")
            entropy = self._heuristic_entropy(context)
            state.entropy = entropy
            print(f"   -> Heuristic entropy: {entropy:.3f}")
            return entropy
        # ==================================================================
        # CRITICAL FIX : NETTOYAGE DE LA STACK PYRO
        # ==================================================================
        try:
            pyro.clear_param_store()
            from pyro.poutine import runtime
            runtime._PYRO_STACK = []
        except Exception as e:
            print(f"[WARN] Warning during Pyro cleanup: {e}")
        # ==================================================================
        # 2. Inf?rence (Importance Sampling)
        try:
            posterior = pyro.infer.Importance(self.model.rupture_brutale_model, num_samples=num_samples).run(context)
            marginal = pyro.infer.EmpiricalMarginal(posterior)
            mean_verdict = marginal.mean.item()
            std_verdict = marginal.variance.sqrt().item()
            print(f"   -> Verdict Probability: {mean_verdict:.2%} (+/-{std_verdict:.2f})")
            p = mean_verdict
            p = max(0.00001, min(0.99999, p))
            entropy = - (p * math.log(p) + (1 - p) * math.log(1 - p))
            state.entropy = entropy
            print(f"   -> System Entropy: {entropy:.4f}")
            return entropy
        except Exception as e:
            print(f"[ERROR] Pyro Inference Error: {e}")
            state.entropy = 0.5
            return 0.5
    def _heuristic_entropy(self, context: Dict[str, Any]) -> float:
        """
        Approximation si Pyro n'est pas dispo : applique une forme de notice-gap + dependance.
        """
        dep_mean = context["dep_alpha"] / (context["dep_alpha"] + context["dep_beta"])
        target_notice = context.get("alexy_notice_months")
        if target_notice is None:
            target_notice = min(36.0, max(1.0, context["seniority_years"] * (0.8 + 0.8 * dep_mean)))
        notice_gap = max(0.0, target_notice - context["actual_notice_months"])
        gap_ratio = notice_gap / target_notice if target_notice else 0.0
        liability = gap_ratio * context["p_established"] * (1 - context["p_grave_fault"])
        liability = max(0.001, min(0.999, liability))
        return - (liability * math.log(liability) + (1 - liability) * math.log(1 - liability))

    def _extract_priors_from_graph(self, state: SystemState) -> Dict[str, Any]:
        """
        Traduit le graphe d'objets Pydantic en tenseurs/floats pour Pyro.
        """
        data = {
            "p_established": 0.6,
            "p_grave_fault": 0.1,
            "dep_alpha": 2.0,
            "dep_beta": 5.0,
            "seniority_years": 0.0,
            "actual_notice_months": 0.0,
            "alexy_notice_months": None,
        }
        dependency_hint = 0.2
        exclusivity_seen = False
        for node in state.graph.nodes.values():
            content = node.content
            if isinstance(content, ExtendedCommercialRelationship):
                data["p_established"] = max(data["p_established"], 0.98)
                years = [pt.year for pt in content.financial_history] if content.financial_history else []
                if years:
                    data["seniority_years"] = max(data["seniority_years"], (max(years) - min(years) + 1))
                if content.dependency_rate:
                    dependency_hint = max(dependency_hint, content.dependency_rate)
                if content.characteristics and getattr(content.characteristics, "is_exclusive", False):
                    exclusivity_seen = True
            elif isinstance(content, FinancialHistory):
                years = [dp.get("year") for dp in content.data_points if isinstance(dp, dict) and "year" in dp]
                years = [y for y in years if y]
                if years:
                    data["seniority_years"] = max(data["seniority_years"], (max(years) - min(years) + 1))
                    data["p_established"] = max(data["p_established"], 0.95)
            elif isinstance(content, RelationshipAttestation):
                data["p_established"] = max(data["p_established"], 0.95)
                data["seniority_years"] = max(
                    data["seniority_years"], content.period_end_year - content.period_start_year + 1
                )
            elif isinstance(content, NonSolicitationCommitment):
                exclusivity_seen = True
            elif isinstance(content, LogisticsCondition):
                dependency_hint = max(dependency_hint, 0.35)
                if content.is_tacitly_renewable:
                    data["actual_notice_months"] = max(data["actual_notice_months"], content.initial_duration_months)
            elif isinstance(content, RuptureContext):
                data["actual_notice_months"] = max(
                    data["actual_notice_months"], content.executed_notice_period_months
                )
            elif isinstance(content, MisconductEvent):
                data["p_grave_fault"] = max(data["p_grave_fault"], 0.6)
            elif isinstance(content, UnilateralModification):
                # Assimile ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ une rupture partielle non prÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½avisÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½e
                data["p_established"] = max(data["p_established"], 0.9)
                data["actual_notice_months"] = min(data["actual_notice_months"], 0.0)
                dependency_hint = max(dependency_hint, 0.4 + content.severity * 0.2)
        if exclusivity_seen:
            dependency_hint = max(dependency_hint, 0.45)
        data["dep_alpha"] = 2.0 + dependency_hint * 10.0
        data["dep_beta"] = 2.0 + (1.0 - dependency_hint) * 10.0
        if data["seniority_years"] == 0:
            data["seniority_years"] = 5.0
        if getattr(state, 'alexy_notice_months', None):
            data["alexy_notice_months"] = state.alexy_notice_months
        data["exclusivity_seen"] = exclusivity_seen
        return data

# ==============================================================================
# SIMULATION D'EXÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â°CUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Setup
    # On reprend l'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©tat oÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ le RegexEngine l'avait laissÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© (thÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©oriquement)
    state = SystemState(case_id="CASE_DEMO_REASONING")
    
    # On peuple manuellement le graphe comme si le RegexEngine avait tournÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©
    # (Pour l'indÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pendance du test)
    from system_state import FinancialHistory
    state.graph.add_node(GraphNode(
        type=NodeType.FACT, 
        label="Historique", 
        world_tag="SHARED", 
        probability_score=1.0,
        content=FinancialHistory(data_points=[{"year": i, "amount": 100} for i in range(2002, 2011)], total_turnover=900, average_annual_turnover=100)
    ))
    
    engine = ReasoningEngine()
    
    # 2. Phase 1 : Planification
    engine.plan_strategy(state)
    
    # 3. Phase 2 : InfÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rence Probabiliste
    # Simulation avec Pyro (si installÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©) ou Mock
    engine.run_inference(state)
    
    # 4. Check Verdict
    print("\n--- DECISION ---")
    if state.entropy < 0.6:
        print("[OK] Certitude suffisante pour un Verdict.")
    else:
        print("[?] Trop d'incertitude -> Besoin d'Active Learning (Module 4).")
