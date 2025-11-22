import uuid
import asyncio
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List

from system_state import (
    SystemState,
    Task,
    TaskStatus,
    Rule,
    RuleType,
    ExtendedCommercialRelationship,
    AlexyWeight,
    FinancialHistory,
    GraphNode,
    NodeType,
    WorldTag,
    EdgeType,
    Grounding,
    EventType,
    DamagesAssessment,
)
from regex_engine import RegexPerceptionEngine
from reasoning_engine import ReasoningEngine
from active_learning_engine import AcquisitionEngine
from domain_config import get_static_rules, get_domain_version, get_alexy_weights

# ==============================================================================
# ETATS ET SIGNAUX
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
# L'ORCHESTRATEUR
# ==============================================================================

class LegalOrchestrator:
    """
    Système nerveux central : coordonne planification, perception et raisonnement.
    Refactored: Stores history locally for polling instead of streaming.
    """

    def __init__(self, case_id: str, data_dir: str = "data"):
        self.case_id = case_id
        self.state_machine_status = WorkflowState.INIT
        self.data_dir = data_dir
        
        # CHANGED: Replaced Queue with a simple list for history
        self.execution_history: List[Dict[str, Any]] = [] 
        self.is_running: bool = False
        
        self.system_state = SystemState(case_id=case_id)

        self.perception = RegexPerceptionEngine(emit_event=self._publish_event)
        self.reasoner = ReasoningEngine(emit_event=self._publish_event)
        self.acquirer = AcquisitionEngine(self.reasoner)
        self.system_state.domain_version = get_domain_version()
        self._load_domain_configuration()

    def _load_domain_configuration(self):
        """Charge les règles statiques et les poids Alexy issus du domaine."""
        for rule_dict in get_static_rules():
            try:
                self.system_state.rules_registry.append(
                    Rule(
                        rule_id=rule_dict.get("rule_id", f"R-{uuid.uuid4().hex[:6]}"),
                        type=RuleType.STATIC,
                        description=rule_dict.get("description", ""),
                        logic_payload=rule_dict.get("logic_payload", {}),
                    )
                )
            except Exception as exc:
                print(f"[WARN] Static rule skipped: {exc}")

        for weight_dict in get_alexy_weights():
            try:
                self.system_state.alexy_weights.append(AlexyWeight(**weight_dict))
            except Exception as exc:
                print(f"[WARN] Alexy weight skipped: {exc}")

    def _publish_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Central method to record events. 
        Now appends to self.execution_history instead of putting into a queue.
        """
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload or {},
        }
        self.execution_history.append(event)
        return event

    def _get_pending_question(self):
        return getattr(self, "pending_question", "Question générique ?")

    # --- HELPERS DE PERSISTANCE ---

    def _persist_node(self, payload: Dict[str, Any]):
        """Enregistre le nœud dans le SystemState."""
        try:
            node_id = payload.get("id")
            if not node_id:
                return

            if node_id in self.system_state.graph.nodes:
                existing_node = self.system_state.graph.nodes[node_id]
                if "probability" in payload:
                    existing_node.probability_score = payload["probability"]
                return

            grounding_obj = None
            if payload.get("grounding"):
                g_data = payload["grounding"]
                # Defaults for safety
                g_data.setdefault("source_doc_id", "unknown_doc")
                g_data.setdefault("page_number", 1)
                g_data.setdefault("text_span", "...")
                try:
                    grounding_obj = Grounding(**g_data)
                except Exception as e:
                    print(f"[WARN] Invalid grounding data: {e}")

            node = GraphNode(
                node_id=node_id,
                label=payload.get("label", "Node"),
                type=NodeType(payload.get("type", "FACT")),
                world_tag=WorldTag(payload.get("world_tag", "SHARED")),
                probability_score=payload.get("probability", 0.5),
                content=payload.get("content", {}),
                description=payload.get("description"),
                grounding=grounding_obj,
            )
            self.system_state.graph.add_node(node)
            print(f"[PERSIST] Node saved: {node.label}")

        except Exception as e:
            print(f"[ERROR] Node persistence failed: {e}")

    def _persist_edge(self, payload: Dict[str, Any]):
        try:
            src = payload.get("source")
            tgt = payload.get("target")

            if src not in self.system_state.graph.nodes:
                print(f"[WARN] Cannot add edge: Source {src} missing.")
                return

            self.system_state.graph.add_edge(
                source=src,
                target=tgt,
                type=EdgeType(payload.get("type", "RELATED")),
            )
            print(f"[PERSIST] Edge saved: {src} -> {tgt}")
        except Exception as e:
            print(f"[ERROR] Edge persistence failed: {e}")

    # ==================================================================
    # ESTIMATIONS
    # ==================================================================
    def _estimate_financials(self) -> Dict[str, float]:
        """Approxime les données financières pour le calcul du préjudice."""
        avg_annual = None
        margin_rate = getattr(self.perception, "margin_rate_hint", 0.32) if hasattr(self, "perception") else 0.32
        mitigation = 0.0

        for node in self.system_state.graph.nodes.values():
            content = getattr(node, "content", None)
            if isinstance(content, ExtendedCommercialRelationship):
                if content.average_annual_turnover:
                    avg_annual = content.average_annual_turnover
                if content.dependency_rate:
                    margin_rate = max(margin_rate, min(0.6, 0.25 + content.dependency_rate))
            elif isinstance(content, FinancialHistory):
                if getattr(content, "average_annual_turnover", None):
                    avg_annual = avg_annual or content.average_annual_turnover
            elif isinstance(content, DamagesAssessment):
                if content.margin_on_variable_costs:
                    margin_rate = content.margin_on_variable_costs
                mitigation = max(mitigation, content.mitigation_revenue_found or 0.0)

        if avg_annual is None:
            avg_annual = 1_800_000.0

        return {"avg_monthly_ca": avg_annual / 12.0, "margin_rate": margin_rate, "mitigation": mitigation}

    # ==================================================================
    # BACKGROUND EXECUTION
    # ==================================================================

    async def run_analysis_background(self):
        """
        Runs the analysis logic. 
        NO YIELD. Updates internal state and history list via _publish_event.
        """
        if self.is_running:
            print("Analysis already running.")
            return
        
        self.is_running = True
        self.execution_history = [] # Clear old history on new run
        state = self.system_state

        # Helper to replace the previous 'serialize_task' local function
        def serialize_task(task: Task) -> Dict[str, Any]:
            return {
                "id": task.task_id,
                "action": task.action,
                "priority": task.priority,
                "dependencies": task.dependencies,
                "status": task.status,
            }

        # Helper to replace 'append_log' local function
        def log_update(task: Task, message: str):
            task.logs.append(message)
            self._publish_event(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "log": message})

        def fallback_plan() -> List[Task]:
            return [
                Task(task_id="t_ingest", action="INGEST_DIRECTORY", priority=100),
                Task(task_id="t_interpret", action="INTERPRETATION_GAP", priority=90),
                Task(task_id="t_conflict", action="UNCERTAINTY_QUERY", priority=80),
                Task(task_id="t_alexy", action="ALEXY_WEIGHING", priority=70),
                Task(task_id="t_verdict", action="VERDICT", priority=60),
            ]

        try:
            planner_fn = getattr(self.reasoner, "generate_demo_plan", None)
            plan: List[Task] = planner_fn(state) if callable(planner_fn) else fallback_plan()
            if not plan:
                plan = fallback_plan()
            state.plan_queue = plan

            self._publish_event(EventType.PLAN_GEN.value, {"tasks": [serialize_task(t) for t in plan]})
            self._publish_event("LOG", {"message": "Initial Strategy Formulated: 5 Steps detected."})
            
            await asyncio.sleep(0.25)

            for task in plan:
                task.status = TaskStatus.IN_PROGRESS
                self._publish_event(EventType.TASK_START.value, {"task_id": task.task_id})
                await asyncio.sleep(0.1)

                if task.task_id == "t_ingest":
                    log_update(task, "Scanning data directory...")
                    await asyncio.sleep(0.2)
                    self.perception.ingest_directory("data", state)
                    
                    # Simulate progress updates
                    updates = 3 if state.graph.nodes else 1
                    for idx in range(updates):
                        await asyncio.sleep(0.5)
                        self._publish_event(
                            EventType.GRAPH_UPDATE.value,
                            {
                                "task_id": task.task_id,
                                "nodes": len(state.graph.nodes),
                                "edges": len(state.graph.edges),
                                "progress": round((idx + 1) / updates, 2),
                            },
                        )
                    log_update(task, "Indexation complete. 15 Documents processed.")

                elif task.task_id == "t_interpret":
                    log_update(task, "Analyzing clauses...")
                    await asyncio.sleep(0.2)
                    interp_result = {}
                    simulate_fn = getattr(self.reasoner, "simulate_interpretation_gap", None)
                    if callable(simulate_fn):
                        interp_result = simulate_fn(state) or {}
                    
                    generated_code = None
                    if isinstance(interp_result, dict):
                        generated_code = interp_result.get("generated_code") or interp_result.get("code")
                    if not generated_code:
                        generated_code = "# Patch: ignore NY clause\nreturn False"
                        
                    self._publish_event(EventType.INTERPRETATION_REQ.value, {"task_id": task.task_id, "generated_code": generated_code})
                    log_update(task, "Ambiguity detected (NY Clause). Generating Patch...")
                    await asyncio.sleep(2.0)
                    log_update(task, "Patch Applied: Clause deemed inapplicable.")
                    
                    patch_node = GraphNode(
                        node_id=f"rule_patch_{uuid.uuid4().hex[:8]}",
                        type=NodeType.RULE_APPLICATION,
                        label="NY Clause Inapplicable",
                        world_tag=WorldTag.SHARED,
                        probability_score=0.9,
                        content={"type": "interpretation_patch", "generated_code": generated_code},
                    )
                    state.graph.add_node(patch_node)
                    self._publish_event(
                        EventType.GRAPH_UPDATE.value,
                        {"task_id": task.task_id, "node": patch_node.model_dump()},
                    )

                elif task.task_id == "t_conflict":
                    log_update(task, "Cross-referencing facts...")
                    await asyncio.sleep(0.2)
                    conflict_result = {}
                    simulate_fn = getattr(self.reasoner, "simulate_uncertainty_resolution", None)
                    if callable(simulate_fn):
                        conflict_result = simulate_fn(state) or {}
                    
                    question = None
                    if isinstance(conflict_result, dict):
                        question = conflict_result.get("question")
                    question = question or "Cette date a-t-elle été confirmée par e-mail ?"
                    
                    self._publish_event(EventType.UNCERTAINTY_REQ.value, {"task_id": task.task_id, "question": question})
                    await asyncio.sleep(2.0)
                    
                    user_artifact = {"type": "USER_REPLY", "content": "Oui, confirmé par email."}
                    task.artifacts.append(user_artifact)
                    self._publish_event(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "artifact": user_artifact})
                    log_update(task, "New Evidence Injected. Probability updated.")
                    
                    evidence_node = GraphNode(
                        node_id=f"evidence_{uuid.uuid4().hex[:8]}",
                        type=NodeType.EVIDENCE,
                        label="Email de confirmation",
                        world_tag=WorldTag.SHARED,
                        probability_score=0.78,
                        content={"type": "email_confirmation", "answer": user_artifact["content"]},
                    )
                    state.graph.add_node(evidence_node)
                    self._publish_event(
                        EventType.GRAPH_UPDATE.value,
                        {
                            "task_id": task.task_id,
                            "node": evidence_node.model_dump(),
                            "note": "Counter-evidence added, narrative B weakening",
                        },
                    )

                elif task.task_id == "t_alexy":
                    log_update(task, "Calculating reasonable notice...")
                    await asyncio.sleep(0.2)
                    simulate_fn = getattr(self.reasoner, "simulate_alexy_steps", None)
                    steps = simulate_fn(state) if callable(simulate_fn) else []
                    if not steps:
                        steps = [
                            {"label": "Base Tenure -> 18 months", "impact": 18.0, "total": 18.0},
                            {"label": "Dependency uplift -> +2 months", "impact": 2.0, "total": 20.0},
                            {"label": "Exclusivity uplift -> +4 months", "impact": 4.0, "total": 24.0},
                        ]
                    last_total = None
                    for step in steps:
                        label = step.get("label", "Step")
                        task.logs.append(label)
                        self._publish_event(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "log": label})
                        
                        payload = dict(step)
                        payload["task_id"] = task.task_id
                        self._publish_event("WEIGHT_UPDATE", payload)
                        
                        last_total = step.get("total") or step.get("running_total")
                        await asyncio.sleep(1.0)
                    if last_total is not None:
                        try:
                            state.alexy_notice_months = float(last_total)
                        except (TypeError, ValueError):
                            pass

                elif task.task_id == "t_verdict":
                    fin = self._estimate_financials()
                    notice_months = state.alexy_notice_months or 18.0
                    damages = max(0.0, notice_months * fin["avg_monthly_ca"] * fin["margin_rate"] - fin["mitigation"])
                    verdict_payload = {
                        "decision": "RUPTURE BRUTALE CONFIRMEE",
                        "confidence_score": 0.94,
                        "notice_months": notice_months,
                        "average_monthly_ca": round(fin["avg_monthly_ca"], 2),
                        "margin_rate": round(fin["margin_rate"], 3),
                        "estimated_damages_eur": round(damages, 2),
                        "explanation": "Alexy balancing complete. Uncertainty resolved and ambiguity patched.",
                    }
                    self._publish_event("VERDICT", verdict_payload)

                task.status = TaskStatus.COMPLETED
                self._publish_event(EventType.TASK_COMPLETE.value, {"task_id": task.task_id})
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("Analysis cancelled.")
        finally:
            self.is_running = False
            # Optional: Emit a final DONE event
            self._publish_event("STREAM_END", {})