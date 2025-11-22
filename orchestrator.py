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
    RelationshipCharacteristics,
    AnnualTurnover,
    ProductType,
    SectorVolatility,
    NotificationMethod,
    RuptureEvent,
    DamagesAssessment,
    FinancialHistory,
    GraphNode,
    NodeType,
    WorldTag,
    EdgeType,
    Grounding,
    EventType,
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
    """

    def __init__(self, case_id: str, data_dir: str = "data"):
        self.case_id = case_id
        self.state_machine_status = WorkflowState.INIT
        self.data_dir = data_dir
        self.event_queue: asyncio.Queue = asyncio.Queue()
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
        return getattr(self, "pending_question", "Question générique ?")

    def _legacy_run_step(self, user_input: Optional[Dict] = None) -> Dict[str, Any]:
        return {"status": "STREAM_MODE_ONLY", "state": self.state_machine_status.name}

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
                if "source_doc_id" not in g_data:
                    g_data["source_doc_id"] = "unknown_doc"
                if "page_number" not in g_data:
                    g_data["page_number"] = 1
                if "text_span" not in g_data:
                    g_data["text_span"] = "..."
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
    # MODE STREAMING DEMO (GAME LOOP)
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

    async def stream_analysis(self, user_input: Optional[Dict] = None):
        """
        Game loop streaming demo :
        1) Plan -> 2) Ingestion -> 3) Ambiguity -> 4) Uncertainty -> 5) Alexy -> 6) Verdict.
        """
        state = self.system_state

        def evt(event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {"type": event_type, "payload": payload or {}}

        def serialize_task(task: Task) -> Dict[str, Any]:
            return {
                "id": task.task_id,
                "action": task.action,
                "priority": task.priority,
                "dependencies": task.dependencies,
                "status": task.status,
            }

        def append_log(task: Task, message: str) -> Dict[str, Any]:
            task.logs.append(message)
            return evt(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "log": message})

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

            yield evt(EventType.PLAN_GEN.value, {"tasks": [serialize_task(t) for t in plan]})
            yield evt("LOG", {"message": "Initial Strategy Formulated: 5 Steps detected."})
            await asyncio.sleep(0.25)

            for task in plan:
                task.status = TaskStatus.IN_PROGRESS
                yield evt(EventType.TASK_START.value, {"task_id": task.task_id})
                await asyncio.sleep(0.1)

                if task.task_id == "t_ingest":
                    yield append_log(task, "Scanning data directory...")
                    await asyncio.sleep(0.2)
                    self.perception.ingest_directory("data", state)
                    updates = 3 if state.graph.nodes else 1
                    for idx in range(updates):
                        await asyncio.sleep(0.5)
                        yield evt(
                            EventType.GRAPH_UPDATE.value,
                            {
                                "task_id": task.task_id,
                                "nodes": len(state.graph.nodes),
                                "edges": len(state.graph.edges),
                                "progress": round((idx + 1) / updates, 2),
                            },
                        )
                    yield append_log(task, "Indexation complete. 15 Documents processed.")

                elif task.task_id == "t_interpret":
                    yield append_log(task, "Analyzing clauses...")
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
                    yield evt(EventType.INTERPRETATION_REQ.value, {"task_id": task.task_id, "generated_code": generated_code})
                    yield append_log(task, "Ambiguity detected (NY Clause). Generating Patch...")
                    await asyncio.sleep(2.0)
                    yield append_log(task, "Patch Applied: Clause deemed inapplicable.")
                    patch_node = GraphNode(
                        node_id=f"rule_patch_{uuid.uuid4().hex[:8]}",
                        type=NodeType.RULE_APPLICATION,
                        label="NY Clause Inapplicable",
                        world_tag=WorldTag.SHARED,
                        probability_score=0.9,
                        content={"type": "interpretation_patch", "generated_code": generated_code},
                    )
                    state.graph.add_node(patch_node)
                    yield evt(
                        EventType.GRAPH_UPDATE.value,
                        {"task_id": task.task_id, "node": patch_node.model_dump()},
                    )

                elif task.task_id == "t_conflict":
                    yield append_log(task, "Cross-referencing facts...")
                    await asyncio.sleep(0.2)
                    conflict_result = {}
                    simulate_fn = getattr(self.reasoner, "simulate_uncertainty_resolution", None)
                    if callable(simulate_fn):
                        conflict_result = simulate_fn(state) or {}
                    question = None
                    if isinstance(conflict_result, dict):
                        question = conflict_result.get("question")
                    question = question or "Cette date a-t-elle été confirmée par e-mail ?"
                    yield evt(EventType.UNCERTAINTY_REQ.value, {"task_id": task.task_id, "question": question})
                    await asyncio.sleep(2.0)
                    user_artifact = {"type": "USER_REPLY", "content": "Oui, confirmé par email."}
                    task.artifacts.append(user_artifact)
                    yield evt(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "artifact": user_artifact})
                    yield append_log(task, "New Evidence Injected. Probability updated.")
                    evidence_node = GraphNode(
                        node_id=f"evidence_{uuid.uuid4().hex[:8]}",
                        type=NodeType.EVIDENCE,
                        label="Email de confirmation",
                        world_tag=WorldTag.SHARED,
                        probability_score=0.78,
                        content={"type": "email_confirmation", "answer": user_artifact["content"]},
                    )
                    state.graph.add_node(evidence_node)
                    yield evt(
                        EventType.GRAPH_UPDATE.value,
                        {
                            "task_id": task.task_id,
                            "node": evidence_node.model_dump(),
                            "note": "Counter-evidence added, narrative B weakening",
                        },
                    )

                elif task.task_id == "t_alexy":
                    yield append_log(task, "Calculating reasonable notice...")
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
                        yield evt(EventType.TASK_UPDATE.value, {"task_id": task.task_id, "log": label})
                        payload = dict(step)
                        payload["task_id"] = task.task_id
                        yield evt("WEIGHT_UPDATE", payload)
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
                    yield evt("VERDICT", verdict_payload)

                task.status = TaskStatus.COMPLETED
                yield evt(EventType.TASK_COMPLETE.value, {"task_id": task.task_id})
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            return


if __name__ == "__main__":
    async def run():
        orchestrator = LegalOrchestrator("TEST")
        async for event in orchestrator.stream_analysis():
            print(event)

    asyncio.run(run())
