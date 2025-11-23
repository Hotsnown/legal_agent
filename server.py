import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from domain_config import load_domain_config
from orchestrator import LegalOrchestrator

app = FastAPI(title="Neuro-Symbolic Legal Engine API (Simple Polling)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Simplified for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------------------
ACTIVE_CASES: Dict[str, LegalOrchestrator] = {}

def get_case_or_404(case_id: str) -> LegalOrchestrator:
    orchestrator = ACTIVE_CASES.get(case_id)
    if orchestrator is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return orchestrator

def get_or_create_case(case_id: str, data_dir: str = "data") -> LegalOrchestrator:
    orchestrator = ACTIVE_CASES.get(case_id)
    if orchestrator is None:
        orchestrator = LegalOrchestrator(case_id=case_id, data_dir=data_dir)
        ACTIVE_CASES[case_id] = orchestrator
    return orchestrator

# ------------------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------------------
class CreateCaseRequest(BaseModel):
    data_dir: str | None = None

class CreateCaseResponse(BaseModel):
    case_id: str
    message: str

class UserReplyRequest(BaseModel):
    answer: str

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.post("/cases", response_model=CreateCaseResponse)
async def create_case(req: CreateCaseRequest) -> CreateCaseResponse:
    case_id = uuid.uuid4().hex[:8]
    orchestrator = LegalOrchestrator(case_id=case_id, data_dir=req.data_dir or "data")
    ACTIVE_CASES[case_id] = orchestrator
    return CreateCaseResponse(case_id=case_id, message="Case initialized.")

# --- REFACTORED LOGIC: Start + Poll ---

@app.post("/cases/{case_id}/run")
async def run_case_analysis(case_id: str, background_tasks: BackgroundTasks):
    """
    Starts the analysis in the background. 
    Returns immediately.
    """
    orchestrator = get_or_create_case(case_id)
    
    if orchestrator.is_running:
        return {"status": "already_running", "message": "Analysis is already in progress."}
    
    # Trigger the orchestrator method in the background
    background_tasks.add_task(orchestrator.run_analysis_background)
    
    return {"status": "started", "message": "Analysis started in background."}

@app.get("/cases/{case_id}/events")
async def get_case_events(case_id: str):
    """
    Returns the full history of events for the case.
    Client should poll this endpoint.
    """
    orchestrator = get_or_create_case(case_id)
    
    return {
        "case_id": case_id,
        "is_running": orchestrator.is_running,
        "event_count": len(orchestrator.execution_history),
        "events": orchestrator.execution_history
    }

# --- END REFACTORED LOGIC ---

@app.post("/cases/{case_id}/reply")
async def reply_to_case(case_id: str, reply: UserReplyRequest) -> Dict[str, Any]:
    orchestrator = get_case_or_404(case_id)
    # Just record the reply, logic handles it in the next steps usually
    orchestrator.receive_interaction(reply.answer)
    return {"status": "received", "case_id": case_id}

@app.get("/cases/{case_id}/graph")
async def get_graph(case_id: str) -> Dict[str, Any]:
    state = get_case_or_404(case_id).system_state

    def encode(obj: Any) -> Any:
        try:
            return jsonable_encoder(obj)
        except Exception:
            return str(obj)

    nodes = []
    for node in state.graph.nodes.values():
        nodes.append(
            {
                "id": node.node_id,
                "label": node.label,
                "type": node.type.value if hasattr(node.type, "value") else str(node.type),
                "world_tag": node.world_tag.value if hasattr(node.world_tag, "value") else str(node.world_tag),
                "probability": node.probability_score,
                "description": getattr(node, "description", ""),
                "grounding": node.grounding.model_dump() if node.grounding else None,
                "content": encode(node.content),
            }
        )

    edges = []
    for edge in state.graph.edges:
        edges.append(
            {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.type.value if hasattr(edge.type, "value") else str(edge.type),
                "weight": edge.weight,
            }
        )

    return {"nodes": nodes, "edges": edges}

@app.get("/static/{doc_path:path}")
async def serve_document(doc_path: str):
    base_dir = Path("data").resolve()
    file_path = (base_dir / doc_path).resolve()
    if base_dir not in file_path.parents and file_path != base_dir:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(file_path)

@app.get("/domain-config")
async def get_domain_manifest():
    return load_domain_config()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)