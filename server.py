# server.py
import uuid
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import de nos modules précédents
from orchestrator import LegalOrchestrator, WorkflowState
from system_state import SystemState
from domain_config import load_domain_config

# Dans server.py

# ... imports ...

app = FastAPI(title="Neuro-Symbolic Legal Engine API")

# MODIFICATION ICI : Remplacez ["*"] par l'adresse explicite de votre frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # <--- Explicite au lieu de ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MÉMOIRE VOLATILE (Simule une DB Redis/Postgres) ---
ACTIVE_CASES: Dict[str, LegalOrchestrator] = {}

# --- MODÈLES API ---
class CreateCaseRequest(BaseModel):
    filename: str

class UserReplyRequest(BaseModel):
    answer: str

# --- ENDPOINTS ---

@app.post("/cases")
def create_case(req: CreateCaseRequest):
    """Initialise un nouveau dossier d'enquête."""
    case_id = str(uuid.uuid4())[:8]
    # On instancie l'Orchestrateur (Module 5)
    orchestrator = LegalOrchestrator(case_id=case_id)
    ACTIVE_CASES[case_id] = orchestrator
    
    return {
        "case_id": case_id, 
        "message": "Dossier initialisé. Prêt pour l'analyse.",
        "state": orchestrator.state_machine_status.name
    }

@app.post("/cases/{case_id}/step")
def run_step(case_id: str):
    """
    Force le système à avancer d'un pas (Step).
    Le Frontend appellera ceci en boucle ou manuellement.
    """
    if case_id not in ACTIVE_CASES:
        raise HTTPException(status_code=404, detail="Case not found")
    
    orchestrator = ACTIVE_CASES[case_id]
    
    # On exécute un pas de la Machine à États
    result = orchestrator.run_step()
    
    # On formate la réponse pour l'UI
    response = {
        "current_state": orchestrator.state_machine_status.name,
        "entropy": orchestrator.system_state.entropy,
        "logs": result, # Ce que le step a retourné (ex: transition info)
        "plan": [t.action for t in orchestrator.system_state.plan_queue],
        "facts_count": len(orchestrator.system_state.graph.nodes),
        "alexy_notice": orchestrator.system_state.alexy_notice_months,
        "interpretations": orchestrator.system_state.interpretations_applied,
        "replans": orchestrator.system_state.replan_history,
        "domain_version": orchestrator.system_state.domain_version,
    }
    
    # Si le système attend une réponse, on renvoie la question
    if orchestrator.state_machine_status == WorkflowState.INTERACTING:
        response["question"] = orchestrator._get_pending_question()
        
    # Si verdict, on renvoie la décision
    if result.get("final_state") == "VERDICT":
        response["verdict"] = result
        
    return response

@app.post("/cases/{case_id}/reply")
def reply_to_system(case_id: str, reply: UserReplyRequest):
    """L'utilisateur répond à une question du Module 4."""
    if case_id not in ACTIVE_CASES:
        raise HTTPException(status_code=404, detail="Case not found")
    
    orchestrator = ACTIVE_CASES[case_id]
    
    # On vérifie qu'on est bien en attente
    if orchestrator.state_machine_status != WorkflowState.INTERACTING:
        raise HTTPException(status_code=400, detail="System is not waiting for input.")
    
    # On injecte la réponse
    # Note: run_step gère l'input utilisateur s'il est fourni
    result = orchestrator.run_step(user_input={"answer": reply.answer})
    
    return {"status": "Reply processed", "next_state": orchestrator.state_machine_status.name}

@app.get("/cases/{case_id}/graph")
def get_graph_data(case_id: str):
    """Pour la visualisation du Graphe."""
    if case_id not in ACTIVE_CASES:
        raise HTTPException(status_code=404, detail="Case not found")
    
    state = ACTIVE_CASES[case_id].system_state
    
    # Sérialisation propre des noeuds ET des arêtes
    nodes_data = []
    for k, v in state.graph.nodes.items():
        nodes_data.append({
            "id": k,
            "label": v.label,
            "type": v.type,
            "world_tag": v.world_tag,
            "probability": v.probability_score,
            "grounding": v.grounding.dict() if v.grounding else None,
            "description": str(v.content) if hasattr(v, 'content') else ""
        })

    edges_data = []
    for e in state.graph.edges:
        edges_data.append({
            "source": e.source_id,
            "target": e.target_id,
            "type": e.type
        })

    return {"nodes": nodes_data, "edges": edges_data}

@app.get("/static/{doc_path:path}")
def serve_document(doc_path: str):
    """Expose les documents sources pour l'aperçu PDF/Doc côté front."""
    base_dir = Path("data").resolve()
    file_path = (base_dir / doc_path).resolve()
    if base_dir not in file_path.parents and file_path != base_dir:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(file_path)


@app.get("/domain-config")
def get_domain_manifest():
    """Expose la configuration du domaine (ontologie, règles, plan)."""
    return load_domain_config()

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting Neuro-Symbolic Backend on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
