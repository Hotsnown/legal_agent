import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# I. ONTOLOGIE JURIDIQUE (DOMAIN VOCABULARY - RUPTURE BRUTALE)
# ==============================================================================

class WorldTag(str, Enum):
    """Le système de Multiverse : A qui appartient ce fait ?"""
    SHARED = "SHARED"           # Fait accepté par les deux parties (ex: Date du contrat)
    NARRATIVE_A = "NARRATIVE_A" # Version du Demandeur (Victime)
    NARRATIVE_B = "NARRATIVE_B" # Version du Défenseur (Auteur rupture)

class EntityType(str, Enum):
    ACTOR = "ACTOR"
    EVIDENCE = "EVIDENCE"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"

# --- 1. Base & Grounding ---

class Grounding(BaseModel):
    """Lien de traçabilité vers le document source (The Golden Thread)."""
    source_doc_id: str
    page_number: int
    bbox: Optional[List[float]] = None # [x1, y1, x2, y2]
    text_span: str = Field(..., description="Citation exacte du texte source")
    vector_embedding: Optional[List[float]] = Field(None, description="Pour recherche sémantique")

class OntologyObject(BaseModel):
    """Classe mère de tous les objets métier."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None

# --- 2. Acteurs (Actors) ---

class Actor(OntologyObject):
    name: str
    role: Literal["DEMANDEUR", "DEFENDEUR", "TIERS"]

class CorporateEntity(Actor):
    """Société commerciale."""
    siren: Optional[str] = None
    legal_form: Optional[str] = None # SA, SARL...
    sector: Optional[str] = None # Important pour le préavis d'usage
    turnover_last_year: Optional[float] = None # Pour évaluer la dépendance éco

class NaturalPerson(Actor):
    """Individu (ex: Gérant, Salarié)."""
    job_title: Optional[str] = None

# --- 3. Preuves (Evidence) ---

class Evidence(OntologyObject):
    title: str
    date: Optional[datetime] = None

class Contract(Evidence):
    """Contrat formel."""
    is_signed: bool = False
    has_tacit_renewal: bool = False # Clause de tacite reconduction

class DigitalMessage(Evidence):
    """Email, SMS, WhatsApp."""
    platform: Literal["EMAIL", "WHATSAPP", "SMS", "SLACK"]
    sender: str
    recipient: str

# --- 4. Événements & Concepts (Rupture Brutale Specifics) ---

class CommercialRelationship(OntologyObject):
    """
    LE SOCLE : Modélise la relation établie (L.442-1 II).
    C'est un Concept dynamique calculé à partir des flux.
    """
    start_date: datetime
    duration_years: float
    average_annual_turnover: float
    is_exclusive: bool = False # Facteur aggravant de dépendance
    sector_practice_notice: Optional[int] = None # Préavis d'usage en mois

class RuptureEvent(OntologyObject):
    """
    LE DÉCLENCHEUR : L'acte de rupture.
    """
    notification_date: datetime
    effective_end_date: datetime
    notice_period_given_months: float # Préavis accordé
    rupture_type: Literal["TOTAL", "PARTIAL_DESREFERENCEMENT", "PARTIAL_VOLUME_DROP"]
    form: Literal["WRITTEN", "ORAL", "TACIT"]

class GraveFaultEvent(OntologyObject):
    """
    L'EXCEPTION : Faute grave justifiant la rupture immédiate.
    """
    date: datetime
    severity: int = Field(..., ge=1, le=10) # 1-10
    was_condoned: bool = False # Si toléré par le passé, ce n'est plus une faute grave (Nemo auditur)
    prior_formal_notice: bool = False # Y a-t-il eu mise en demeure ?

class AnnualTurnover(BaseModel):
    """Pour prouver la 'régularité' et l'augmentation des flux."""
    year: int
    amount: float
    percentage_of_provider_total_revenue: Optional[float] = Field(
        None, description="Taux de dépendance économique pour cette année"
    )

class RelationshipCharacteristics(BaseModel):
    """Facteurs aggravants influençant la durée du préavis raisonnable."""
    is_private_label_manufacturer: bool = Field(
        False, description="Produits sous Marque de Distributeur (MDD) -> Aggravant"
    )
    has_dedicated_investments: bool = Field(
        False, description="Investissements non récupérables faits pour le client"
    )
    is_exclusive: bool = Field(False, description="Clause d'exclusivité de fait ou de droit")
    tender_frequency: Optional[Literal["ANNUAL", "PLURI_ANNUAL", "NONE"]] = Field(
        "NONE", description="Si remise en concurrence annuelle systématique -> Précarité"
    )

class ExtendedCommercialRelationship(OntologyObject):
    """
    Version enrichie de CommercialRelationship.
    Remplace la version précédente.
    """
    start_date: datetime
    contract_history_ids: List[str] = Field(default_factory=list)
    financial_history: List[AnnualTurnover] = Field(default_factory=list)
    characteristics: RelationshipCharacteristics = Field(default_factory=RelationshipCharacteristics)
    average_annual_turnover: Optional[float] = None
    dependency_rate: Optional[float] = None
    last_active_year: Optional[int] = None
    counterparty_group_name: Optional[str] = None

# ==============================================================================
# 2. LE DÉCLENCHEUR : NUANCES DE LA RUPTURE & IMPUTABILITÉ
# ==============================================================================

class PartialRuptureDetails(BaseModel):
    previous_volume_average: float
    new_imposed_volume: float
    price_decrease_percentage: float
    dereferenced_products_count: int

class RuptureMechanics(BaseModel):
    method: Literal["WRITTEN_LETTER", "EMAIL", "ORAL", "SILENCE", "CALL_FOR_TENDER"]
    is_brutal_termination: bool = Field(..., description="Arrêt immédiat sans préavis ?")
    initiator_apparent: Literal["DEMANDEUR", "DEFENDEUR"]
    initiator_actual_claim: Literal["DEMANDEUR", "DEFENDEUR"]
    partial_details: Optional[PartialRuptureDetails] = None

class RuptureContext(OntologyObject):
    notification_date: Optional[datetime] = None
    effective_end_date: datetime
    mechanics: RuptureMechanics
    executed_notice_period_months: float 

# ==============================================================================
# 3. LE VICE : ANALYSE DU PRÉAVIS (ACCORDÉ VS RAISONNABLE)
# ==============================================================================

class NoticeCalculationFactors(BaseModel):
    seniority_years: float
    dependency_rate: float
    reconversion_difficulty: Literal["LOW", "MEDIUM", "HIGH", "IMPOSSIBLE"]
    sector_custom_months: int = 6

class BrutalityAssessment(OntologyObject):
    target_notice_reasonable: Optional[float] = None
    actual_notice_given: float
    delta_months: float

# ==============================================================================
# 4. L'EXONÉRATION : LA FAUTE ET LA TOLÉRANCE
# ==============================================================================

class MisconductEvent(OntologyObject):
    date_of_occurrence: datetime
    date_of_discovery: datetime
    type: Literal["NON_PAYMENT", "QUALITY_ISSUE", "LATE_DELIVERY", "DISLOYALTY"]
    description: str
    was_previously_tolerated: bool = Field(False)
    formal_notice_sent: bool = Field(False)

class ForceMajeureClaim(OntologyObject):
    event_description: str
    is_unpredictable: bool
    is_irresistible: bool
    is_external: bool

# ==============================================================================
# 5. LA SANCTION : CALCUL DU PRÉJUDICE (MARGE SUR COÛTS VARIABLES)
# ==============================================================================

class MarginAnalysis(BaseModel):
    total_revenue_lost: float
    variable_costs_saved: float
    gross_margin_rate: float

class DamagesAssessment(OntologyObject):
    reference_period_months: int = 24
    average_monthly_margin: float
    mitigation_revenue_found: float = Field(0.0)
    estimated_total_damages: float

# --- NOUVELLES CLASSES POUR LES DOCUMENTS FOURNIS ---

class LogisticsCondition(OntologyObject):
    condition_name: str = "FRANCO_DE_PORT"
    threshold_amount: float
    start_date: datetime
    initial_duration_months: int
    is_tacitly_renewable: bool
    status: str = "ACTIVE"

class InvoiceEvidence(Evidence):
    invoice_number: str
    amount_ht: float
    currency: str = "EUR"
    confirms_conditions: List[str] = []

class FinancialHistory(OntologyObject):
    data_points: List[dict] = Field(default_factory=list)
    total_turnover: float
    average_annual_turnover: float

class RelationshipAttestation(Evidence):
    period_start_year: int
    period_end_year: int
    notes: List[str] = Field(default_factory=list)

class NonSolicitationCommitment(OntologyObject):
    scope: str
    start_date: Optional[datetime] = None
    territory: Optional[str] = None

class PurchaseOrderEvidence(Evidence):
    order_number: str
    amount_ht: Optional[float] = None
    incoterm: Optional[str] = None
    franco_threshold: Optional[float] = None
    payment_terms: Optional[str] = None
    comparative_client: Optional[str] = None

class DamagesScenario(BaseModel):
    notice_months: int
    average_turnover_millions: float
    margin_rate: float
    estimated_loss: float
    currency: str = "EUR"
    formula: Optional[str] = None

class DamagesTable(OntologyObject):
    scenarios: List[DamagesScenario] = Field(default_factory=list)
    reference_period_label: Optional[str] = None

class UnilateralModification(OntologyObject):
    change_type: Literal["PRIX", "VOLUME", "CONDITIONS_LOGISTIQUES", "PAIEMENT"]
    severity: float = Field(..., ge=0.0, le=1.0)
    effective_date: Optional[datetime] = None

# ==============================================================================
# II. LE GRAPHE NARRATIF (THE MULTIVERSE)
# ==============================================================================

class NodeType(str, Enum):
    FACT = "FACT"
    EVIDENCE = "EVIDENCE" 
    RULE_APPLICATION = "RULE_APP"
    EVENT = "EVENT"     # Added for V2 scenario (Modifications unilatérales)
    CONCEPT = "CONCEPT" # Added for V2 scenario (Relation établie)

class GraphNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType
    label: str
    world_tag: WorldTag
    probability_score: float = Field(0.5, ge=0.0, le=1.0)
    content: Union[
        CommercialRelationship,
        ExtendedCommercialRelationship,
        RuptureEvent,
        RuptureContext,
        BrutalityAssessment,
        GraveFaultEvent,
        MisconductEvent,
        ForceMajeureClaim,
        DamagesAssessment,
        CorporateEntity,
        Contract,
        DigitalMessage,
        Evidence,
        LogisticsCondition,
        InvoiceEvidence,
        FinancialHistory,
        RelationshipAttestation,
        NonSolicitationCommitment,
        PurchaseOrderEvidence,
        DamagesTable,
        UnilateralModification,
        dict,
    ]
    grounding: Optional[Grounding] = None

class EdgeType(str, Enum):
    CAUSES = "CAUSES"
    PROVES = "PROVES"
    CONTRADICTS = "CONTRADICTS"
    INTERPRETS = "INTERPRETS"
    BREAKS = "BREAKS"       # Added: Modification qui casse la relation
    CONFIRMS = "CONFIRMS"   # Added: Facture qui confirme un usage

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0

class NarrativeGraph(BaseModel):
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node

    def add_edge(self, source: str, target: str, type: EdgeType):
        self.edges.append(GraphEdge(source_id=source, target_id=target, type=type))

# ==============================================================================
# III. RÈGLES & PONDÉRATION (LOGIC & ALEXY)
# ==============================================================================

class AlexyWeight(BaseModel):
    principle_name: str
    intensity: float = Field(..., ge=0, le=10)
    abstract_weight: float = Field(..., ge=0, le=10)
    reliability: float = Field(..., ge=0, le=1.0)

    @property
    def score(self) -> float:
        return self.intensity * self.abstract_weight * self.reliability

class RuleType(str, Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"

class Rule(BaseModel):
    rule_id: str
    type: RuleType
    description: str
    logic_payload: Optional[Dict] = None 
    python_snippet: Optional[str] = None 

# ==============================================================================
# IV. PLANIFICATEUR (STRATEGY)
# ==============================================================================

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str
    target_node_id: Optional[str] = None
    priority: int = Field(50, ge=0, le=200)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)

# ==============================================================================
# V. THE SYSTEM STATE (SNAPSHOT FINAL)
# ==============================================================================

class SystemState(BaseModel):
    case_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    graph: NarrativeGraph = Field(default_factory=NarrativeGraph)
    rules_registry: List[Rule] = []
    plan_queue: List[Task] = []
    entropy: float = 1.0
    alexy_notice_months: Optional[float] = None
    alexy_weights: List[AlexyWeight] = Field(default_factory=list)
    interpretations_applied: List[str] = Field(default_factory=list)
    replan_history: List[str] = Field(default_factory=list)
    domain_version: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

if __name__ == "__main__":
    # Un simple test pour vérifier que le modèle Pydantic est valide
    state = SystemState(case_id="TEST_LOAD")
    print("System State Loaded Correctly.")