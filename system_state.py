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

from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator

# On reprend les classes de base définies précédemment (Grounding, OntologyObject)
# et on étend spécifiquement la partie métier.

# ==============================================================================
# 1. LE SOCLE : DÉTAIL DE LA RELATION COMMERCIALE (INTENSITÉ & STABILITÉ)
# ==============================================================================

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
    # Gestion des successions de contrats (ex: suite de CDD)
    contract_history_ids: List[str] = Field(default_factory=list)
    financial_history: List[AnnualTurnover] = Field(default_factory=list)
    characteristics: RelationshipCharacteristics = Field(default_factory=RelationshipCharacteristics)
    average_annual_turnover: Optional[float] = None
    dependency_rate: Optional[float] = None
    last_active_year: Optional[int] = None
    # Group Implication : La relation est-elle avec une filiale ou le groupe ?
    counterparty_group_name: Optional[str] = None

# ==============================================================================
# 2. LE DÉCLENCHEUR : NUANCES DE LA RUPTURE & IMPUTABILITÉ
# ==============================================================================

class PartialRuptureDetails(BaseModel):
    """Détails si la rupture n'est pas totale mais 'substantielle'."""
    previous_volume_average: float
    new_imposed_volume: float
    price_decrease_percentage: float
    dereferenced_products_count: int

class RuptureMechanics(BaseModel):
    """Comment la rupture s'est-elle opérée ?"""
    method: Literal["WRITTEN_LETTER", "EMAIL", "ORAL", "SILENCE", "CALL_FOR_TENDER"]
    is_brutal_termination: bool = Field(..., description="Arrêt immédiat sans préavis ?")
    
    # Imputabilité : Qui est le 'vrai' responsable ?
    # Ex: Le fournisseur arrête de livrer (Apparent) car le distributeur ne paie plus (Réel)
    initiator_apparent: Literal["DEMANDEUR", "DEFENDEUR"]
    initiator_actual_claim: Literal["DEMANDEUR", "DEFENDEUR"] # Ce que le système doit déduire
    
    partial_details: Optional[PartialRuptureDetails] = None

class RuptureContext(OntologyObject):
    """
    Objet complet remplaçant RuptureEvent.
    """
    notification_date: Optional[datetime] = None # Peut être None si rupture tacite
    effective_end_date: datetime
    
    mechanics: RuptureMechanics
    
    # Le préavis qui a été REELLEMENT respecté (ex: relation continuée pendant 3 mois)
    executed_notice_period_months: float 

# ==============================================================================
# 3. LE VICE : ANALYSE DU PRÉAVIS (ACCORDÉ VS RAISONNABLE)
# ==============================================================================

class NoticeCalculationFactors(BaseModel):
    """Inputs pour l'algorithme de calcul du préavis raisonnable."""
    seniority_years: float
    dependency_rate: float # % du CA réalisé avec ce client
    reconversion_difficulty: Literal["LOW", "MEDIUM", "HIGH", "IMPOSSIBLE"]
    sector_custom_months: int = 6 # Usage interprofessionnel par défaut

class BrutalityAssessment(OntologyObject):
    """
    Nœud de synthèse comparant les deux valeurs.
    Ce nœud est souvent la cible de la tâche d'inférence.
    """
    target_notice_reasonable: Optional[float] = None # Calculé par le système
    actual_notice_given: float
    
    delta_months: float # Si négatif -> Brutalité

# ==============================================================================
# 4. L'EXONÉRATION : LA FAUTE ET LA TOLÉRANCE
# ==============================================================================

class MisconductEvent(OntologyObject):
    """Manquement contractuel invoqué par le défendeur."""
    date_of_occurrence: datetime
    date_of_discovery: datetime # Crucial pour la tolérance
    type: Literal["NON_PAYMENT", "QUALITY_ISSUE", "LATE_DELIVERY", "DISLOYALTY"]
    description: str
    
    # Nemo auditur : La faute a-t-elle été tolérée ?
    was_previously_tolerated: bool = Field(
        False, description="Si True, ne peut justifier une rupture brutale sans préavis"
    )
    formal_notice_sent: bool = Field(
        False, description="Y a-t-il eu une mise en demeure de corriger le tir ?"
    )

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
    variable_costs_saved: float # Électricité, matières premières non achetées...
    gross_margin_rate: float # Taux de marge (ex: 0.40)

class DamagesAssessment(OntologyObject):
    """
    Le chèque à faire à la fin.
    Formule : (Marge mensuelle moyenne) x (Mois de préavis manquants)
    """
    reference_period_months: int = 24 # Sur quelle période calcule-t-on la moyenne ?
    average_monthly_margin: float
    
    # Facteurs d'atténuation (Mitigation)
    mitigation_revenue_found: float = Field(
        0.0, description="CA réalisé avec de nouveaux clients grâce au temps libéré"
    )
    
    estimated_total_damages: float

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from system_state import OntologyObject, Evidence, CommercialRelationship

# --- NOUVELLES CLASSES POUR LES DOCUMENTS FOURNIS ---

class LogisticsCondition(OntologyObject):
    """
    Définit les règles logistiques (ex: Franco de port).
    Issu de : Email_2008-05-29
    """
    condition_name: str = "FRANCO_DE_PORT"
    threshold_amount: float # Le seuil (ex: 1€)
    start_date: datetime
    initial_duration_months: int
    is_tacitly_renewable: bool
    status: str = "ACTIVE" # ACTIVE, REVOKED

class InvoiceEvidence(Evidence):
    """
    Preuve d'activité commerciale à un instant T.
    Issu de : Facture_RB-2009-0315
    """
    invoice_number: str
    amount_ht: float
    currency: str = "EUR"
    
    # Ce champ est crucial : il prouve que les conditions de 2008 s'appliquent encore en 2009
    confirms_conditions: List[str] = [] # ex: ["Franco de port maintenu"]

class FinancialHistory(OntologyObject):
    """
    Conteneur pour l'historique CA.
    Issu de : Extrait_CA_HT_2002_2011
    """
    data_points: List[dict] = Field(default_factory=list) # [{year: 2002, amount: 2.475}, ...]
    total_turnover: float
    average_annual_turnover: float

class RelationshipAttestation(Evidence):
    """
    Attestation humaine confirmant la continuitǸ de la relation.
    """
    period_start_year: int
    period_end_year: int
    notes: List[str] = Field(default_factory=list)

class NonSolicitationCommitment(OntologyObject):
    """
    Engagement contractuel de non-prospection / exclusivitǸ de fait.
    """
    scope: str
    start_date: Optional[datetime] = None
    territory: Optional[str] = None

class PurchaseOrderEvidence(Evidence):
    """
    Commande structurante (utile pour prouver l'activitǸ r��cente).
    """
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
    """
    Tableur de pertes de marge (prǸavis vs dommage).
    """
    scenarios: List[DamagesScenario] = Field(default_factory=list)
    reference_period_label: Optional[str] = None

class UnilateralModification(OntologyObject):
    """
    Modification unilat��rale des conditions (assimilable �� une rupture partielle).
    """
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

class GraphNode(BaseModel):
    """Nœud du graphe encapsulant un objet de l'ontologie."""
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType
    label: str
    
    # Mécanique du Multiverse
    world_tag: WorldTag
    
    # Moteur Bayésien
    probability_score: float = Field(0.5, ge=0.0, le=1.0, description="Probabilité que ce fait soit vrai")
    
    # Contenu Sémantique (Polymorphisme)
    content: Union[
        CommercialRelationship, ExtendedCommercialRelationship, RuptureEvent, GraveFaultEvent,
        CorporateEntity, Contract, DigitalMessage, Evidence, LogisticsCondition, InvoiceEvidence, FinancialHistory,
        RelationshipAttestation, NonSolicitationCommitment, PurchaseOrderEvidence, DamagesTable,
        UnilateralModification, dict
    ]
    
    grounding: Optional[Grounding] = None

class EdgeType(str, Enum):
    CAUSES = "CAUSES"         # A cause B
    PROVES = "PROVES"         # Document X prouve Fait Y
    CONTRADICTS = "CONTRADICTS" # Fait Y contredit Fait Z (Déclenche Alexy)
    INTERPRETS = "INTERPRETS" # Règle X interprète Fait Y

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0

class NarrativeGraph(BaseModel):
    """Le Graphe Orienté Acyclique (DAG)."""
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
    """
    Vecteur de pondération pour les conflits de principes.
    Formula: Weight = (I * W * R)
    """
    principle_name: str
    intensity: float = Field(..., ge=0, le=10, description="Gravité de l'atteinte (I)")
    abstract_weight: float = Field(..., ge=0, le=10, description="Importance hiérarchique (W)")
    reliability: float = Field(..., ge=0, le=1.0, description="Certitude empirique (R)")

    @property
    def score(self) -> float:
        return self.intensity * self.abstract_weight * self.reliability

class RuleType(str, Enum):
    STATIC = "STATIC"   # Hardcoded / JSON Logic
    DYNAMIC = "DYNAMIC" # Python Snippet généré par LLM

class Rule(BaseModel):
    rule_id: str
    type: RuleType
    description: str
    # Pour les règles statiques (ex: JSON Logic)
    logic_payload: Optional[Dict] = None 
    # Pour les règles dynamiques (Code Python brut)
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
    action: str # ex: "EXTRACT_DATE", "CHECK_JURISPRUDENCE"
    target_node_id: Optional[str] = None
    priority: int = Field(50, ge=0, le=100) # 100 = Urgent/Defeater
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list) # Liste des task_ids requis avant exécution

# ==============================================================================
# V. THE SYSTEM STATE (SNAPSHOT FINAL)
# ==============================================================================

class SystemState(BaseModel):
    """
    L'objet unique qui est serialise/deserialise a chaque etape.
    """
    case_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # 1. Le Monde
    graph: NarrativeGraph = Field(default_factory=NarrativeGraph)

    # 2. Les Lois
    rules_registry: List[Rule] = []

    # 3. La Strategie
    plan_queue: List[Task] = []

    # 4. Metriques Globales
    entropy: float = 1.0  # Incertitude globale (1.0 = flou total)
    alexy_notice_months: Optional[float] = None
    alexy_weights: List[AlexyWeight] = Field(default_factory=list)
    interpretations_applied: List[str] = Field(default_factory=list)
    replan_history: List[str] = Field(default_factory=list)
    domain_version: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
# ==============================================================================
# DEMO / TEST UNITAIRE
# ==============================================================================

if __name__ == "__main__":
    # Initialisation du dossier
    state = SystemState(case_id="DOSSIER_RUPTURE_XYZ_2025")
    
    # 1. Création d'un Acteur (Victime)
    victim = CorporateEntity(
        name="PetitFournisseur SAS", 
        role="DEMANDEUR", 
        sector="Agroalimentaire",
        turnover_last_year=150000.0
    )
    
    node_victim = GraphNode(
        type=NodeType.FACT,
        label="Identité Demandeur",
        world_tag=WorldTag.SHARED,
        probability_score=1.0, # Fait certain
        content=victim
    )
    state.graph.add_node(node_victim)

    # 2. Création d'une Rupture (Récit A : Brutale)
    # Le LLM a extrait que la rupture a été faite oralement 1 mois avant
    rupture_brutale = RuptureEvent(
        notification_date=datetime(2023, 9, 1),
        effective_end_date=datetime(2023, 10, 1),
        notice_period_given_months=1.0, # 1 mois seulement
        rupture_type="TOTAL",
        form="ORAL"
    )
    
    node_rupture = GraphNode(
        type=NodeType.FACT,
        label="Rupture Orale",
        world_tag=WorldTag.NARRATIVE_A, # Selon le demandeur
        probability_score=0.85,
        content=rupture_brutale,
        grounding=Grounding(
            source_doc_id="doc_temoignage_employe",
            page_number=1,
            text_span="Le directeur m'a dit de tout arrêter le 1er octobre."
        )
    )
    state.graph.add_node(node_rupture)
    
    # 3. Ajout d'une Règle Statique (L442-1)
    rule_l442 = Rule(
        rule_id="L442-1-II",
        type=RuleType.STATIC,
        description="Si Ancienneté > X alors Préavis = Y",
        logic_payload={"if": {">": ["seniority", 2], "then": 6, "else": 3}}
    )
    state.rules_registry.append(rule_l442)
    
    print(f"System State Initialized for Case: {state.case_id}")
    print(f"Nodes in Graph: {len(state.graph.nodes)}")
    print(f"Rupture Narrative Tag: {state.graph.nodes[node_rupture.node_id].world_tag}")
