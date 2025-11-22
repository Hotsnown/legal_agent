import csv
import os
import re
import unicodedata
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from system_state import (
    SystemState,
    GraphNode,
    NodeType,
    WorldTag,
    Grounding,
    LogisticsCondition,
    InvoiceEvidence,
    FinancialHistory,
    ExtendedCommercialRelationship,
    AnnualTurnover,
    RelationshipAttestation,
    PurchaseOrderEvidence,
    NonSolicitationCommitment,
    DamagesTable,
    DamagesScenario,
    CorporateEntity,
    UnilateralModification,
    EdgeType,
    DigitalMessage,
)


class RegexPerceptionEngine:
    """
    Module d'extraction � base de regex pour le dataset local.
    S'appuie sur les fichiers ./data (docx/csv) pour remplir le graphe.
    """

    def __init__(self):
        self.financial_cache: Dict[int, float] = {}
        self.exclusive_flag: bool = False
        self.margin_rate_hint: Optional[float] = None
        self.relationship_bounds: List[Optional[int]] = [None, None]
        self.relationship_node_id: Optional[str] = None
        self.company_nodes: Dict[str, str] = {}
        self._edges_added: set = set()
        self._data_dir_root: Optional[str] = None
        self._last_doc_id: Optional[str] = None
        self._demo_nodes_injected: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_directory(self, directory: str, state: SystemState):
        if not os.path.isdir(directory):
            print(f"[REGEX] Data folder not found: {directory}")
            return

        # Memorize root dir to validate doc ids later
        self._data_dir_root = os.path.abspath(directory)
        self._last_doc_id = None

        for entry in sorted(os.listdir(directory)):
            path = os.path.join(directory, entry)
            if not os.path.isfile(path):
                continue
            text = self._read_file_as_text(path)
            if text is None:
                continue
            self.ingest_document(os.path.basename(path), text, state, source_path=path)

        # Finalize with last valid doc id (avoid passing directory name)
        self._finalize_relationship(state, source_doc_id=self._last_doc_id)
        self._inject_demo_nodes(state)
        self._synthesize_edges(state)

    def ingest_document(self, doc_name: str, full_text: str, state: SystemState, source_path: Optional[str] = None):
        print(f"--- [REGEX ENGINE] Processing {doc_name} ---")
        lower_name = doc_name.lower()
        normalized = self._normalize(full_text)

        dispatched = False

        if "email_2008-05-29" in lower_name or "franco_de_port" in lower_name:
            self._process_email_franco(doc_name, full_text, normalized, state)
            dispatched = True
        if "facture" in lower_name:
            self._process_invoice(doc_name, full_text, normalized, state)
            dispatched = True
        if "extrait_ca" in lower_name or "historique_ca" in lower_name:
            self._process_financial_table(doc_name, normalized, state)
            dispatched = True
        if "etat_pertes" in lower_name:
            self._process_damages_table(doc_name, full_text, normalized, state)
            dispatched = True
        if "attestation_mme_vivion" in lower_name:
            self._process_attestation_vivion(doc_name, full_text, normalized, state)
            dispatched = True
        if "attestation_ec_boutin" in lower_name:
            self._process_attestation_margin(doc_name, full_text, normalized, state)
            dispatched = True
        if "attestation_m_papon_2014-07-25" in lower_name:
            self._process_attestation_papon(doc_name, full_text, normalized, state)
            dispatched = True
        if "courrier_rboutin" in lower_name:
            self._process_non_solicitation_letter(doc_name, full_text, normalized, state)
            dispatched = True
        if "commande_1947" in lower_name:
            self._process_purchase_order(doc_name, full_text, normalized, state)
            dispatched = True
        if "kbis" in lower_name:
            self._process_kbis(doc_name, full_text, normalized, state)
            dispatched = True
        if "concessions" in lower_name:
            self._process_modification_concessions(doc_name, full_text, normalized, state)
            dispatched = True

        if not dispatched:
            print(f"[WARN] Doc type unknown for regex parsing: {doc_name}")

        current_doc_id = self._canonical_doc_id(source_path or doc_name)
        self._last_doc_id = current_doc_id or self._last_doc_id

        self._finalize_relationship(state, source_doc_id=current_doc_id)

    def _inject_demo_fault_event(self, state: SystemState):
        """Crée le Monde B : Le fournisseur prétend qu'il y a eu des impayés."""
        from system_state import MisconductEvent
        
        fault = MisconductEvent(
            date_of_occurrence=datetime(2011, 5, 20),
            date_of_discovery=datetime(2011, 6, 1),
            type="NON_PAYMENT",
            description="Retard de paiement facture F-2011-05",
            was_previously_tolerated=False 
        )
        
        node = GraphNode(
            type=NodeType.FACT,
            label="Allégation: Impayés répétés",
            world_tag=WorldTag.NARRATIVE_B,  # <--- MONDE B (DÉFENDEUR)
            probability_score=0.60,          # Incertain au début
            content=fault,
            grounding=Grounding(
                source_doc_id="Email_Relance_2011.msg", # Fichier fictif pour la démo
                page_number=1,
                text_span="Nous constatons encore un retard de paiement ce mois-ci.", # <--- CE QUI SERA SURLIGNÉ
                bbox=[100, 200, 500, 250] # Coordonnées fictives pour l'UI
            )
        )
        state.graph.add_node(node)
        # On lie ce fait au noeud "Rupture" via une relation "JUSTIFIES" (nouveau EdgeType ?)
        # Pour l'instant on utilise CAUSES
        # state.graph.add_edge(node.node_id, self.rupture_node_id, EdgeType.CAUSES)

    def _inject_demo_clause_competence(self, state: SystemState):
        """
        Simule l'injection d'une clause de compétence (ex: New York) 
        qui agit comme un Defeater potentiel (bloquant).
        """
        print("   -> [REGEX] Simulated extraction of NY Jurisdiction Clause.")
        
        # On importe les types nécessaires si ce n'est pas déjà fait en haut du fichier
        # (Normalement déjà importés, mais par sécurité :)
        from system_state import GraphNode, NodeType, WorldTag, Grounding

        # Simulation d'une clause trouvée dans un document fictif ou existant
        node = GraphNode(
            type=NodeType.FACT,
            label="Clause Compétence: New York",
            world_tag=WorldTag.SHARED,
            probability_score=0.6, # Incertitude initiale
            content={
                "type": "jurisdiction_clause",
                "jurisdiction": "New York",
                "text": "Toute contestation relative au présent accord sera soumise aux tribunaux de New York."
            },
            grounding=Grounding(
                source_doc_id="Courrier_RBOUTIN_TI_2000-11-09_conditions_commerciales.docx", # On "ancre" cela sur un doc existant pour la démo
                page_number=1,
                text_span="Toute contestation sera soumise aux tribunaux de New York."
            )
        )
        state.graph.add_node(node)

    def _inject_demo_tolerance_evidence(self, state: SystemState):
        """Le Twist : Un email prouve que ces retards étaient acceptés."""
        
        # 1. Créer la preuve
        node = GraphNode(
            type=NodeType.EVIDENCE,
            label="Preuve: Email de Tolérance",
            world_tag=WorldTag.SHARED, # Accepté par tous, c'est une preuve physique
            probability_score=0.99,
            content={"type": "email", "subject": "Pas de souci pour le délai"},
            grounding=Grounding(
                source_doc_id="Email_Accord_Finance.msg",
                page_number=1,
                text_span="Ne vous inquiétez pas pour le retard, on gère ça le mois prochain comme d'habitude."
            )
        )
        state.graph.add_node(node)

        # 2. Mise à jour immédiate du graphe (Simuler le Reasoning Engine ici pour la démo)
        # On cherche le noeud de faute et on le marque comme "Toléré" ou on baisse sa proba
        for n in state.graph.nodes.values():
            if n.label == "Allégation: Impayés répétés":
                n.probability_score = 0.05 # COLLAPSE DU MONDE B
                n.world_tag = WorldTag.NARRATIVE_B # Reste tagué B pour montrer qu'il a perdu
                # On peut ajouter un champ dans content pour expliquer
                if hasattr(n.content, 'was_condoned'):
                    n.content.was_condoned = True

    # ------------------------------------------------------------------ #
    # File helpers
    # ------------------------------------------------------------------ #
    def _read_file_as_text(self, path: str) -> Optional[str]:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".docx":
                with zipfile.ZipFile(path) as z:
                    xml = z.read("word/document.xml").decode("utf-8")
                return re.sub(r"<[^>]+>", " ", xml)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
        except FileNotFoundError:
            print(f"[WARN] File missing: {path}")
            return None
        except KeyError:
            print(f"[WARN] word/document.xml missing in {path}")
            return None

    def _is_valid_source(self, filename: Optional[str]) -> bool:
        if not filename:
            return False
        if self._data_dir_root:
            return os.path.isfile(os.path.join(self._data_dir_root, filename))
        return False

    def _canonical_doc_id(self, source_doc_id: Optional[str]) -> Optional[str]:
        """
        Normalize any path-like hint to a simple filename present under data/.
        Prevents sending `data\\...` paths to the frontend which break /static.
        """
        candidate = os.path.basename(source_doc_id) if source_doc_id else None
        if candidate and "." in candidate and self._is_valid_source(candidate):
            return candidate
        if self._last_doc_id and self._is_valid_source(self._last_doc_id):
            return self._last_doc_id
        if candidate and "." in candidate:
            return candidate
        return None

    def _get_or_create_reason_node(
        self,
        state: SystemState,
        node_id: str,
        label: str,
        prob: float,
        description: str = "",
    ) -> str:
        """
        Create a synthetic reasoning node (RULE_APPLICATION) if absent.
        These nodes materialize intermediate conclusions (ex: relation établie).
        """
        if node_id in state.graph.nodes:
            return node_id
        node = GraphNode(
            node_id=node_id,
            type=NodeType.RULE_APPLICATION,
            label=label,
            world_tag=WorldTag.SHARED,
            probability_score=prob,
            content={
                "type": "synthetic_reasoning",
                "description": description,
            },
        )
        state.graph.add_node(node)
        return node_id

    def _normalize(self, text: str) -> str:
        cleaned = unicodedata.normalize("NFKD", text)
        cleaned = cleaned.replace("\xa0", " ")
        cleaned = cleaned.lower()
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _parse_float(self, value: str) -> Optional[float]:
        try:
            normalized = value.replace(" ", "").replace("\u202f", "")
            normalized = normalized.replace(",", ".")
            return float(normalized)
        except Exception:
            return None

    def _parse_date(self, text: str) -> Optional[datetime]:
        m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%d/%m/%Y")
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    # Parsers
    # ------------------------------------------------------------------ #
    def _process_email_franco(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        threshold_match = re.search(r"franco[^0-9]{0,12}([0-9]+(?:[.,][0-9]+)?)", normalized)
        date = self._parse_date(normalized) or datetime(2008, 5, 29)
        renewal_flag = "tacite" in normalized or "reconduit" in normalized
        duration = 6 if re.search(r"duree[^0-9]+(six|6)", normalized) else 0

        threshold = self._parse_float(threshold_match.group(1)) if threshold_match else None
        if threshold is None:
            print("[WARN] Franco threshold not found in email.")
            return

        condition = LogisticsCondition(
            condition_name="FRANCO_DE_PORT",
            threshold_amount=threshold,
            start_date=date,
            initial_duration_months=duration or 6,
            is_tacitly_renewable=renewal_flag,
        )

        node = GraphNode(
            type=NodeType.FACT,
            label="Condition Franco 2008",
            world_tag=WorldTag.SHARED,
            probability_score=0.98,
            content=condition,
            grounding=Grounding(
                source_doc_id=doc_id,
                page_number=1,
                text_span=threshold_match.group(0) if threshold_match else "",
            ),
        )
        state.graph.add_node(node)
        print(f"  -> Franco detected at {threshold} EUR (renewal={renewal_flag})")

    def _process_invoice(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        id_match = re.search(r"facture\s*n[oir-]*\s*([a-z0-9-]+)", normalized)
        amount_match = re.search(r"total\s*ht\s*:\s*([0-9\s.,]+)", normalized)
        date_match = self._parse_date(normalized)
        maintenance_flag = "franco de port maintenu" in normalized

        if not id_match or not amount_match:
            return

        amount = self._parse_float(amount_match.group(1))
        invoice = InvoiceEvidence(
            title=f"Facture {id_match.group(1).upper()}",
            invoice_number=id_match.group(1).upper(),
            date=date_match,
            amount_ht=amount or 0.0,
            confirms_conditions=["Franco maintenu"] if maintenance_flag else [],
        )

        node = GraphNode(
            type=NodeType.EVIDENCE,
            label=f"Preuve: {invoice.invoice_number}",
            world_tag=WorldTag.SHARED,
            content=invoice,
            grounding=Grounding(
                source_doc_id=doc_id,
                page_number=1,
                text_span=amount_match.group(0),
            ),
        )
        state.graph.add_node(node)
        print(f"  -> Invoice {invoice.invoice_number} captured (HT {invoice.amount_ht})")

    def _process_financial_table(self, doc_id: str, normalized: str, state: SystemState):
        matches = re.findall(r"((?:19|20)\d{2})\s*[,;:]?\s*(\d+[.,]\d+)", normalized)
        if not matches:
            return

        for year_str, amount_str in matches:
            year = int(year_str)
            amount = self._parse_float(amount_str)
            if amount is None:
                continue
            # Les montants sont exprim�s en millions d'euros dans les docs
            self.financial_cache[year] = amount * 1_000_000

        print(f"  -> Financial rows merged ({len(matches)} points)")

    def _process_damages_table(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        scenarios: List[DamagesScenario] = []
        lines = [l for l in raw_text.splitlines() if l.strip() and l[0].isdigit()]
        for line in lines:
            parts = [p for p in line.split(",") if p]
            if len(parts) < 5:
                continue
            notice = self._parse_float(parts[0])
            avg_ca = self._parse_float(parts[1])
            margin = self._parse_float(parts[3])
            loss = self._parse_float(parts[4])
            if notice is None or avg_ca is None or margin is None or loss is None:
                continue
            scenarios.append(
                DamagesScenario(
                    notice_months=int(notice),
                    average_turnover_millions=avg_ca,
                    margin_rate=margin / 100.0 if margin > 1 else margin,
                    estimated_loss=loss,
                    formula="moyenne_CA*(N/12)*taux",
                )
            )

        if not scenarios:
            # Fallback regex on normalized text
            pattern = re.findall(r"(\d{2})\s+[0-9.]+\s+[0-9.]+\s+(\d{2})%\s+([0-9.]+)", normalized)
            for notice, margin, loss in pattern:
                scenarios.append(
                    DamagesScenario(
                        notice_months=int(notice),
                        average_turnover_millions=2.9,
                        margin_rate=float(margin) / 100.0,
                        estimated_loss=float(loss) * 1_000_000,
                    )
                )

        if not scenarios:
            return

        table = DamagesTable(
            scenarios=scenarios,
            reference_period_label="2002-2009",
        )
        node = GraphNode(
            type=NodeType.FACT,
            label="Table pertes de marge",
            world_tag=WorldTag.SHARED,
            probability_score=0.9,
            content=table,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="tableau preavis/pertes"),
        )
        state.graph.add_node(node)
        print(f"  -> Damages table parsed ({len(scenarios)} scenarios)")

    def _process_attestation_vivion(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        span = re.search(r"entre\s+(20\d{2})\s+et\s+(20\d{2})", normalized)
        start = int(span.group(1)) if span else 2007
        end = int(span.group(2)) if span else 2011
        self.relationship_bounds[0] = min(filter(None, [self.relationship_bounds[0], start]), default=start)
        self.relationship_bounds[1] = max(filter(None, [self.relationship_bounds[1], end]), default=end)

        notes: List[str] = []
        if "franco de port" in normalized:
            notes.append("Franco de port applique a partir de 2008 et non denonce avant 2010")
        if "commandes" in normalized:
            notes.append("Commandes regulieres 2007-2011")

        attestation = RelationshipAttestation(
            title="Attestation Elodie Vivion",
            date=self._parse_date(normalized),
            period_start_year=start,
            period_end_year=end,
            notes=notes,
        )
        node = GraphNode(
            type=NodeType.EVIDENCE,
            label="Attestation relation 2007-2011",
            world_tag=WorldTag.SHARED,
            probability_score=0.95,
            content=attestation,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="Relations et volumes 2007-2011"),
        )
        state.graph.add_node(node)
        print("  -> Relationship attestation ingested.")

    def _process_attestation_margin(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        rate_match = re.search(r"(\d{2})\s*%.*marge", normalized)
        if rate_match:
            self.margin_rate_hint = float(rate_match.group(1)) / 100.0
        attestation = RelationshipAttestation(
            title="Attestation marge brute",
            date=self._parse_date(normalized) or datetime(2014, 7, 25),
            period_start_year=1998,
            period_end_year=2009,
            notes=["Taux de marge reconstitue sur series fiscales"],
        )
        node = GraphNode(
            type=NodeType.EVIDENCE,
            label="Attestation EC 1998-2009",
            world_tag=WorldTag.SHARED,
            probability_score=0.9,
            content=attestation,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="ATTESTATION D'EXPERT-COMPTABLE"),
        )
        state.graph.add_node(node)
        print("  -> Margin attestation captured.")

    def _process_attestation_papon(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        start_year = 2009
        end_year = 2011
        period_match = re.search(r"(20\d{2}).*?(20\d{2})", normalized)
        if period_match:
            start_year = int(period_match.group(1))
            end_year = int(period_match.group(2))

        notes = ["Attestation EC sur CA client TFVM 2009-2011", "Controle par rapprochement comptable"]
        if "tfvm" in normalized:
            notes.append("Client TFVM explicite")

        attestation = RelationshipAttestation(
            title="Attestation EC CA TFVM",
            date=self._parse_date(normalized) or datetime(2014, 7, 25),
            period_start_year=start_year,
            period_end_year=end_year,
            notes=notes,
        )
        node = GraphNode(
            type=NodeType.EVIDENCE,
            label="Attestation EC TFVM 2009-2011",
            world_tag=WorldTag.SHARED,
            probability_score=0.9,
            content=attestation,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="Attestation EC CA TFVM"),
        )
        state.graph.add_node(node)
        # Favorise l'etablissement de la relation et une dependance renforcee
        self.relationship_bounds[0] = min(filter(None, [self.relationship_bounds[0], start_year]), default=start_year)
        self.relationship_bounds[1] = max(filter(None, [self.relationship_bounds[1], end_year]), default=end_year)
        self.exclusive_flag = True
        print("  -> Papon attestation (TFVM) ingested.")

    def _process_non_solicitation_letter(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        self.exclusive_flag = True
        commit = NonSolicitationCommitment(
            scope="Non-prospection directe zone Herve Thermique",
            start_date=self._parse_date(normalized) or datetime(2000, 11, 9),
            territory="Zone Herv� Thermique",
        )
        node = GraphNode(
            type=NodeType.FACT,
            label="Engagement de non-prospection",
            world_tag=WorldTag.SHARED,
            probability_score=0.9,
            content=commit,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="non-prospection"),
        )
        state.graph.add_node(node)
        print("  -> Non-prospection commitment detected (exclusivity flag on).")

    def _process_purchase_order(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        order_match = re.search(r"commande\s*n[^\d]*(\d+)", normalized)
        date = self._parse_date(normalized) or datetime(2011, 3, 15)
        franco = None
        franco_match = re.search(r"franco[^0-9]+([0-9]+)", normalized)
        if franco_match:
            franco = self._parse_float(franco_match.group(1))

        order = PurchaseOrderEvidence(
            title="Commande TFVM comparative",
            date=date,
            order_number=order_match.group(1) if order_match else "1947",
            amount_ht=None,
            incoterm="DAP" if "dap" in normalized else None,
            franco_threshold=franco,
            payment_terms="45 jours fin de mois" if "45 jours" in normalized else None,
            comparative_client="TFVM",
        )
        node = GraphNode(
            type=NodeType.EVIDENCE,
            label=f"Commande {order.order_number}",
            world_tag=WorldTag.SHARED,
            probability_score=0.85,
            content=order,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="Commande n"),
        )
        state.graph.add_node(node)
        print(f"  -> Purchase order {order.order_number} ingested.")

    def _process_modification_concessions(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        """
        Detecte la mise en place du franco comme concession/condition imposee unilaterale.
        """
        date = self._parse_date(normalized) or datetime(2008, 5, 29)
        severity = 0.6 if "concessions" in normalized or "acceptons" in normalized else 0.4
        modification = UnilateralModification(
            change_type="CONDITIONS_LOGISTIQUES",
            severity=severity,
            effective_date=date,
            description="Mise en place du franco de port comme concession",
        )
        node = GraphNode(
            type=NodeType.FACT,
            label="Modification unilaterale (conditions franco)",
            world_tag=WorldTag.SHARED,
            probability_score=0.85,
            content=modification,
            grounding=Grounding(source_doc_id=doc_id, page_number=1, text_span="concessions / conditions de franco"),
        )
        state.graph.add_node(node)
        # Modif unilaterale suggere un preavis non respecte -> favorise la dependance
        self.exclusive_flag = True
        print("  -> Unilateral modification (concessions franco) captured.")

    def _process_kbis(self, doc_id: str, raw_text: str, normalized: str, state: SystemState):
        name_match = re.search(r"denomination\s+([a-z0-9\.\s]+?)(?:\s+2\.|\s+forme|$)", normalized)
        siren_match = re.search(r"siren\s+([\d\s]{9,})", normalized)
        sector_match = re.search(r"naf\s*[/ ]ape\s*([0-9a-z]+)", normalized)

        if not siren_match:
            return
        name = name_match.group(1).strip().upper() if name_match else doc_id.split(".")[0].upper()
        siren = siren_match.group(1).replace(" ", "")
        sector = sector_match.group(1).upper() if sector_match else None

        self._upsert_company(state, name=name, siren=siren, sector=sector, source_doc_id=doc_id)

    # ------------------------------------------------------------------ #
    # Relationship synthesis
    # ------------------------------------------------------------------ #
    def _upsert_company(self, state: SystemState, name: str, siren: str, sector: Optional[str], source_doc_id: str):
        if name in self.company_nodes:
            return
        entity = CorporateEntity(name=name, role="TIERS", sector=sector, siren=siren)
        node = GraphNode(
            type=NodeType.FACT,
            label=f"Societe {name}",
            world_tag=WorldTag.SHARED,
            probability_score=0.9,
            content=entity,
            grounding=Grounding(source_doc_id=source_doc_id, page_number=1, text_span="Kbis"),
        )
        state.graph.add_node(node)
        self.company_nodes[name] = node.node_id
        print(f"  -> Company {name} (SIREN {siren}) added.")

    def _finalize_relationship(self, state: SystemState, source_doc_id: Optional[str]):
        if not self.financial_cache:
            return

        doc_id = self._canonical_doc_id(source_doc_id) or self._last_doc_id
        if doc_id != self._last_doc_id and doc_id is not None:
            self._last_doc_id = doc_id
        if doc_id is None:
            return

        years = sorted(self.financial_cache)
        total = sum(self.financial_cache.values())
        avg = total / len(years)
        data_points = [{"year": y, "amount": amt} for y, amt in sorted(self.financial_cache.items())]

        fin_history = FinancialHistory(
            data_points=data_points,
            total_turnover=total,
            average_annual_turnover=avg,
        )

        self._upsert_financial_history_node(state, fin_history, doc_id)
        self._upsert_extended_relationship(state, fin_history, doc_id)

    def _upsert_financial_history_node(self, state: SystemState, fin_history: FinancialHistory, source_doc_id: str):
        target_node = None
        for node in state.graph.nodes.values():
            if isinstance(node.content, FinancialHistory):
                target_node = node
                break
        if target_node:
            target_node.content = fin_history
        else:
            node = GraphNode(
                type=NodeType.FACT,
                label="Historique Financier 1987-2012",
                world_tag=WorldTag.SHARED,
                probability_score=0.95,
                content=fin_history,
                grounding=Grounding(source_doc_id=source_doc_id, page_number=1, text_span="Extrait du CA"),
            )
            state.graph.add_node(node)

    def _upsert_extended_relationship(self, state: SystemState, fin_history: FinancialHistory, source_doc_id: str):
        start_year = years_min = min(item["year"] for item in fin_history.data_points)
        end_year = years_max = max(item["year"] for item in fin_history.data_points)
        if self.relationship_bounds[0]:
            start_year = min(start_year, self.relationship_bounds[0])
        if self.relationship_bounds[1]:
            end_year = max(end_year, self.relationship_bounds[1])

        dependency_rate = 0.25
        if self.exclusive_flag:
            dependency_rate += 0.2
        if fin_history.average_annual_turnover > 2_500_000:
            dependency_rate += 0.1
        dependency_rate = min(dependency_rate, 0.85)

        existing_node = None
        for node in state.graph.nodes.values():
            if isinstance(node.content, ExtendedCommercialRelationship):
                existing_node = node
                break

        characteristics = None
        if existing_node:
            characteristics = existing_node.content.characteristics
            characteristics.is_exclusive = characteristics.is_exclusive or self.exclusive_flag
        else:
            from system_state import RelationshipCharacteristics

            characteristics = RelationshipCharacteristics(is_exclusive=self.exclusive_flag)

        relationship = ExtendedCommercialRelationship(
            start_date=datetime(start_year, 1, 1),
            contract_history_ids=[],
            financial_history=[AnnualTurnover(year=dp["year"], amount=dp["amount"]) for dp in fin_history.data_points],
            average_annual_turnover=fin_history.average_annual_turnover,
            dependency_rate=dependency_rate,
            last_active_year=end_year,
            counterparty_group_name="R. BOUTIN S.A",
            characteristics=characteristics,
        )

        if existing_node:
            existing_node.content = relationship
        else:
            node = GraphNode(
                type=NodeType.FACT,
                label="Relation commerciale TI / R. Boutin",
                world_tag=WorldTag.SHARED,
                probability_score=0.97,
                content=relationship,
                grounding=Grounding(source_doc_id=source_doc_id, page_number=1, text_span="Historique CA"),
            )
            state.graph.add_node(node)
        self.relationship_node_id = node.node_id

    def _inject_demo_nodes(self, state: SystemState):
        """
        Inject synthetic interpretations to showcase demo features (WhatsApp + clause compétence).
        """
        if self._demo_nodes_injected:
            return

        self._demo_nodes_injected = True

        # 1. Message WhatsApp interprété comme écrit
        whatsapp = DigitalMessage(
            title="Whatsapp manager -> fournisseur",
            platform="WHATSAPP",
            sender="Directeur Achats",
            recipient="Fournisseur TI",
        )
        whatsapp_node = GraphNode(
            type=NodeType.EVIDENCE,
            label="Whatsapp du 12/06/2011",
            world_tag=WorldTag.SHARED,
            probability_score=0.74,
            content=whatsapp,
            grounding=Grounding(
                source_doc_id="demo_whatsapp.txt",
                page_number=1,
                text_span="Je confirme par WhatsApp que la relation s'arrêtera fin août.",
            ),
        )
        state.graph.add_node(whatsapp_node)

        # 2. Clause de compétence internationale (New York) -> force replan
        clause_node = GraphNode(
            type=NodeType.FACT,
            label="Clause Compétence New York",
            world_tag=WorldTag.SHARED,
            probability_score=0.6,
            content={
                "type": "synthetic_clause",
                "text": "Clause: toute contestation sera portée devant les tribunaux de New York.",
            },
            grounding=Grounding(
                source_doc_id="demo_competence.pdf",
                page_number=2,
                text_span="Les parties conviennent de la compétence exclusive des tribunaux de New York.",
            ),
        )
        state.graph.add_node(clause_node)
    def _synthesize_edges(self, state: SystemState):
        """
        Cr��e des liens PROVES/CAUSES basiques pour visualiser le graphe m��me
        en l'absence d'inf��rence compl��t��e (evite un canvas vide).
        """
        if not state.graph.nodes:
            return

        # Root = relation commerciale si d��tect��e, sinon premier n��ud
        root_id = self.relationship_node_id or next(iter(state.graph.nodes.keys()))

        for node_id, node in state.graph.nodes.items():
            if node_id == root_id:
                continue
            etype = None
            if node.type == NodeType.EVIDENCE:
                etype = EdgeType.PROVES
            elif node.type == NodeType.FACT:
                etype = EdgeType.CAUSES
            if etype and (node_id, root_id, etype) not in self._edges_added:
                state.graph.add_edge(node_id, root_id, etype)
                self._edges_added.add((node_id, root_id, etype))

        # --- Cha�ne de raisonnement interm�diaire vers "Responsabilit� ?" ---
        reason_established = self._get_or_create_reason_node(
            state,
            node_id="reason_relation_etablie",
            label="Relation commerciale �tablie",
            prob=0.9 if self.relationship_node_id else 0.75,
            description="Synth�se: dur�e, volumes, exclusivit�",
        )
        reason_brutal = self._get_or_create_reason_node(
            state,
            node_id="reason_rupture_brutale",
            label="Rupture brutale (pr�avis insuffisant)",
            prob=0.7,
            description="Pr�avis donn� vs pr�avis raisonnable / modifications unilat�rales",
        )
        reason_liability = self._get_or_create_reason_node(
            state,
            node_id="reason_responsabilite",
            label="Responsabilit� pour rupture brutale ?",
            prob=0.6,
            description="Conclusion provisoire sur la responsabilit�",
        )

        def add_edge(src: str, tgt: str, etype: EdgeType):
            key = (src, tgt, etype)
            if key in self._edges_added:
                return
            state.graph.add_edge(src, tgt, etype)
            self._edges_added.add(key)

        # Faits/preuves soutenant la relation �tablie
        for node_id, node in state.graph.nodes.items():
            content = node.content
            if node_id in {reason_established, reason_brutal, reason_liability}:
                continue
            supports_relation = False
            if isinstance(content, (FinancialHistory, ExtendedCommercialRelationship, RelationshipAttestation)):
                supports_relation = True
            if isinstance(content, (PurchaseOrderEvidence, InvoiceEvidence, NonSolicitationCommitment, LogisticsCondition)):
                supports_relation = True
            if supports_relation:
                add_edge(
                    node_id,
                    reason_established,
                    EdgeType.PROVES if node.type == NodeType.EVIDENCE else EdgeType.CAUSES,
                )

        # Noeud relation commerciale alimente le pivot "relation �tablie"
        if self.relationship_node_id:
            add_edge(self.relationship_node_id, reason_established, EdgeType.CAUSES)

        # Construction de "rupture brutale"
        add_edge(reason_established, reason_brutal, EdgeType.CAUSES)
        for node_id, node in state.graph.nodes.items():
            content = node.content
            if node_id in {reason_established, reason_brutal, reason_liability}:
                continue
            if isinstance(content, UnilateralModification):
                add_edge(node_id, reason_brutal, EdgeType.CAUSES)
            if isinstance(content, LogisticsCondition):
                add_edge(node_id, reason_brutal, EdgeType.PROVES)

        # Th�se finale
        add_edge(reason_established, reason_liability, EdgeType.CAUSES)
        add_edge(reason_brutal, reason_liability, EdgeType.CAUSES)


# ---------------------------------------------------------------------- #
# DEMO
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    state = SystemState(case_id="TEST_DATA_FOLDER")
    engine = RegexPerceptionEngine()
    engine.ingest_directory("data", state)

    print("\n--- SYNTHESIS ---")
    print(f"Nodes: {len(state.graph.nodes)}")
    for node in state.graph.nodes.values():
        if isinstance(node.content, ExtendedCommercialRelationship):
            rel = node.content
            print(
                f"[RELATIONSHIP] start={rel.start_date.date()} years={len(rel.financial_history)} "
                f"avg={rel.average_annual_turnover/1e6:.2f}M dependency={rel.dependency_rate}"
            )
        if isinstance(node.content, LogisticsCondition):
            print(f"[LOGISTICS] {node.content.condition_name} threshold={node.content.threshold_amount}")
        if isinstance(node.content, DamagesTable):
            print(f"[DAMAGES] {len(node.content.scenarios)} scenarios loaded")
