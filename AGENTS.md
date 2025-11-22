SPÉCIFICATION TECHNIQUE : SYSTÈME DE RAISONNEMENT JURIDIQUE NEURO-SYMBOLIQUE
Version : 1.0 - Release Candidate
Architecture : Hybride (LLM + Programmation Probabiliste + GOFAI)
I. Vue d'Ensemble de l'Architecture
Le système est un moteur d'enquête juridique autonome. Il ne se contente pas de traiter du texte, il construit une représentation mentale d'un dossier (Graphe), planifie des actions de vérification (Stratégie), calcule des probabilités de succès (Inférence) et pose des questions pour lever les doutes (Acquisition).
Principes Directeurs
Neuro-Symbolique : Le LLM est utilisé pour la perception (lecture) et la créativité (planification). Le Code (Python/Pyro) est utilisé pour la rigueur logique et le calcul de probabilités.
Multiverse (A vs B) : Le système ne cherche pas "La Vérité", mais modélise la compétition entre deux récits (Demandeur vs Défense).
Transparence Radicale (Glass Box) : Tout verdict doit être traçable jusqu'au pixel près dans le document source.
II. Module 1 : Modélisation des Données (The System State)
Ce module est passif. Il définit la structure de données stricte qui sert de mémoire partagée à tous les autres modules.
1.1. Le Graphe Narratif (The Multiverse)
L'état du dossier est stocké sous forme d'un Graphe Orienté Acyclique (DAG) probabiliste.
Structure du Nœud (Base Node) :
Chaque nœud (Fait, Preuve, Concept) hérite de cette structure :
{
  "id": "uuid_v4",
  "type": "FACT | EVIDENCE | RULE_APP",
  "world_tag": "SHARED | NARRATIVE_A | NARRATIVE_B",
  "probability_distribution": { "type": "Bernoulli", "params": { "p": 0.85 } },
  "grounding": {
    "source_doc_id": "doc_123",
    "bbox": [x1, y1, x2, y2], // Coordonnées visuelles
    "text_span": "Citation exacte..."
  }}
World Tag : Permet la coexistence de faits contradictoires (ex: "Faute Grave" dans le monde A, "Insuffisance Pro" dans le monde B).
1.2. L'Ontologie (Abstraction)
L'ontologie est définie via Pydantic. Elle sert de contrainte de génération pour le LLM.

Actors : Personnes Physiques, Morales (Attributs : Siège, Rôle, Statut).
Events : Faits générateurs (ex: Rupture, Accident, Signature). Tous les événements ont un timestamp normalisé.
Evidence : Supports matériels (Contrats, Emails, Logs).
1.3. Le Registre des Règles (The Ruleset)
Le système gère deux types de normes :

A. Règles Statiques (Hard Rules) :

Définies en JSON Logic ou Python pur.
Immuables (Lois, Articles du Code).
Exemple : if date_notification > date_effet: return "Procedure Irreguliere"
B. Règles Dynamiques (Interpretations) :

Générées à la volée par le LLM en cas de lacune (Interprétation).
Format : Snippets Python exécutés dans une Sandbox.
Cycle de Vie : Génération par Module 3 $\to$ Validation Humaine (Optionnelle) $\to$ Exécution.
1.4. Vecteurs de Pondération (Alexy’s Framework)
Pour les conflits de principes, les données sont stockées sous forme vectorielle :
$$V = {Intensity (I), Weight (W), Reliability (R)}$$
Ces valeurs sont des variables aléatoires (distributions) et non des scalaires fixes.
III. Module 2 : Moteur de Perception (Neuro Layer)
Rôle : Transformer des données non structurées ("Sales") en instances de l'Ontologie (Module 1).
2.1. Pipeline d'Ingestion
OCR Layout-Aware : Utilisation d'un moteur conservant la structure spatiale (Azure DI / Tesseract). Conservation des blocs hiérarchiques (Titres, Articles).
Chunking Sémantique : Découpage intelligent respectant les frontières juridiques (ne pas couper un article de loi au milieu).
2.2. Extraction Contrainte & Ambiguïté
Prompting : Injection du schéma JSON (Ontologie) dans le prompt système.
Détection d'Ambiguïté :
Si le LLM détecte un terme flou critique, il DOIT instancier un AmbiguityNode.
Exemple : Un email dit "C'est d'accord". Le LLM crée :Branche A (Accord sur le prix).
Branche B (Accord de principe sans prix).
Traçabilité (Grounding) : Le LLM doit retourner l'index des tokens ou la bbox source pour chaque champ extrait.
IV. Module 3 : Moteur de Raisonnement & Stratégie (The Logic Core)
Ce module orchestre la résolution du problème. Il est hybride : Planification par LLM, Exécution par Algorithmes.
3.1. Le Planificateur Stratégique (Planner - LLM+P)
Utilise une approche "Code as Policies".

Input : État courant $\mathcal{S}_t$ + Objectif (ex: "Qualifier la rupture").
Génération : Le LLM génère un graphe de dépendances (DAG) de tâches.Priorisation des Defeaters : Le prompt force le LLM à placer les tâches bloquantes (Prescription, Compétence) en début de graphe ("Fail Fast").
Révision (Re-planning) :
Si une tâche échoue (donnée manquante) ou révèle un fait nouveau majeur (ex: "Clause Compromissoire"), l'exécution s'arrête.
Le Planner est rappelé avec l'état d'erreur pour régénérer la suite du graphe.
3.2. Le Moteur d'Inférence (Pyro - Probabilistic Programming)
Implémenté en Pyro (Python). C'est un Réseau Bayésien Dynamique.

Construction du Modèle :

Priors : Les nœuds du Graphe (Module 1) deviennent des distributions (Bernoulli, Beta).
Likelihood (Règles) :
Hard Rules : Agissent comme des masques logiques (multiplication par 0 ou 1).
Soft Rules (Alexy) : Fonction de score : $Score = \frac{I_1 \cdot W_1}{I_2 \cdot W_2}$.
Inférence : Exécution de pyro.infer.SVI ou ImportanceSampling pour obtenir la distribution postérieure du Verdict.
Mécanisme d'Effondrement (Collapse) :

Si $P(Récit_A) < \epsilon$ (seuil d'epsilon), la branche est désactivée ("Pruned") pour économiser du calcul.
V. Module 4 : Moteur d'Acquisition (Active Learning)
Rôle : Poser la "Question Qui Tue" (Smoking Gun).
4.1. Calcul de l'Entropie & EIG
Métrique d'Arrêt : Entropie de Shannon $H(Y)$ sur le verdict final.
Expected Information Gain (EIG) :
Pour chaque variable latente incertaine $X_i$ (ex: "Date de réception", "Intensité Faute") :Simuler la réponse (Monte Carlo).
Mesurer la réduction d'entropie induite.
Sélectionner $X_{best} = \text{argmax}(EIG)$.
4.2. Fonction de Coût et Génération
Cost-Sensitive Logic : $Utility = \frac{EIG}{Cost}$.Coût faible : Question Oui/Non.
Coût élevé : Demande de document externe.
Génération de Question (NLG) : Le LLM traduit la variable $X_{best}$ en langage naturel contextuel.Input : Variable contract_signature_date.
Output : "Le document analysé n'est pas daté. À quelle date précise avez-vous signé le contrat ?"
VI. Module 5 : Orchestration (Workflow Engine)
Rôle : Machine à États Finis (FSM) gérant le cycle de vie et la persistance.
6.1. Machine à États (State Machine)
INIT : Chargement Ontologie, Parsing initial.
PLANNING : Appel Module 3.1 (LLM génère le DAG des tâches).
EXECUTING : Dépilement des tâches du plan. Appels asynchrones au Module 2 (Perception) et Module 3.2 (Pyro).
EVALUATING : Check Entropie.Si $H < Seuil$ $\to$ VERDICT.
Si $H > Seuil$ et Plan vide $\to$ INTERACTING.
Si Erreur Logique $\to$ PLANNING (Re-planification).
INTERACTING : Attente input utilisateur (Appel Module 4).
VERDICT : Génération rapport, sauvegarde.
6.2. Sécurité & Persistance
Persistance : À chaque transition, l'objet SystemState complet est sérialisé (Pickle/JSON) en DB. Permet le "Time Travel".
Watchdogs :
Loop Detector : Hash des états visités pour éviter les cycles.
Budget : Compteur de tokens/coûts API. Arrêt forcé si dépassement.
VII. Module 6 : UX & Couche de Confiance (Trust Layer)
Rôle : Interface "Glass Box" pour l'utilisateur final.
7.1. Argument Graph (Visualisation)
Représentation dynamique du Graphe (Module 1).
Nœuds : Faits (cercles), Règles (losanges), Preuves (carrés).
Liens : Causalité, Inférence, Contradiction.
Code Couleur : Rouge/Bleu pour les narratifs concurrents. Opacité pour la certitude.
7.2. Traçabilité (The Golden Thread)
Interaction "Drill-Down" obligatoire :

Clic sur le Verdict.
Highlight du chemin critique dans le graphe.
Clic sur un Nœud Fait.
Ouverture du PDF Viewer (Panel latéral).
Scroll & Highlight automatique sur le passage extrait (via bbox du Module 2).
7.3. Simulation (What-If Sandbox)
Panneau de contrôle avec "Sliders" liés aux variables probabilistes du Module 3.
Action : Utilisateur modifie "Intensité Préjudice" (0.8 $\to$ 0.3).
Réaction : Appel API $\to$ Recalcul Pyro $\to$ Mise à jour visuelle du Graphe et du Verdict en temps réel.
VIII. Stack Technique Recommandée (2025)
Orchestration : LangGraph (Python).
Inférence Probabiliste : Pyro (basé sur PyTorch).
LLM Framework : LangChain / DSPy (pour l'optimisation des prompts).
Backend : FastAPI (Async).
Frontend : React + React Flow (Graphes) + React-PDF-Highlighter.
Database : PostgreSQL (JSONB pour stocker l'état) + Vector DB (FAISS/Chroma pour la recherche sémantique).
IX. Critères de succès (Definition of Done)
Le système est considéré fonctionnel si :

Il refuse de conclure s'il manque une preuve critique (Entropie haute).
Il identifie correctement des branches contradictoires (Ambiguïté).
Il priorise les motifs de rejet rapide (Defeaters) dans son plan.
L'utilisateur peut tracer chaque affirmation jusqu'au document source.
Une modification manuelle des poids (UX) met à jour le verdict logiquement.
Description synthéthique du domaine, pour les classes, qu'il faudra étendre dans un second temps:
Voici une analyse conceptuelle structurée du système de règles juridiques fourni. Ce système modélise le raisonnement juridique complexe lié à la Rupture Brutale des Relations Commerciales Établies (anciennement art. L.442-6, I, 5° du Code de commerce, désormais recodifié).
L'analyse décompose le système en ses entités fondamentales, sa logique séquentielle et ses mécanismes de défense.
I. Architecture Logique du Système
Le système repose sur un syllogisme juridique strict. Pour que la responsabilité de l'auteur de la rupture soit engagée, une chaîne de conditions cumulatives doit être validée.
Le raisonnement suit cet ordre logique :

Existence : Y a-t-il une "Relation Commerciale Établie" ?
Fait Générateur : Y a-t-il eu une "Rupture" (totale ou partielle) ?
Qualification : Cette rupture est-elle "Brutale" (insuffisance de préavis) ?
Imputabilité : Qui est responsable de la rupture ?
Exceptions : Y a-t-il une cause exonératoire (Faute grave, Force majeure) ?
Dommage : Quel est le préjudice réparable ?
II. Analyse des Entités Principales
1. La Relation Commerciale Établie (Le Socle)
C'est le concept pivot. Sans cette qualification, le régime protecteur ne s'applique pas. Le système définit cette relation par des critères concrets dépassant le simple contrat écrit.

Critères de définition :
Stabilité & Régularité : La relation n'est pas ponctuelle. Elle se caractérise par un flux d'affaires continu et une croyance légitime en sa pérennité.
Durée : Un facteur clé. Plus la relation est longue, plus l'attente de continuité est forte. Le système calcule le point de départ (premier flux significatif) et le point d'arrivée.
Intensité : Le volume d'affaires et la progression du chiffre d'affaires sont des indicateurs de la solidité de la relation.
Problématiques complexes traitées :
Succession de contrats : Une suite de CDD peut constituer une relation établie globale.
Groupes de sociétés : La relation peut exister avec une entité juridique distincte mais appartenant au même groupe économique si une confusion ou une immixtion est prouvée.
Absence d'écrit : Le "courant d'affaires" factuel prime sur le formalisme contractuel.
2. La Rupture (Le Déclencheur)
Le système distingue deux formes de rupture, traitant non seulement l'arrêt total mais aussi l'asphyxie économique progressive.

Rupture Totale : Cessation complète des commandes ou résiliation formelle.
Rupture Partielle (Modification Substantielle) : C'est une subtilité majeure du système. Une rupture brutale peut être caractérisée sans arrêt des relations, par :Une baisse significative et imposée du volume d'affaires.
Un changement unilatéral des tarifs ou conditions de paiement (déréférencement partiel).
Une mise en concurrence soudaine (appel d'offres) qui remet en cause l'exclusivité historique.
3. La Brutalité (Le Vice)
La brutalité ne réside pas dans le fait de rompre, mais dans l'imprévisibilité de la rupture pour la victime.

Le Préavis (L'amortisseur) : Le cœur du contentieux. Le système confronte deux durées :Le Préavis Accordé : La durée réelle laissée au partenaire avant la fin effective.
Le Préavis Raisonnable (Dû) : Calculé selon une formule multifactorielle (Durée de la relation, dépendance économique, secteur d'activité, produits sous marque de distributeur - MDD, temps de reconversion).
Règle de calcul : Si Préavis Accordé < Préavis Raisonnable = BRUTALITÉ.
Formalisme : Le préavis doit être écrit et explicite. Une rupture orale ou tacite est souvent jugée brutale par défaut car le préavis est alors de "zéro".
III. Dynamique de la Preuve et de la Responsabilité
Le système met en scène une bataille probatoire entre le Demandeur (victime présumée) et le Défendeur (auteur de la rupture).
1. L'Imputabilité (Qui a rompu ?)
C'est une zone de contestation critique. Le défendeur tentera souvent de prouver que la rupture n'est pas de son fait.

Rupture à l'initiative de la victime : Si la victime cesse de livrer ou augmente ses prix unilatéralement, l'auteur "apparent" (qui notifie la fin) ne fait que "prendre acte" de la rupture initiée par la victime.
Prise d'acte : Le système analyse qui a réellement rendu la poursuite de la relation impossible.
2. Les Causes d'Exonération (Les "Jokers" du Défendeur)
Même si la rupture est brutale (sans préavis), l'auteur peut échapper à la condamnation s'il prouve :

La Faute Grave (Manquements Graves) :
L'auteur doit prouver une inexécution contractuelle ou une violation légale suffisamment grave (ex: fraude, défauts sécurité, impayés répétés).
Condition critique : Le manquement ne doit pas avoir été "pardonné" ou toléré par le passé (Nemo auditur). Souvent, une mise en demeure préalable est requise pour caractériser la gravité si aucune correction n'est faite.
La Force Majeure : Événement imprévisible, irrésistible et extérieur (très restrictivement admis).
IV. Le Préjudice (La Sanction)
La finalité du système est la réparation du dommage causé par le manque de temps pour se retourner.

Nature du dommage : Perte de la marge brute sur coûts variables (marge sur coûts directs) que la victime aurait réalisée pendant la durée du préavis qui n'a pas été respecté.
Dommage certain et direct : Il ne s'agit pas de compenser la perte du contrat indéfiniment, mais seulement la perte de chance de se reconvertir pendant la période de brutalité.
Atténuation : Si la victime a retrouvé immédiatement un partenaire équivalent (reconversion facile), le préjudice peut être réduit voire annulé.
V. Résumé des Règles Clés (Synthèse)
ConceptRègle PrincipaleException / NuanceRelation ÉtablieFlux d'affaires régulier, stable et significatif créant une attente de continuité.L'absence de contrat écrit n'empêche pas la qualification.StabilitéAncienneté de la relation (souvent > 2 ans ou succession de contrats).La précarité intrinsèque (appels d'offres systématiques) peut exclure la stabilité.RuptureCessation totale ou modification substantielle (prix, volumes) imposée.Une baisse conjoncturelle ou subie par l'auteur (perte de marché) n'est pas une rupture.PréavisDoit tenir compte de la durée de la relation et de la dépendance. Doit être écrit.Un préavis "effectif" (relation continuée de fait) peut parfois compenser l'absence d'écrit.Faute GraveJustifie la rupture immédiate sans indemnité.La faute ne doit pas être un prétexte (doit être réelle, prouvée et non tolérée auparavant).
Conclusion
Ce système expert juridique vise à protéger la partie faible ou dépendante dans une relation commerciale contre l'arbitraire d'une fin soudaine. Il repose sur une analyse économique de la relation (flux, durée, dépendance) plutôt que sur la seule lettre du contrat, imposant un devoir de loyauté et de prévisibilité temporelle lors de la désunion.