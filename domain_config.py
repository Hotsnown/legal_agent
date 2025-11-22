import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "domain_config.json"


@lru_cache(maxsize=1)
def load_domain_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Domain configuration missing at {_CONFIG_PATH}")
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_domain_version() -> str:
    return load_domain_config().get("version", "0.0.0")


def get_ontology_manifest() -> Dict[str, Any]:
    return load_domain_config().get("ontology", {})


def get_static_rules() -> List[Dict[str, Any]]:
    return load_domain_config().get("rules", {}).get("static", [])


def get_dynamic_rule_templates() -> List[Dict[str, Any]]:
    return load_domain_config().get("rules", {}).get("dynamic_templates", [])


def get_planner_blueprint() -> List[Dict[str, Any]]:
    return load_domain_config().get("planner", {}).get("tasks", [])


def get_interpretation_templates() -> List[Dict[str, Any]]:
    return load_domain_config().get("interpretations", [])
