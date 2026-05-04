"""
conceptnet.py
-------------
ConceptNet API client with local JSON cache.

Provides semantic question templates from relation types (IsA, HasA, etc.)
for use alongside static boolean attributes in QuestionAgent.

Fails gracefully on network errors — returns empty results so the game
falls back to static attribute behavior automatically.

Cache lives at data/conceptnet_cache.json, keyed by lowercase word.
Words already cached are never re-fetched.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

RELATION_TEMPLATES = {
    "IsA":         "Is it a type of {obj}?",
    "HasA":        "Does it have {obj}?",
    "UsedFor":     "Is it used for {obj}?",
    "PartOf":      "Is it part of {obj}?",
    "AtLocation":  "Can you find it at {obj}?",
    "HasProperty": "Is it {obj}?",
    "CapableOf":   "Can it {obj}?",
}

_API_URL = "https://api.conceptnet.io/c/en/{word}?limit=200"
_TOP_PER_RELATION = 5


def _strip_article(label: str) -> str:
    """Remove leading 'a ', 'an ', 'the ' for cleaner question text."""
    for article in ("a ", "an ", "the "):
        if label.startswith(article):
            return label[len(article):]
    return label


def _normalize(label: str) -> str:
    """Lowercase, strip articles, replace spaces with underscores for dict keys."""
    return _strip_article(label.strip().lower()).replace(" ", "_")


def _build_key(relation: str, obj: str) -> str:
    """Canonical question key: 'cn:IsA:bird', 'cn:CapableOf:fly', etc."""
    return f"cn:{relation}:{_normalize(obj)}"


def _build_text(relation: str, obj: str) -> str:
    template = RELATION_TEMPLATES.get(relation, "Does it relate to {obj}?")
    return template.format(obj=_strip_article(obj.strip().lower()))


class ConceptNetClient:
    """
    Fetches and caches ConceptNet edges for candidate words.

    Cache is a JSON file keyed by lowercase word. Each entry is a list of
    {"relation": str, "obj": str, "weight": float} dicts, limited to
    RELATION_TEMPLATES types and top-5 by weight per relation.
    """

    def __init__(self, cache_path: str = "data/conceptnet_cache.json"):
        self._cache_path = Path(cache_path)
        self._cache: dict[str, list[dict]] = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def fetch_relations(self, word: str) -> list[dict]:
        """
        Returns cached relations for word, fetching from ConceptNet if not cached.
        Each entry: {"relation": str, "obj": str, "weight": float}
        Returns [] on network failure.
        """
        key = word.strip().lower()
        if key in self._cache:
            return self._cache[key]

        url = _API_URL.format(word=key)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "neural-20q/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            self._cache[key] = []
            self._save_cache()
            return []

        by_relation: dict[str, list] = {}
        seen: set = set()

        for edge in data.get("edges", []):
            rel_label = edge.get("rel", {}).get("label", "")
            if rel_label not in RELATION_TEMPLATES:
                continue

            start_id = edge.get("start", {}).get("@id", "")
            end_id = edge.get("end", {}).get("@id", "")
            end_label = edge.get("end", {}).get("label", "")

            # Only forward edges: word → object (not "X is a {word}")
            if not start_id.startswith(f"/c/en/{key}"):
                continue
            # Only English targets
            if not end_id.startswith("/c/en/"):
                continue
            if not end_label:
                continue

            obj = end_label.strip()
            dedup = (rel_label, _normalize(obj))
            if dedup in seen:
                continue
            seen.add(dedup)

            weight = edge.get("weight", 1.0)
            by_relation.setdefault(rel_label, []).append(
                {"relation": rel_label, "obj": obj, "weight": weight}
            )

        # Keep top _TOP_PER_RELATION by weight per relation type
        relations: list[dict] = []
        for entries in by_relation.values():
            entries.sort(key=lambda x: x["weight"], reverse=True)
            relations.extend(entries[:_TOP_PER_RELATION])

        self._cache[key] = relations
        self._save_cache()
        return relations

    def prefetch(self, words: list[str]) -> int:
        """Fetches all words not yet cached. Returns count of new fetches."""
        count = 0
        for word in words:
            key = word.strip().lower()
            if key not in self._cache:
                self.fetch_relations(word)
                count += 1
        return count

    def get_question_candidates(self, candidates: list[str]) -> list[tuple[str, str]]:
        """
        Returns (question_key, question_text) pairs for all relations
        found across the given candidates. Each key appears at most once.
        """
        seen_keys: set[str] = set()
        results: list[tuple[str, str]] = []
        for word in candidates:
            for rel in self.fetch_relations(word):
                q_key = _build_key(rel["relation"], rel["obj"])
                if q_key not in seen_keys:
                    seen_keys.add(q_key)
                    results.append((q_key, _build_text(rel["relation"], rel["obj"])))
        return results

    def get_relation_coverage(self, question_key: str, candidates: list[str]) -> float:
        """
        Fraction of candidates that have the relation described by question_key.
        Returns 0.0 if key is malformed or candidates is empty.
        """
        if not candidates:
            return 0.0
        parts = question_key.split(":", 2)
        if len(parts) != 3 or parts[0] != "cn":
            return 0.0
        count = sum(1 for word in candidates if self.has_relation(word, question_key))
        return count / len(candidates)

    def has_relation(self, word: str, question_key: str) -> bool:
        """
        Returns True if word has the relation described by question_key.
        question_key format: "cn:{Relation}:{normalized_obj}"
        """
        parts = question_key.split(":", 2)
        if len(parts) != 3 or parts[0] != "cn":
            return False
        _, relation, obj_norm = parts
        for rel in self.fetch_relations(word):
            if rel["relation"] == relation and _normalize(rel["obj"]) == obj_norm:
                return True
        return False
