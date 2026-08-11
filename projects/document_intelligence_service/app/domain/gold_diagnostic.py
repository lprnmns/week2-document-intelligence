"""Framework-independent contracts for the Gold-Aware Diagnostic Mode.

The normal query path has no ground truth and must not assign a root cause.
These small value objects are used only by curated, trusted diagnostic cases.
"""

from dataclasses import dataclass
from enum import StrEnum
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path


class DiagnosticRootCause(StrEnum):
    """Deterministic attribution labels exposed by the mentor demo."""

    PASS = "PASS"
    DATASET_GOLD_INVALID = "DATASET_GOLD_INVALID"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    DENSE_BRANCH_MISS_RECOVERED = "DENSE_BRANCH_MISS_RECOVERED"
    BM25_BRANCH_MISS_RECOVERED = "BM25_BRANCH_MISS_RECOVERED"
    FUSION_LOSS = "FUSION_LOSS"
    RERANKER_LOSS = "RERANKER_LOSS"
    EVIDENCE_SELECTION_LOSS = "EVIDENCE_SELECTION_LOSS"
    PROMPT_CONSTRUCTION_LOSS = "PROMPT_CONSTRUCTION_LOSS"
    EVIDENCE_SAFETY_BLOCK = "EVIDENCE_SAFETY_BLOCK"
    ANSWERABILITY_FALSE_NEGATIVE = "ANSWERABILITY_FALSE_NEGATIVE"
    GENERATION_DEPENDENCY_FAILURE = "GENERATION_DEPENDENCY_FAILURE"
    GENERATION_CLAIM_MISMATCH = "GENERATION_CLAIM_MISMATCH"
    CORRECT_BUT_UNGROUNDED = "CORRECT_BUT_UNGROUNDED"
    NO_ANSWER_CORRECT = "NO_ANSWER_CORRECT"
    SECURITY_POLICY_CORRECT = "SECURITY_POLICY_CORRECT"
    UNATTRIBUTED = "UNATTRIBUTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class DiagnosticVerdict(StrEnum):
    """Comparison outcome; REVIEW_REQUIRED is intentionally not PASS/FAIL."""

    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class GoldLocator:
    """Stable document/page/text locator resolved after normal ingestion."""

    document_key: str
    page: int
    must_contain: str


@dataclass(frozen=True, slots=True)
class GoldClaim:
    """A deterministic expected or forbidden answer claim."""

    claim_id: str
    claim_type: str
    value: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None


@dataclass(frozen=True, slots=True)
class GoldCase:
    """One curated diagnostic case loaded from the committed manifest."""

    case_id: str
    category: str
    question: str
    expected_decision: str
    expected_answer: str
    expected_reason: str | None = None
    expected_claims: tuple[GoldClaim, ...] = ()
    forbidden_claims: tuple[GoldClaim, ...] = ()
    gold_evidence: tuple[GoldLocator, ...] = ()
    forbidden_evidence: tuple[GoldLocator, ...] = ()
    required_qualifiers: tuple[str, ...] = ()
    adversarial_evidence: tuple[GoldLocator, ...] = ()
    scope_document_keys: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a safe case selector projection without runtime source IDs."""

        return {
            "case_id": self.case_id,
            "category": self.category,
            "question": self.question,
            "expected_decision": self.expected_decision,
            "expected_answer": self.expected_answer,
            "expected_reason": self.expected_reason,
            "expected_claims": [_claim_dict(claim) for claim in self.expected_claims],
            "forbidden_claims": [
                _claim_dict(claim) for claim in self.forbidden_claims
            ],
            "gold_evidence": [
                {
                    "document_key": locator.document_key,
                    "page": locator.page,
                    "must_contain": locator.must_contain,
                }
                for locator in self.gold_evidence
            ],
            "required_qualifiers": list(self.required_qualifiers),
            "scope_document_keys": list(self.scope_document_keys),
            "has_trusted_gold": bool(self.gold_evidence),
        }


def load_gold_cases(path: str | Path) -> tuple[GoldCase, ...]:
    """Load and validate a curated JSON manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("gold diagnostic manifest must contain an object")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("gold diagnostic manifest cases must be a list")
    cases = tuple(_case_from_mapping(item) for item in raw_cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("gold diagnostic case IDs must be unique")
    return cases


def manifest_documents(path: str | Path) -> dict[str, str]:
    """Return the manifest's demo document-key to filename mapping."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("gold diagnostic manifest must contain an object")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("gold diagnostic manifest documents must be a list")
    result: dict[str, str] = {}
    for item in raw_documents:
        if not isinstance(item, Mapping):
            raise ValueError("gold diagnostic document must be an object")
        key = item.get("key")
        filename = item.get("filename")
        if not isinstance(key, str) or not key:
            raise ValueError("gold diagnostic document key is required")
        if not isinstance(filename, str) or not filename.lower().endswith(".pdf"):
            raise ValueError("gold diagnostic document filename must be a PDF")
        if key in result:
            raise ValueError(f"duplicate gold document key: {key}")
        result[key] = filename
    return result


def normalize_gold_text(value: str) -> str:
    """Normalize locator/claim text without discarding code punctuation."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def claim_matches(claim: GoldClaim, answer: str) -> bool:
    """Evaluate one high-confidence structured claim deterministically."""

    normalized_answer = normalize_gold_text(answer)
    if claim.claim_type in {"normalized_contains", "date_text", "quoted_term"}:
        value = claim.value
        if not value:
            return False
        return normalize_gold_text(value) in normalized_answer
    if claim.claim_type == "exact_code":
        value = claim.value
        if not value:
            return False
        pattern = rf"(?<![a-z0-9]){re.escape(normalize_gold_text(value))}(?![a-z0-9])"
        return re.search(pattern, normalized_answer) is not None
    if claim.claim_type in {"number", "percentage"}:
        if not claim.value:
            return False
        return re.search(rf"(?<!\d){re.escape(claim.value)}(?!\d)", answer) is not None
    if claim.claim_type == "relation":
        if not claim.subject or not claim.object:
            return False
        return _relation_matches(claim, normalized_answer)
    return False


def _relation_matches(claim: GoldClaim, normalized_answer: str) -> bool:
    """Require an explicit, non-negated relation rather than co-occurrence.

    Gold predicates are small structured labels, not an invitation to build an
    NLP parser.  Clause-local matching keeps the diagnostic deterministic and
    prevents an unrelated subject/object mention from being treated as a
    relation.  Predicate tokens act as a high-confidence cue when present.
    """

    assert claim.subject is not None
    assert claim.object is not None
    subject = normalize_gold_text(claim.subject)
    object_value = normalize_gold_text(claim.object)
    predicate_terms = tuple(
        term
        for term in re.split(r"[_\s-]+", normalize_gold_text(claim.predicate or ""))
        if len(term) >= 3
    )
    clauses = re.split(r"[.!?;:\n]+", normalized_answer)
    negations = ("not", "does not", "doesn't", "never", "no", "hayir", "hayır", "değil", "degil")
    for clause in clauses:
        if subject not in clause or object_value not in clause:
            continue
        if any(negation in clause for negation in negations):
            continue
        if predicate_terms and not any(term in clause for term in predicate_terms):
            continue
        return True
    return False


def compare_claims(
    *,
    expected_claims: Sequence[GoldClaim],
    forbidden_claims: Sequence[GoldClaim],
    answer: str | None,
) -> dict[str, object]:
    """Return structured claim coverage used for verdicts, never similarity."""

    text = answer or ""
    expected = [
        {
            "claim_id": claim.claim_id,
            "type": claim.claim_type,
            "value": claim.value,
            "matched": claim_matches(claim, text),
        }
        for claim in expected_claims
    ]
    forbidden = [
        {
            "claim_id": claim.claim_id,
            "type": claim.claim_type,
            "value": claim.value,
            "matched": claim_matches(claim, text),
        }
        for claim in forbidden_claims
    ]
    return {
        "expected": expected,
        "forbidden": forbidden,
        "expected_claims_passed": sum(1 for item in expected if item["matched"]),
        "expected_claim_count": len(expected),
        "forbidden_claims_found": sum(1 for item in forbidden if item["matched"]),
    }


def compare_decision(
    *,
    expected_decision: str,
    actual_decision: str,
    actual_reason: str | None,
    claims: Mapping[str, object],
    expected_reason: str | None = None,
) -> DiagnosticVerdict:
    """Compare trusted decision/claims without any embedding oracle."""

    expected = expected_decision.upper()
    actual = actual_decision.upper()
    if expected == "SECURITY_POLICY":
        return (
            DiagnosticVerdict.PASS
            if actual == "NO_ANSWER"
            and actual_reason == (expected_reason or "SECURITY_POLICY")
            else DiagnosticVerdict.FAIL
        )
    if expected == "NO_ANSWER":
        return (
            DiagnosticVerdict.PASS
            if actual == "NO_ANSWER"
            and (expected_reason is None or actual_reason == expected_reason)
            else DiagnosticVerdict.FAIL
        )
    if expected != "ANSWERED":
        return DiagnosticVerdict.REVIEW_REQUIRED
    expected_count = _object_int(claims.get("expected_claim_count"))
    passed_count = _object_int(claims.get("expected_claims_passed"))
    forbidden_found = _object_int(claims.get("forbidden_claims_found"))
    return (
        DiagnosticVerdict.PASS
        if actual == "ANSWERED"
        and passed_count == expected_count
        and forbidden_found == 0
        else DiagnosticVerdict.FAIL
    )


def _case_from_mapping(value: object) -> GoldCase:
    if not isinstance(value, Mapping):
        raise ValueError("gold diagnostic case must be an object")

    def required_string(key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"gold diagnostic case field {key!r} is required")
        return item

    return GoldCase(
        case_id=required_string("case_id"),
        category=required_string("category"),
        question=required_string("question"),
        expected_decision=required_string("expected_decision"),
        expected_answer=required_string("expected_answer"),
        expected_reason=(
            value.get("expected_reason")
            if isinstance(value.get("expected_reason"), str)
            else None
        ),
        expected_claims=_claims(value.get("expected_claims")),
        forbidden_claims=_claims(value.get("forbidden_claims")),
        gold_evidence=_locators(value.get("gold_evidence")),
        forbidden_evidence=_locators(value.get("forbidden_evidence")),
        required_qualifiers=_strings(value.get("required_qualifiers")),
        adversarial_evidence=_locators(value.get("adversarial_evidence")),
        scope_document_keys=_strings(value.get("scope_document_keys")),
    )


def _claims(value: object) -> tuple[GoldClaim, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("claims must be a list")
    result: list[GoldClaim] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("claim must be an object")
        claim_id = item.get("id")
        claim_type = item.get("type")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("claim id is required")
        if not isinstance(claim_type, str) or not claim_type:
            raise ValueError("claim type is required")
        result.append(
            GoldClaim(
                claim_id=claim_id,
                claim_type=claim_type,
                value=item.get("value") if isinstance(item.get("value"), str) else None,
                subject=item.get("subject") if isinstance(item.get("subject"), str) else None,
                predicate=item.get("predicate") if isinstance(item.get("predicate"), str) else None,
                object=item.get("object") if isinstance(item.get("object"), str) else None,
            )
        )
    return tuple(result)


def _locators(value: object) -> tuple[GoldLocator, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("gold evidence must be a list")
    result: list[GoldLocator] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("gold locator must be an object")
        document_key = item.get("document_key")
        page = item.get("page")
        must_contain = item.get("must_contain")
        if not isinstance(document_key, str) or not document_key:
            raise ValueError("gold locator document_key is required")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError("gold locator page must be positive")
        if not isinstance(must_contain, str) or not must_contain:
            raise ValueError("gold locator must_contain is required")
        result.append(GoldLocator(document_key, page, must_contain))
    return tuple(result)


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("qualifiers must be a string list")
    return tuple(value)


def _claim_dict(claim: GoldClaim) -> dict[str, str | None]:
    return {
        "id": claim.claim_id,
        "type": claim.claim_type,
        "value": claim.value,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
    }


def _object_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
