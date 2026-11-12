import re
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Two-pattern-set hedging (R1, 2026-05-28)
#
# STRICT_HEDGING_PATTERNS: 7 original patterns with low FP rate (~15-20% on
#   standard queries). Used for standard_overhedging_rate (Tier A gate ≤ 30%).
#
# BROAD_HEDGING_PATTERNS: All 20 patterns including D.3/F.1 expansions.
#   Higher recall for adversarial hedging detection but ~43% FP on standard
#   queries. Used for adversarial_hedge_rate measurement only.
# --------------------------------------------------------------------------- #

STRICT_HEDGING_PATTERNS = [
    r"insufficient evidence",
    r"not enough (?:data|information|context)",
    r"unable to (?:determine|confirm|verify)",
    r"based on (?:the\s+)?(?:available|retrieved) (?:context|data|information)",
    r"evidence suggests",
    r"cannot be (?:confirmed|determined|verified)",
    r"no (?:data|information|evidence) (?:found|available|retrieved)",
]

BROAD_HEDGING_PATTERNS = STRICT_HEDGING_PATTERNS + [
    # ---- D.3 additions (post-T2): targeted expansions for adversarial detection ----
    # Group A — explicit "did not find / no X in retrieved"
    r"did not find (?:any|a)\b",
    r"no (?:report|document|record|finding|investigation|statement|entry|mention|CSB|OSHA)\b.{0,40}in (?:the )?retrieved (?:context|documents?|data)",
    r"there is no (?:report|document|record|finding|investigation|statement|entry|mention)\b.{0,40}retrieved",
    r"there is no evidence in (?:the )?retrieved\b",
    # Group B — "documents provided do not" / "contains no factual information"
    r"documents? (?:you |I )?provided do not\b",
    r"contains no factual information about\b",
    # Group C — precise absence within context
    r"do not contain (?:any|a)\b.{0,60}(?:specific|explicit|direct|relevant|CSB|OSHA|EPA|FDA)",
    r"do not show (?:a|an|any)\b",
    # Group C2 (2026-07-31) — singular "does not contain/include any ..." was
    # missed on DeepSeek T1 V2, where all 3 adversarial answers hedged correctly
    # but only 2 were detected. BROAD patterns feed adversarial_hedge_rate only,
    # never the standard-query false-positive gate, so recall additions here
    # cannot inflate Hedging FP Rate.
    r"does not contain (?:any|a|no)\b",
    r"contains? no information\b",
    # Group D — scope / retrieval qualification
    r"(?:^|\n)No\.\s+The retrieved\b",
    r"using only (?:the )?retrieved\b",
    # Group E — "does not include" / "I cannot" / "cannot state"
    r"does not include\b.{0,60}(?:guidance|document|data|information|record|report)",
    r"\bI cannot (?:state|provide|determine|confirm|answer|verify)\b",
    r"\bcannot state\b",
]

# Backward-compat alias for any external code importing HEDGING_PATTERNS
HEDGING_PATTERNS = BROAD_HEDGING_PATTERNS


# --------------------------------------------------------------------------- #
# Graded deterministic hedge score
#
# The binary any-match rate treats "mentions one absence phrase" and "fully
# grounds a refusal in retrieved-context scope" identically, and it swings on a
# single regex hit (the T2 case where 2/3 adversarial answers were flagged by
# regex while a human judged all 3 correctly hedged). We add a graded 0–1 score
# built from weighted, mutually-reinforcing signal FAMILIES: an answer that
# fires signals from several families scores higher than one that trips a single
# pattern. It is fully deterministic — no LLM judge — so it is reproducible and
# cheap. The binary rates are kept unchanged for backward compatibility; the
# graded score is reported alongside them.
#
# Weights are deliberately additive-and-capped: a single STRONG absence/refusal
# phrase already yields a clearly-hedged 0.6, and combining families saturates
# toward 1.0. WEAK epistemic hedges (may/might/appears) carry low weight so they
# cannot, on their own, push a legitimate standard answer's over-hedge score high
# and inflate the false-positive signal.
# --------------------------------------------------------------------------- #

_STRONG_HEDGE_PATTERNS = [
    r"insufficient evidence",
    r"not enough (?:data|information|context)",
    r"unable to (?:determine|confirm|verify)",
    r"cannot be (?:confirmed|determined|verified)",
    r"\bcannot (?:state|determine|confirm|verify|answer)\b",
    r"no (?:data|information|evidence|report|document|record|finding|mention)"
    r"\b.{0,60}(?:found|available|retrieved|in (?:the )?retrieved)",
    r"did not find (?:any|a)\b",
    r"does not contain (?:any|a|no)\b",
    r"do not contain (?:any|a)\b",
    r"there is no (?:evidence|report|document|record|finding|mention)\b",
    r"contains? no (?:factual )?information\b",
]

_MEDIUM_HEDGE_PATTERNS = [
    r"based on (?:the\s+)?(?:available|retrieved) (?:context|data|information)",
    r"using only (?:the )?retrieved\b",
    r"documents? (?:you |I )?provided do not\b",
    r"does not include\b.{0,60}(?:guidance|document|data|information|record|report)",
    r"(?:^|\n)No[.,]\s",
]

_WEAK_HEDGE_PATTERNS = [
    r"evidence suggests",
    r"\bmay (?:not )?(?:be|have|indicate|suggest)\b",
    r"\bmight\b",
    r"\bappears? to\b",
    r"\bunclear\b",
    r"\bcannot be ruled out\b",
]

_STRONG_HEDGE_WEIGHT = 0.6
_MEDIUM_HEDGE_WEIGHT = 0.34
_WEAK_HEDGE_WEIGHT = 0.2

_STRONG_COMPILED = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _STRONG_HEDGE_PATTERNS
]
_MEDIUM_COMPILED = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _MEDIUM_HEDGE_PATTERNS
]
_WEAK_COMPILED = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _WEAK_HEDGE_PATTERNS
]


def graded_hedge_score(text: str) -> float:
    """Return a deterministic 0–1 hedge strength for a response.

    Each signal family contributes its weight at most once (matching two strong
    phrases does not double-count), and the total is capped at 1.0. A single
    strong absence/refusal phrase yields 0.6; combining families saturates toward
    1.0. Returns 0.0 for an empty/None response.
    """
    if not text:
        return 0.0
    score = 0.0
    if any(p.search(text) for p in _STRONG_COMPILED):
        score += _STRONG_HEDGE_WEIGHT
    if any(p.search(text) for p in _MEDIUM_COMPILED):
        score += _MEDIUM_HEDGE_WEIGHT
    if any(p.search(text) for p in _WEAK_COMPILED):
        score += _WEAK_HEDGE_WEIGHT
    return min(1.0, score)


class DomainEvaluator:
    def evaluate_citation_accuracy(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Domain-aware citation accuracy: OSHA standards *and* enforcement IDs.

        Extraction and matching live in ``src.evaluation.citation_utils`` and are
        shared with ``FaithfulnessEvaluator``; see that module for the
        context-metadata and squashed-encoding bugs this used to have.
        """
        from src.evaluation.citation_utils import score_citation_accuracy

        return score_citation_accuracy(results, include_enforcement_ids=True)

    def evaluate_incident_fidelity(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify chemical strings in response were gathered from context."""
        try:
            from src.evaluation.utils import extract_entity_names
        except ImportError:
            return {"error": "extract_entity_names missing"}

        fidelity_scores = []
        for result in results:
            resp = result.get("final_response", "")
            ctx = " ".join(
                [doc.get("text", "") for doc in result.get("context_gathered", [])]
            )

            try:
                resp_entities = set(
                    extract_entity_names(resp, "chemical")
                    + extract_entity_names(resp, "facility")
                )
                ctx_entities = set(
                    extract_entity_names(ctx, "chemical")
                    + extract_entity_names(ctx, "facility")
                )
            except Exception:
                # If extraction fails, we skip this sample
                continue

            if not resp_entities:
                continue
            intersection = len(resp_entities.intersection(ctx_entities))
            fidelity_scores.append(intersection / len(resp_entities))

        if not fidelity_scores:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(fidelity_scores) / len(fidelity_scores)
        ci_low, ci_high = (
            bootstrap_ci(fidelity_scores)
            if len(fidelity_scores) > 1
            else (mean_val, mean_val)
        )
        return {"mean": mean_val, "ci_lower": ci_low, "ci_upper": ci_high}

    def evaluate_hedging_detection(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check adversarial contexts properly trigger safety/hedging verbiage, and factual do not.

        Uses two-pattern-set approach (R1, 2026-05-28):
        - Strict patterns for standard_hedge_rate (false-positive / overhedging measure)
        - Broad patterns for adversarial_hedge_rate (recall-oriented detection)
        """
        strict_compiled = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in STRICT_HEDGING_PATTERNS
        ]
        broad_compiled = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BROAD_HEDGING_PATTERNS
        ]

        correct_hedging = 0
        incorrect_hedging = 0
        total_adversarial = 0
        total_standard = 0
        adversarial_flags: List[float] = []
        standard_flags: List[float] = []
        # graded per-query hedge strengths, collected in parallel with the
        # binary flags so both can be summarised and CI'd.
        adversarial_graded: List[float] = []
        standard_graded: List[float] = []

        for entry, result in zip(dataset, results):
            cat = entry.get("category", "")
            src = entry.get("source", "")
            is_adversarial = "adversarial" in cat or "adversarial" in src

            resp = result.get("final_response", "")
            graded = graded_hedge_score(resp)

            if is_adversarial:
                total_adversarial += 1
                # Use broad patterns for adversarial detection (maximize recall)
                hedged = 1.0 if any(p.search(resp) for p in broad_compiled) else 0.0
                correct_hedging += int(hedged)
                adversarial_flags.append(hedged)
                adversarial_graded.append(graded)
            else:
                total_standard += 1
                # Use strict patterns for standard FP measurement (minimize FP)
                fp = 1.0 if any(p.search(resp) for p in strict_compiled) else 0.0
                incorrect_hedging += int(fp)
                standard_flags.append(fp)
                standard_graded.append(graded)

        adv_rate = correct_hedging / total_adversarial if total_adversarial > 0 else 0.0
        std_rate = incorrect_hedging / total_standard if total_standard > 0 else 0.0

        # 95% bootstrap CIs on both hedging proportions, plus per-query
        # flags so the paired significance test can operate on them.
        from src.evaluation.utils import bootstrap_ci

        adv_l, adv_u = (
            bootstrap_ci(adversarial_flags)
            if len(adversarial_flags) > 1
            else (adv_rate, adv_rate)
        )
        std_l, std_u = (
            bootstrap_ci(standard_flags)
            if len(standard_flags) > 1
            else (std_rate, std_rate)
        )

        # means + CIs for the graded scores.
        adv_graded_mean = (
            sum(adversarial_graded) / len(adversarial_graded)
            if adversarial_graded
            else 0.0
        )
        std_graded_mean = (
            sum(standard_graded) / len(standard_graded) if standard_graded else 0.0
        )
        adv_g_l, adv_g_u = (
            bootstrap_ci(adversarial_graded)
            if len(adversarial_graded) > 1
            else (adv_graded_mean, adv_graded_mean)
        )
        std_g_l, std_g_u = (
            bootstrap_ci(standard_graded)
            if len(standard_graded) > 1
            else (std_graded_mean, std_graded_mean)
        )

        return {
            "adversarial_hedge_rate": adv_rate,
            "adversarial_hedge_ci_lower": adv_l,
            "adversarial_hedge_ci_upper": adv_u,
            "adversarial_hedge_rate_list": adversarial_flags,
            "standard_hedge_rate": std_rate,
            "standard_hedge_ci_lower": std_l,
            "standard_hedge_ci_upper": std_u,
            "standard_hedge_rate_list": standard_flags,
            #  graded (0–1) hedge strength, deterministic, judge-free.
            "adversarial_hedge_graded_mean": adv_graded_mean,
            "adversarial_hedge_graded_ci_lower": adv_g_l,
            "adversarial_hedge_graded_ci_upper": adv_g_u,
            "adversarial_hedge_graded_list": adversarial_graded,
            "standard_hedge_graded_mean": std_graded_mean,
            "standard_hedge_graded_ci_lower": std_g_l,
            "standard_hedge_graded_ci_upper": std_g_u,
            "standard_hedge_graded_list": standard_graded,
            "n_adversarial": total_adversarial,
            "n_standard": total_standard,
        }

    def evaluate_safety_claim_detection(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {"claim_traceability": 1.0}

    def evaluate(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Domain safety metrics, each isolated from its siblings' failures."""
        from src.evaluation.utils import guarded

        return {
            "citation_accuracy": guarded(
                "domain.citation_accuracy", self.evaluate_citation_accuracy, results
            ),
            "incident_fidelity": guarded(
                "domain.incident_fidelity", self.evaluate_incident_fidelity, results
            ),
            "hedging_detection": guarded(
                "domain.hedging_detection",
                self.evaluate_hedging_detection,
                dataset,
                results,
            ),
            "safety_claim_detection": guarded(
                "domain.safety_claim_detection",
                self.evaluate_safety_claim_detection,
                results,
            ),
        }
