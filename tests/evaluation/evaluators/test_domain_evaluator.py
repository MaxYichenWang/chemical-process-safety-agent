from src.evaluation.evaluators.domain_evaluator import DomainEvaluator


class TestDomainEvaluator:
    def test_hedging_detection_correct(self):
        evaluator = DomainEvaluator()
        dataset = [
            {"category": "adversarial"},
            {"category": "enforcement_factual"}
        ]
        results = [
            {"final_response": "There is insufficient evidence to say."},
            {"final_response": "The fatality happened on Tuesday."}
        ]

        hedge_metrics = evaluator.evaluate_hedging_detection(dataset, results)
        assert hedge_metrics["adversarial_hedge_rate"] == 1.0
        assert hedge_metrics["standard_hedge_rate"] == 0.0

    def test_hedging_detection_incorrect(self):
        evaluator = DomainEvaluator()
        dataset = [
            {"category": "adversarial"},
            {"category": "enforcement_factual"}
        ]
        results = [
            {"final_response": "I know everything did not happen."},
            {"final_response": "Based on retrieved context, I don't know."}
        ]

        hedge_metrics = evaluator.evaluate_hedging_detection(dataset, results)
        assert hedge_metrics["adversarial_hedge_rate"] == 0.0
        assert hedge_metrics["standard_hedge_rate"] == 1.0

    # ------------------------------------------------------------------
    # D.3 extended hedging pattern tests
    # ------------------------------------------------------------------

    def _evaluate(self, adv_response: str, std_response: str = "The fatality occurred on Tuesday."):
        evaluator = DomainEvaluator()
        dataset = [{"category": "adversarial"}, {"category": "enforcement_factual"}]
        results = [{"final_response": adv_response}, {"final_response": std_response}]
        return evaluator.evaluate_hedging_detection(dataset, results)

    def test_d3_did_not_find_any(self):
        """'did not find any' phrase triggers adversarial hedge detection."""
        m = self._evaluate("I did not find any CSB findings about a '2023 Springfield' explosion in the retrieved context.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_no_document_in_retrieved(self):
        """'no document in the retrieved context' triggers hedge detection."""
        m = self._evaluate("There is no document in the retrieved context that identifies a 2024 incident at Gotham Chemical Corporation.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_documents_provided_do_not(self):
        """'documents provided do not include' triggers hedge detection."""
        m = self._evaluate("The documents you provided do not include any report, CSB investigation, or OSHA citation.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_there_is_no_finding_retrieved(self):
        """'there is no finding...retrieved' triggers hedge detection."""
        m = self._evaluate("There is no finding related to the Metropolis Refinery in the retrieved data.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_there_is_no_evidence_in_retrieved(self):
        """'there is no evidence in the retrieved' triggers hedge detection."""
        m = self._evaluate("There is no evidence in the retrieved BP Texas City documents that addressed liquid nitrogen storage.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_do_not_contain_any_specific(self):
        """'do not contain any specific/CSB/OSHA/EPA' triggers hedge detection."""
        m = self._evaluate("The documents do not contain any specific CSB recommendation issued following the Husky Energy incident.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_contains_no_factual_information_about(self):
        """'contains no factual information about' triggers hedge detection."""
        m = self._evaluate("The context contains no factual information about the Arkema Crosby flooding.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_do_not_show_any(self):
        """'do not show any' triggers hedge detection."""
        m = self._evaluate("The retrieved CSB documents do not show any chlorine gas release at the West Fertilizer Company.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_no_followed_by_retrieved_context(self):
        """'No. The retrieved context...' at line start triggers hedge detection."""
        m = self._evaluate("No. The retrieved context contains only OSHA regulatory text and does not include FDA guidelines.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_using_only_retrieved(self):
        """'using only the retrieved' triggers hedge detection."""
        m = self._evaluate("Based solely on the retrieved data and using only retrieved information, I cannot project 2027 incidents.")
        assert m["adversarial_hedge_rate"] == 1.0

    def test_d3_factual_response_no_fp(self):
        """Standard factual responses must NOT trigger hedging."""
        evaluator = DomainEvaluator()
        standard_responses = [
            "The BP Texas City explosion occurred on March 23, 2005, at the isomerization unit.",
            "OSHA cited 29 CFR 1910.119(d) violations with a penalty of $87,430.",
            "The CSB investigation found that inadequate process hazard analysis contributed to the incident.",
            "According to the retrieved context, mechanical integrity failures were identified.",
        ]
        for resp in standard_responses:
            dataset = [{"category": "enforcement_factual"}]
            results = [{"final_response": resp}]
            m = evaluator.evaluate_hedging_detection(dataset, results)
            assert m["standard_hedge_rate"] == 0.0, (
                f"FP triggered for factual response: {resp[:80]!r}"
            )


class TestGradedHedgeScore:
    """graded, deterministic 0–1 hedge strength."""

    def _score(self, text):
        from src.evaluation.evaluators.domain_evaluator import graded_hedge_score

        return graded_hedge_score(text)

    def test_empty_and_none_score_zero(self):
        assert self._score("") == 0.0
        assert self._score(None) == 0.0

    def test_plain_factual_answer_scores_zero(self):
        assert self._score(
            "The BP Texas City explosion occurred on March 23, 2005."
        ) == 0.0

    def test_single_strong_phrase_is_clearly_hedged(self):
        # One strong absence phrase -> 0.6 (unambiguously hedged, no other family)
        assert self._score(
            "I did not find any CSB report about that incident."
        ) == 0.6

    def test_weak_epistemic_alone_stays_low(self):
        # A lone soft hedge must not read as a strong refusal, so it cannot
        # inflate the standard-query over-hedge signal.
        s = self._score("The cause might be related to corrosion.")
        assert 0.0 < s <= 0.2

    def test_combining_families_saturates_higher(self):
        strong_only = self._score("There is no evidence in the retrieved context.")
        combined = self._score(
            "No. There is no evidence in the retrieved context, and using only "
            "the retrieved documents I cannot determine the cause; it might be unrelated."
        )
        assert combined > strong_only
        assert combined <= 1.0

    def test_score_is_bounded_and_deterministic(self):
        text = (
            "No. I did not find any report; the documents do not contain any "
            "record and it is unclear whether it might have occurred."
        )
        first = self._score(text)
        assert first == self._score(text)
        assert 0.0 <= first <= 1.0

    def test_graded_gives_partial_credit_where_binary_misses(self):
        """The motivating T2 case: an answer a strict binary rate would miss
        still earns non-zero graded credit."""
        from src.evaluation.evaluators.domain_evaluator import DomainEvaluator

        evaluator = DomainEvaluator()
        # Three adversarial refusals; the middle one uses only softer language.
        dataset = [{"category": "adversarial"}] * 3
        results = [
            {"final_response": "I did not find any such CSB investigation."},
            {"final_response": "That event appears unclear and might not have occurred."},
            {"final_response": "The retrieved documents do not contain any record of it."},
        ]
        m = evaluator.evaluate_hedging_detection(dataset, results)
        assert m["adversarial_hedge_graded_mean"] > 0.0
        assert len(m["adversarial_hedge_graded_list"]) == 3
        # graded acknowledges the soft-language answer that binary regex may miss
        assert all(x > 0.0 for x in m["adversarial_hedge_graded_list"])

    def test_metric_dict_exposes_graded_keys_with_cis(self):
        from src.evaluation.evaluators.domain_evaluator import DomainEvaluator

        evaluator = DomainEvaluator()
        dataset = [{"category": "adversarial"}, {"category": "enforcement_factual"}]
        results = [
            {"final_response": "There is no evidence in the retrieved context."},
            {"final_response": "The penalty was $87,430."},
        ]
        m = evaluator.evaluate_hedging_detection(dataset, results)
        for key in (
            "adversarial_hedge_graded_mean",
            "adversarial_hedge_graded_ci_lower",
            "adversarial_hedge_graded_ci_upper",
            "standard_hedge_graded_mean",
            "standard_hedge_graded_ci_lower",
            "standard_hedge_graded_ci_upper",
        ):
            assert key in m
        assert m["standard_hedge_graded_mean"] == 0.0
