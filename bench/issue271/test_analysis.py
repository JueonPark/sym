"""Score decisions against every measured feasible alternative, including failures."""
import unittest

from bench.issue271.analyze import actual, compare, controls, plan_id


def row(**updates):
    return {"status": "ok", "setup_ms": 5, "first_ms": 3, "close_ms": 1,
            "samples_ms": [2, 1, 4, 2], "lifecycle5_ms": 19, "paced": [], **updates}


class ScoringTests(unittest.TestCase):
    def test_objectives_charge_first_completion_and_cleanup_once(self):
        self.assertEqual(actual(row(), "warm"), 2)
        self.assertEqual(actual(row(), "lifecycle"), 19)
        self.assertIsNone(actual(row(status="oom"), "warm"))

    def test_refusal_and_bad_admission_are_scored_against_feasible_oracle(self):
        cells = {"roomy": row(), "fails": row(status="refused")}
        self.assertEqual(compare(None, cells, "warm")["outcome"], "missed_feasible")
        self.assertEqual(compare("fails", cells, "warm")["outcome"], "infeasible")
        self.assertEqual(compare(None, {"fails": cells["fails"]}, "warm")["outcome"], "no_oracle")
        self.assertEqual(compare("roomy", cells, "warm")["regret_pct"], 0)

    def test_slo_scores_the_deployed_rate_and_includes_queue_time(self):
        r = row(paced=[dict(arrival_ms=[0, 1, 2], start_ms=[0, 4, 5], finish_ms=[1, 5, 6]),
                       dict(arrival_ms=[0, 4, 8], start_ms=[0, 4, 8], finish_ms=[1, 5, 9])])
        self.assertEqual(compare("p", {"p": r}, "slo", 2, 0)["outcome"], "slo_miss")
        self.assertAlmostEqual(compare("p", {"p": r}, "slo", 2, 1)["actual"], 1000 / 3)
        self.assertEqual(compare(None, {"p": r}, "slo", 0.2)["outcome"], "no_oracle")

    def test_ablations_vary_only_the_declared_dimensions(self):
        base = dict(placement="gpu", resident=0, chunk=1, depth=1, reuse=False)
        variants = [base, dict(base, placement="forward"), dict(base, depth=2),
                    dict(base, reuse=True), dict(base, resident=4, placement="cpu")]
        arms = controls(variants, "gpu", 4)
        self.assertEqual(arms["per-transfer"], [plan_id(base)])
        self.assertEqual(set(arms["schedule-only"]), {plan_id(variants[i]) for i in (0, 2)})
        self.assertEqual(set(arms["buffer-only"]), {plan_id(variants[i]) for i in (0, 3)})
        self.assertEqual(set(arms["placement-only"]), {plan_id(variants[i]) for i in (0, 1, 4)})

    def test_resident_compute_can_overlap_first_dma_with_one_slot(self):
        from bench.issue271.measure import workloads
        from bench.issue271.region_cost import describe

        class ConstantProfile:
            def fixed(self, kind):
                return 1

            def at(self, kind, size):
                return 20 if kind == "paired_link_compute" else 10

        work = workloads()[0]
        config = dict(placement="gpu", resident=2, chunk=1, depth=1, reuse=True)
        model = describe(work, config, ConstantProfile())
        self.assertEqual(model["overlap_slowdowns"], [dict(resources=["link", "gpu"], factor=2)])
        self.assertFalse(describe(work, dict(config, resident=0), ConstantProfile())["overlap_slowdowns"])

    def test_published_oracle_and_cpu_recommendation(self):
        from pathlib import Path
        from bench.issue271.analyze import read_evidence, report, score_campaign
        from bench.issue271.measure import workloads
        from bench.issue271.select_plan import recommend

        root = Path(__file__).parent
        data = read_evidence(root / "measurements.json.gz")
        scored = score_campaign(data)
        self.assertEqual(report(data, scored), (root / "RESULTS.md").read_text())
        work = next(w for w in workloads() if w["id"] == "held_final")
        budget = work["budgets_bytes"][0]
        offline = recommend(data["calibration"], work, budget, "warm")
        observed = next(r for r in scored["decisions"] if r["workload"] == work["id"]
                        and r["budget_bytes"] == budget and r["policy"] == "joint" and r["objective"] == "warm")
        self.assertEqual(plan_id(offline["plan"]), observed["oracle_plan"][0])
        self.assertIsNone(recommend([], work, budget, "warm")["plan"])
