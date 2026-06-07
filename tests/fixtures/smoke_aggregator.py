"""End-to-end stub run that produces a tiny RunReport for sanity checking."""
from src.core.instruction_parser import load_specs
from src.core.scenario_matrix import build_scenarios
from src.core.user_simulator import build_user_simulator
from src.core.sut_client import build_sut_client
from src.core.orchestrator import run_dialogue, save_trace
from src.evaluators.aggregator import evaluate_case, aggregate_run


def main() -> None:
    specs = load_specs()
    cases = []
    sut = build_sut_client(use_stub=True, seed=42)
    user_sim = build_user_simulator(use_stub=True, seed=42)
    for spec in specs:
        scenarios, vmap = build_scenarios(spec, seed=42)
        for scenario in scenarios[:2]:
            trace = run_dialogue(
                spec,
                scenario,
                sut,
                user_sim,
                variables=vmap[scenario.id],
                run_id="smoke-aggregator",
                max_turns=8,
                trace_dir="traces/smoke-aggregator",
            )
            case = evaluate_case(spec, scenario, trace, client=None)
            cases.append(case)
    run = aggregate_run("smoke-aggregator", cases, config={"mode": "stub"})
    print("overall_mean:", run.aggregate["overall_mean"])
    print("ci:", run.aggregate["confidence_interval"])
    for case in run.cases:
        print(
            f"  {case.trace.case_id}: total={case.weighted_total} "
            f"dim_scores={[(d.id, d.score) for d in case.dimensions]}"
        )
    print("top_failures:")
    for f in run.aggregate["top_failures"][:3]:
        print(
            f"  - {f['dim_id']} {f['criterion_id']} loss={f['weighted_loss']} ({f['rationale']})"
        )


if __name__ == "__main__":
    main()
