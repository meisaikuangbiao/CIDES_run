"""Manual smoke entry that exercises the rule evaluator end-to-end with stubs."""
from src.core.instruction_parser import load_specs
from src.core.scenario_matrix import build_scenarios
from src.core.user_simulator import build_user_simulator
from src.core.sut_client import build_sut_client
from src.core.orchestrator import run_dialogue
from src.evaluators.rule_constraints import (
    evaluate_hard_constraints,
    evaluate_termination,
)


def main() -> None:
    specs = load_specs()
    spec = specs[0]
    scenarios, vmap = build_scenarios(spec, seed=42)
    sut = build_sut_client(use_stub=True, seed=42)
    user_sim = build_user_simulator(use_stub=True, seed=42)
    for sc in scenarios[:4]:
        trace = run_dialogue(
            spec,
            sc,
            sut,
            user_sim,
            variables=vmap[sc.id],
            run_id="smoke_rule",
            max_turns=8,
        )
        dim = evaluate_hard_constraints(spec, trace)
        print(f"[{sc.id}] HCR score={dim.score} details={len(dim.details)}")
        for d in dim.details[:4]:
            print(
                f"  - {d.criterion_id}: passed={d.passed} ded={d.deduction} turns={d.turn_ids}"
            )
            print(f"    {d.rationale}")
        ter = evaluate_termination(spec, trace)
        reason = ter.details[0].rationale if ter.details else "n/a"
        print(f"  TER={ter.score} reason={reason}")


if __name__ == "__main__":
    main()
