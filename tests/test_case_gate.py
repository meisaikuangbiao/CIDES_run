from src.core.schemas import (
    CaseReport,
    DialogueTrace,
    DimensionScore,
    ScoreDetail,
)
from src.evaluators.aggregator import assess_case_pass


def _make_case(
    *,
    total: float = 0.9,
    gsr: float = 0.9,
    hcr: float = 0.9,
    hcr_detail_passed: bool = True,
) -> CaseReport:
    return CaseReport(
        trace=DialogueTrace(
            run_id="t",
            case_id="1__cooperative",
            instruction_id="1",
            scenario_id="cooperative",
        ),
        weighted_total=total,
        dimensions=[
            DimensionScore(id="gsr", name="目标完成度", score=gsr, source="llm"),
            DimensionScore(
                id="hcr",
                name="硬约束合规",
                score=hcr,
                source="rule",
                details=[
                    ScoreDetail(
                        criterion_id="hcr.chars",
                        label="字数",
                        passed=hcr_detail_passed,
                        deduction=0.0 if hcr_detail_passed else 1.0,
                    )
                ],
            ),
        ],
    )


def test_assess_case_pass_when_all_good() -> None:
    passed, reasons = assess_case_pass(_make_case())
    assert passed is True
    assert reasons == []


def test_assess_case_pass_when_total_low() -> None:
    passed, reasons = assess_case_pass(_make_case(total=0.5))
    assert passed is False
    assert any("总分" in r for r in reasons)


def test_assess_case_pass_when_hcr_detail_fails() -> None:
    passed, reasons = assess_case_pass(_make_case(hcr_detail_passed=False))
    assert passed is False
    assert any("硬约束" in r for r in reasons)
