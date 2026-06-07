from src.core.case_validator import validate_case_matrix


def test_validate_case_matrix_has_expected_cases() -> None:
    result = validate_case_matrix("data/instructions")
    assert result["n_instructions"] >= 3
    assert result["n_cases"] >= 20
    assert any(c["case_id"] == "3__cooperative" for c in result["cases"])
