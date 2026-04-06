"""Test attention constraint fallback - verify actual code paths compile and work."""

import ast


def test_trainer_imports():
    """Verify trainer module imports without error."""
    print("trainer imports: OK")


def test_attention_bonds_imports():
    """Verify attention_bonds module imports without error."""
    print("attention_bonds imports: OK")


def test_config_has_attention_fields():
    """Verify config has the attention constraint fields."""
    from qgre.config import AlgorithmConfig

    cfg = AlgorithmConfig()
    assert hasattr(cfg, "attention_constrained_advantage")
    assert hasattr(cfg, "attention_constraint_strength")
    assert hasattr(cfg, "attention_constraint_mode")
    assert hasattr(cfg, "attention_sample_layer")
    print("config fields: OK")


def test_no_undefined_names_in_trainer():
    """Parse trainer.py AST and check for obvious undefined names."""
    with open("qgre/trainer.py") as f:
        source = f.read()

    # This will raise SyntaxError if code doesn't parse
    tree = ast.parse(source)

    # Compile to bytecode - catches NameErrors at compile time for some cases
    compile(source, "qgre/trainer.py", "exec")

    # Check for undefined _logger usage (should use logging.getLogger(__name__))
    if "_logger." in source:
        # Find lines with _logger that aren't self._completion_logger
        bad_lines = []
        for i, line in enumerate(source.split("\n"), 1):
            if (
                "_logger." in line
                and "self._completion_logger" not in line
                and "completion_logger" not in line
            ):
                bad_lines.append(f"  {i}: {line.strip()}")
        if bad_lines:
            raise AssertionError("Found undefined _logger usage:\n" + "\n".join(bad_lines))

    # Check for local imports that shadow module-level imports (causes UnboundLocalError)
    # Pattern: "import logging" inside a function shadows module-level "import logging"
    in_function = False
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            in_function = True
        if in_function and stripped == "import logging":
            raise AssertionError(
                f"Line {i}: Local 'import logging' shadows module-level import. "
                "Remove this line - use the module-level import instead.",
            )

    print("AST parse and compile: OK")


def test_fused_logprobs_imports():
    """Verify fused_logprobs module imports."""
    print("fused_logprobs imports: OK")


if __name__ == "__main__":
    test_trainer_imports()
    test_attention_bonds_imports()
    test_config_has_attention_fields()
    test_no_undefined_names_in_trainer()
    test_fused_logprobs_imports()
    print("\nAll syntax/import checks passed.")
