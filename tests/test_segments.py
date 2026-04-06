"""Tests for token segmentation (Step 0d — segmentation)."""

from qgre.segments import (
    CLOSE_ANGLE,
    CLOSE_SLASH,
    OPEN_ANGLE,
    STEP_NUM_TOKENS,
    STEP_TOKEN,
    THINK_END,
    THINK_START,
    _hif_json_segmenter_impl,
    make_hif_json_segmenter,
    segment_completion,
)


def test_segment_known_sequence(known_token_ids):
    """Hand-crafted tokens → correct region labels."""
    regions = segment_completion(known_token_ids)

    # <think> block: tokens 0-4
    assert regions[0] == "THINK"  # THINK_START
    assert regions[1] == "THINK"
    assert regions[2] == "THINK"
    assert regions[3] == "THINK"
    assert regions[4] == "THINK"  # THINK_END

    # <step1_extraction> opening tag: tokens 5-9 → FORMAT
    assert regions[5] == "FORMAT"  # <
    assert regions[6] == "FORMAT"  # step
    assert regions[7] == "FORMAT"  # 1
    assert regions[8] == "FORMAT"  # _extraction
    assert regions[9] == "FORMAT"  # >

    # step 1 content: tokens 10-12 → STEP_1
    assert regions[10] == "STEP_1"
    assert regions[11] == "STEP_1"
    assert regions[12] == "STEP_1"

    # </step1_extraction> closing tag: tokens 13-17 → FORMAT
    assert regions[13] == "FORMAT"
    assert regions[14] == "FORMAT"
    assert regions[15] == "FORMAT"
    assert regions[16] == "FORMAT"
    assert regions[17] == "FORMAT"

    # <step2_shared_context> opening tag: tokens 18-23 → FORMAT
    assert regions[18] == "FORMAT"

    # step 2 content: tokens 24-25 → STEP_2
    assert regions[24] == "STEP_2"
    assert regions[25] == "STEP_2"


def test_segment_no_think_block():
    """nothink template: no THINK tokens → no THINK regions."""
    tokens = [
        OPEN_ANGLE,
        STEP_TOKEN,
        16,
        94842,
        CLOSE_ANGLE,  # <step1_extraction>
        100,
        101,  # content
        CLOSE_SLASH,
        STEP_TOKEN,
        16,
        94842,
        CLOSE_ANGLE,  # </step1_extraction>
    ]
    regions = segment_completion(tokens)

    assert "THINK" not in regions
    assert "STEP_1" in regions
    assert regions[5] == "STEP_1"
    assert regions[6] == "STEP_1"


def test_segment_malformed_tags():
    """Missing closing tag → region extends to end as current type."""
    tokens = [
        OPEN_ANGLE,
        STEP_TOKEN,
        16,
        94842,
        CLOSE_ANGLE,  # <step1_extraction>
        100,
        101,
        102,
        103,  # content with no closing tag
    ]
    regions = segment_completion(tokens)

    # Content tokens should all be STEP_1 since no closing tag resets to OTHER
    assert regions[5] == "STEP_1"
    assert regions[6] == "STEP_1"
    assert regions[7] == "STEP_1"
    assert regions[8] == "STEP_1"


def test_segment_all_steps():
    """Full completion with steps 1-N → N STEP regions."""
    num_steps = len(STEP_NUM_TOKENS)
    tokens = []
    for step_num_tok, step_num in STEP_NUM_TOKENS.items():
        tokens.extend([OPEN_ANGLE, STEP_TOKEN, step_num_tok, 9999, CLOSE_ANGLE])
        tokens.extend([100 + step_num, 200 + step_num])
        tokens.extend([CLOSE_SLASH, STEP_TOKEN, step_num_tok, 9999, CLOSE_ANGLE])

    regions = segment_completion(tokens)

    for step_num in range(1, num_steps + 1):
        assert f"STEP_{step_num}" in regions, f"STEP_{step_num} not found"

    assert regions.count("FORMAT") == num_steps * 10


def test_segment_empty_input():
    """Empty token list → empty regions."""
    assert segment_completion([]) == []


def test_segment_no_step_tags():
    """Tokens with no step structure → all OTHER."""
    regions = segment_completion([100, 101, 102, 103])
    assert all(r == "OTHER" for r in regions)


# --- HIF JSON segmenter tests ---


class MockHIFTokenizer:
    """Mock tokenizer that simulates Qwen3 decode behavior for HIF JSON."""

    def __init__(self):
        self._vocab = {
            THINK_START: "<think>",
            THINK_END: "</think>",
            1000: "thinking about the problem",
            1001: '{"network-type": "mesh",',
            1002: ' "metadata": {"v": 1},',
            1003: ' "nodes": [{"id": "A"}, {"id": "B"}],',
            1004: ' "edges": [{"id": "e1"}],',
            1005: ' "incidences": [{"node": "A", "edge": "e1"}],',
            1006: ' "scan-results": {"score": 0.8},',
            1007: ' "hamiltonian-score": 0.75}',
            1008: "some plain text",
            1009: '{"nodes": [',
            1010: '"value with edges inside"',
        }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._vocab.get(i, f"[{i}]") for i in ids)


def test_hif_segmenter_all_sections():
    """HIF JSON with all sections → correct region assignment."""
    tok = MockHIFTokenizer()
    ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007]
    regions = _hif_json_segmenter_impl(ids, tok)

    assert regions[0] == "STEP_1"  # network-type
    assert regions[1] == "STEP_1"  # metadata
    assert regions[2] == "STEP_2"  # nodes
    assert regions[3] == "STEP_3"  # edges
    assert regions[4] == "STEP_3"  # incidences
    assert regions[5] == "STEP_4"  # scan-results
    assert regions[6] == "STEP_4"  # hamiltonian-score


def test_hif_segmenter_think_block():
    """Think block tokens get THINK, not overwritten by JSON sections."""
    tok = MockHIFTokenizer()
    ids = [THINK_START, 1000, THINK_END, 1001, 1003]
    regions = _hif_json_segmenter_impl(ids, tok)

    assert regions[0] == "THINK"
    assert regions[1] == "THINK"
    assert regions[2] == "THINK"
    assert regions[3] == "STEP_1"  # network-type
    assert regions[4] == "STEP_2"  # nodes


def test_hif_segmenter_empty():
    """Empty input → empty output."""
    tok = MockHIFTokenizer()
    assert _hif_json_segmenter_impl([], tok) == []


def test_hif_segmenter_no_json_sections():
    """Plain text with no JSON keys → all STEP_5 (default)."""
    tok = MockHIFTokenizer()
    ids = [1008, 1008, 1008]
    regions = _hif_json_segmenter_impl(ids, tok)
    assert all(r == "STEP_5" for r in regions)


def test_hif_segmenter_partial_json():
    """Partial JSON (only nodes, no other sections) → STEP_2 + STEP_5."""
    tok = MockHIFTokenizer()
    ids = [1008, 1009, 1008]  # text, then "nodes": [, then text
    regions = _hif_json_segmenter_impl(ids, tok)

    assert regions[0] == "STEP_5"  # before nodes
    assert regions[1] == "STEP_2"  # nodes
    assert regions[2] == "STEP_2"  # still in nodes section (no next key)


def test_hif_segmenter_make_factory():
    """make_hif_json_segmenter returns a callable matching Segmenter protocol."""
    tok = MockHIFTokenizer()
    segmenter = make_hif_json_segmenter(tok)

    ids = [THINK_START, 1000, THINK_END, 1001, 1003]
    regions = segmenter(ids)

    assert regions[0] == "THINK"
    assert regions[3] == "STEP_1"
    assert regions[4] == "STEP_2"


def test_hif_segmenter_decode_failure():
    """If tokenizer decode fails, fall back to think-annotated + STEP_5 default."""

    class BrokenTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            raise RuntimeError("decode failed")

    tok = BrokenTokenizer()
    ids = [THINK_START, 100, THINK_END, 200, 300]
    regions = _hif_json_segmenter_impl(ids, tok)

    assert regions[0] == "THINK"
    assert regions[1] == "THINK"
    assert regions[2] == "THINK"
    assert regions[3] == "STEP_5"  # fallback
    assert regions[4] == "STEP_5"
