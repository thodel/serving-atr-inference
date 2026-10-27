"""scripts/apertus_vq_roundtrip.py: the parts that decide what the test means.

The size rule must be Apertus' own, or the round trip shows a page the model
never sees; the key mapping must find Apertus' tensors under whatever prefix the
checkpoint uses, or the weight comparison compares nothing. torch is not in the
gateway's test environment, so these skip there and run where the script runs.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apertus_vq_roundtrip.py"


@pytest.fixture(scope="module")
def rt():
    spec = importlib.util.spec_from_file_location("apertus_vq_roundtrip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hw,expected", [
    ((3491, 1101), (2496, 784)),    # a Kurrent page from dh-unibe/image-text_kurrent-xix
    ((100, 100), (256, 256)),       # below min_pixels: scaled up
    ((800, 640), (800, 640)),       # inside the budget, already multiples of 16
])
def test_size_rule_is_apertus_own(rt, hw, expected):
    h, w = rt.smart_resize(*hw)
    assert (h, w) == expected
    assert h % 16 == 0 and w % 16 == 0 and h * w <= rt.MAX_PIXELS + 16 * max(h, w)


@pytest.mark.parametrize("key,expected", [
    ("model.vision_tokenizer.encoder.conv_in.weight", "encoder.conv_in.weight"),
    ("vision_tokenizer.quantize.embedding.weight", "quantize.embedding.weight"),
    ("quant_conv.bias", "quant_conv.bias"),
    ("model.vision_tokenizer.post_quant_conv.weight", None),   # decoder side: not Apertus'
    ("model.text_model.layers.0.mlp.weight", None),
])
def test_keys_are_found_under_any_prefix(rt, key, expected):
    assert rt._strip(key) == expected


def test_cer_ignores_layout_and_hyphenation(rt):
    assert rt.cer("Pfar¬\nrer Finsler", "Pfarrer   Finsler") == 0.0
    assert rt.cer("abcd", "abed") == 0.25
    assert rt.cer("", "x") is None
