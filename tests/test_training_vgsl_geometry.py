"""S10: does a spec leave CTC enough timesteps for the material (#91)?"""

import pytest

from atr_serving.training.vgsl_geometry import (
    FLOOR_REFUSE,
    MEDIEVAL_ASPECT_PER_CHAR_P10,
    LineGeometryError,
    aspect_per_char,
    check_line_geometry,
    frames_per_char,
    input_height,
    width_stride,
)

KRAKEN_PLUS = ("[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 "
               "S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]")
KRAKEN_DEFAULT = ("[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 "
                  "Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 "
                  "Do0.1,2 Lbx200 Do]")


def test_both_trained_architectures_reduce_width_by_eight():
    """The two specs this project has actually trained. kraken+ gets there through
    one strided convolution and two poolings, the default through three poolings —
    different routes, same number, which is the point of computing it."""
    assert width_stride(KRAKEN_PLUS) == 8
    assert width_stride(KRAKEN_DEFAULT) == 8


def test_pooling_without_strides_uses_its_window():
    assert width_stride("[1,64,0,1 Mp2,2 Lbx256]") == 2
    assert width_stride("[1,64,0,1 Mp1,2,1,2 Lbx256]") == 2
    # A pooling window that does not move horizontally leaves the width alone.
    assert width_stride("[1,64,0,1 Mp2,1 Lbx256]") == 1


def test_convolution_stride_is_the_fifth_field_not_the_second():
    # Cr4,2,8,4,2 is a 4x2 kernel, 8 filters, y-stride 4, x-stride 2.
    assert width_stride("[1,64,0,1 Cr4,2,8,4,2 Lbx256]") == 2
    # Without explicit strides a convolution does not downsample.
    assert width_stride("[1,64,0,1 Cr3,13,32 Lbx256]") == 1


def test_layers_that_cannot_change_width_are_ignored():
    assert width_stride("[1,64,0,1 Mp2,2 Do0.5 S1(1x0)1,3 Lbx256 Do0.5 O1c10]") == 2


def test_named_layers_are_parsed():
    assert width_stride("[1,64,0,1 C{conv1}r3,3,32,1,2 Mp{pool1}2,2 Lbx256]") == 4


def test_input_height_comes_from_the_input_block():
    assert input_height(KRAKEN_PLUS) == 64
    assert input_height(KRAKEN_DEFAULT) == 120


def test_frames_per_char_scales_with_input_height():
    """The whole point of the rewrite: kraken normalises each crop to the spec's
    height and scales width with it, so the same page yields twice the horizontal
    resolution at 120 px that it does at 64."""
    tall = frames_per_char(KRAKEN_DEFAULT, 0.246)
    short = frames_per_char(KRAKEN_PLUS, 0.246)
    assert tall == pytest.approx(short * 120 / 64)
    assert short == pytest.approx(1.968, abs=1e-3)
    assert tall == pytest.approx(3.69, abs=1e-3)


def test_the_two_trained_architectures_land_where_they_were_measured():
    """run 2 at height 64 is tight; run 3 at height 120 is comfortable — and run 3
    is the better model (CER 0.1335 vs 0.181). The guard must reproduce that
    ordering rather than invert it."""
    short = check_line_geometry(KRAKEN_PLUS, MEDIEVAL_ASPECT_PER_CHAR_P10)
    tall = check_line_geometry(KRAKEN_DEFAULT, MEDIEVAL_ASPECT_PER_CHAR_P10)
    assert short.severity == "warn"
    assert tall.severity == "ok"
    assert short.ok and tall.ok  # neither is refused


def test_aggressive_downsampling_is_refused():
    spec = "[1,48,0,1 Mp2,2 Mp2,2 Mp2,2 Mp2,2 Lbx256]"  # height 48, stride 16
    verdict = check_line_geometry(spec, MEDIEVAL_ASPECT_PER_CHAR_P10)
    assert verdict.width_stride == 16
    assert verdict.severity == "refuse"
    assert verdict.ok is False
    assert "blanks CTC needs" in verdict.reason


def test_floor_sits_below_what_is_known_to_work():
    """1.97 frames/char is what run 2 used and it trained to CER 0.181; the refusal
    floor must stay beneath it or the guard vetoes reality."""
    assert FLOOR_REFUSE < 1.97


def test_aspect_per_char_uses_a_low_percentile_by_default():
    """It is the dense hands that run out of frames, so the default must not be the
    median."""
    samples = [(300.0, 90.0, 30)] * 50 + [(200.0, 100.0, 40)] * 10
    assert aspect_per_char(samples) == pytest.approx(0.05)
    assert aspect_per_char(samples, percentile=50) == pytest.approx(0.1111, abs=1e-4)


def test_aspect_per_char_drops_stub_lines():
    """A two-character line's ratio is margins, not handwriting."""
    with pytest.raises(LineGeometryError):
        aspect_per_char([(100.0, 50.0, 2), (80.0, 40.0, 1)])


def test_unparseable_specs_raise_rather_than_score():
    with pytest.raises(LineGeometryError):
        width_stride("")
    with pytest.raises(LineGeometryError):
        width_stride("[1,64,0,1 Lbx256 Do0.5]")  # no conv, no pooling
    with pytest.raises(LineGeometryError):
        frames_per_char(KRAKEN_PLUS, 0)
    with pytest.raises(LineGeometryError):
        input_height("[Cr3,3,32 Mp2,2]")  # no input block
    with pytest.raises(LineGeometryError):
        input_height("[1,0,0,1 Cr3,3,32 Mp2,2]")  # variable height
