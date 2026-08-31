"""S10: does a spec leave CTC enough timesteps for the material (#91)?"""

import pytest

from atr_serving.training.vgsl_geometry import (
    FLOOR_REFUSE,
    LineGeometryError,
    MEDIEVAL_PX_PER_CHAR,
    check_line_geometry,
    frames_per_char,
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


def test_frames_per_char_is_material_over_stride():
    assert frames_per_char(KRAKEN_DEFAULT, px_per_char=16.0) == pytest.approx(2.0)


def test_the_shipped_architectures_warn_but_are_not_refused():
    """Guard against the floor we nearly shipped: an earlier draft of #91 proposed
    refusing anything under 2.0 frames/char, which would have refused run 3 — the
    best model this project has produced (CER 0.1335)."""
    verdict = check_line_geometry(KRAKEN_DEFAULT, MEDIEVAL_PX_PER_CHAR)
    assert verdict.severity == "warn"
    assert verdict.ok is True


def test_aggressive_downsampling_is_refused():
    spec = "[1,64,0,1 Mp2,2 Mp2,2 Mp2,2 Mp2,2 Lbx256]"  # width stride 16
    verdict = check_line_geometry(spec, px_per_char=12.15)
    assert verdict.width_stride == 16
    assert verdict.severity == "refuse"
    assert verdict.ok is False
    assert "fewer frames than characters" in verdict.reason


def test_generous_geometry_is_clean():
    verdict = check_line_geometry("[1,120,0,1 Mp2,2 Cr3,3,64 Lbx256]", px_per_char=12.15)
    assert verdict.severity == "ok"
    assert verdict.frames_per_char == pytest.approx(6.075)


def test_floor_sits_below_what_is_known_to_work():
    """1.69 frames/char is what the working models use; the refusal floor must be
    beneath it or the guard vetoes reality."""
    assert FLOOR_REFUSE < 1.69


def test_unparseable_specs_raise_rather_than_score():
    with pytest.raises(LineGeometryError):
        width_stride("")
    with pytest.raises(LineGeometryError):
        width_stride("[1,64,0,1 Lbx256 Do0.5]")  # no conv, no pooling
    with pytest.raises(LineGeometryError):
        frames_per_char(KRAKEN_PLUS, px_per_char=0)
