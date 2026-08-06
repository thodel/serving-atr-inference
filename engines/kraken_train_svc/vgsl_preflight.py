"""Build the VGSL network and report its shapes — before burning GPU hours.

``ketos train`` will construct the same network, but it does so *after* compiling
the dataset and loading the data module, so a spec problem surfaces late. This
takes seconds and needs no data.

It is worth reading the output for the ``kraken+`` default (see
``docs/TRAINING_PLAN.md`` §3a):

* the stride chain ``(4,2)·(4,2)·(1,2)`` should take height 64 → 4 and width → ⅛,
  so ``S1(1x0)1,3`` folds 4 × 64 = 256 features into the ``Lbx256`` that follows;
* after that reshape the height is 1, so the trailing ``Cr255,1,85,1,1`` — kernel
  height 255 — sees almost nothing but padding;
* kraken appends its own ``O1c<codec+1>`` output layer at training time, so the
  ``85`` here is a hidden width, not the alphabet size.

Run standalone in the training venv::

    .venvs/kraken-train/bin/python -m kraken_train_svc.vgsl_preflight
    .venvs/kraken-train/bin/python -m kraken_train_svc.vgsl_preflight '[1,48,0,1 …]'
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atr_serving.training.contracts import KRAKEN_PLUS_SPEC

__all__ = ["LayerShape", "SpecReport", "describe_spec"]


@dataclass
class LayerShape:
    name: str
    output_shape: tuple | None


@dataclass
class SpecReport:
    spec: str
    #: (batch, channels, height, width) as kraken parses the input block.
    input_shape: tuple | None = None
    output_shape: tuple | None = None
    layers: list[LayerShape] = field(default_factory=list)
    parameters: int = 0

    def render(self) -> str:
        lines = [f"VGSL spec: {self.spec}",
                 f"input  (batch, channels, height, width): {self.input_shape}",
                 f"output (batch, channels, height, width): {self.output_shape}",
                 f"parameters: {self.parameters:,}",
                 "layers:"]
        lines += [f"  {ls.name:<24} -> {ls.output_shape}" for ls in self.layers]
        lines.append(
            "NOTE kraken appends O1c<codec+1> at training time; the last layer above is "
            "NOT the alphabet."
        )
        return "\n".join(lines)


def describe_spec(spec: str = KRAKEN_PLUS_SPEC) -> SpecReport:
    """Instantiate the network and collect its shapes. Raises whatever kraken
    raises on an invalid spec — that is the point."""
    from kraken.lib.vgsl import TorchVGSLModel  # heavy; imported only on demand

    net = TorchVGSLModel(vgsl=spec)
    report = SpecReport(
        spec=spec,
        input_shape=tuple(net.input) if net.input is not None else None,
        output_shape=tuple(net.output) if getattr(net, "output", None) is not None else None,
        parameters=sum(p.numel() for p in net.nn.parameters()),
    )
    for name, module in net.nn.named_children():
        shape = tuple(getattr(module, "output_shape", ()) or ()) or None
        report.layers.append(LayerShape(name=name, output_shape=shape))
    return report


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    spec = argv[0] if argv else KRAKEN_PLUS_SPEC
    try:
        print(describe_spec(spec).render())
    except Exception as exc:  # noqa: BLE001 — this is a CLI; report and exit non-zero
        print(f"VGSL spec is not buildable: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
