"""Training request contracts (#33)."""

import pytest

from atr_serving.training.contracts import (
    KRAKEN_PLUS_SPEC,
    DatasetSpec,
    KrakenTrainParams,
    TrainRequest,
)

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"


def dataset() -> DatasetSpec:
    return DatasetSpec(hf_repo=REPO, train_projects=["GT_Thun-Training_(TEST-DEMO)"])


def test_minimal_request_gets_the_agreed_defaults():
    req = TrainRequest(model_id="kraken-thun-missiven-v1", datasets=[dataset()])
    assert req.engine == "kraken"
    assert req.base_model is None
    assert req.params.spec == KRAKEN_PLUS_SPEC
    assert (req.params.batch_size, req.params.schedule, req.params.lrate) == (256, "1cycle", 1e-4)
    assert req.params.weights_format == "coreml"  # until #36 lands
    assert req.datasets[0].partition == 0.9 and req.datasets[0].seed == 42


@pytest.mark.parametrize("bad", ["Kraken-Thun", "thun model", "", "-leading", "a/b"])
def test_model_id_must_be_a_safe_slug(bad):
    """It becomes a directory name and a registry id."""
    with pytest.raises(ValueError):
        TrainRequest(model_id=bad, dataset=dataset())


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
<<<<<<< HEAD
        TrainRequest(engine="ocr", model_id="x", dataset=dataset())
=======
        TrainRequest(engine="ocr", model_id="x", datasets=[dataset()])
>>>>>>> a8c8009 (fix(#55): CER decomposition — insertions/deletions/substitutions, length_ratio, truncated_cer)


def test_partition_must_be_a_fraction():
    with pytest.raises(ValueError):
        DatasetSpec(hf_repo=REPO, partition=1.0)


def test_json_round_trip():
    req = TrainRequest(model_id="m", datasets=[dataset()],
                       params=KrakenTrainParams(batch_size=64, accumulate_grad_batches=4))
    back = TrainRequest.model_validate_json(req.model_dump_json())
    assert back == req
    assert back.params.effective_batch_size == 256
