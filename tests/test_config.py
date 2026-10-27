"""Released configurations and the hyperparameter mapping of the search (Section 3.3, Appendix A)."""

from pathlib import Path

import pytest

from thinvessel.config import Study, all_studies, apply_params, load_config, resolve_paths

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def config_path(s: Study) -> Path:
    return CONFIGS / s.regime / s.dataset / f"{s.model}.yaml"


@pytest.mark.parametrize("study", all_studies(), ids=lambda s: s.relpath)
def test_every_config_loads(study: Study) -> None:
    cfg = load_config(config_path(study))
    assert cfg["model_name"] == study.model and cfg["dataset_name"] == study.dataset
    assert cfg["seed"] == 42 and isinstance(cfg["selected_trial"], int)
    tp = cfg["training_params"]
    assert tp["optimizer"] == "AdamW" and tp["max_epochs"] == 1000
    assert tp["patience"] * tp.get("check_val_every_n_epoch", 1) == 50
    assert list(cfg["search_space"])[:3] == ["optimizer", "lr", "weight_decay"]
    assert Study.from_config_path(config_path(study)) == study


def test_there_are_36_studies() -> None:
    assert len(all_studies()) == 36 and len({s.relpath for s in all_studies()}) == 36


def test_resolve_paths(tmp_path) -> None:
    path = config_path(Study("finetuned", "drive", "medsam"))
    cfg = resolve_paths(load_config(path), path, tmp_path / "data", tmp_path / "weights")
    dp = cfg["dataset_params"]
    assert dp["train_image_dir"] == str(tmp_path / "data" / "DRIVE" / "training" / "images")
    assert Path(dp["val_csv"]).read_text().split() == ["37_training", "38_training", "39_training", "40_training"]
    assert cfg["model_params"]["pretrained_weights_path"] == str(tmp_path / "weights" / "medsam_vit_b.pth")


def test_apply_params_mapping() -> None:
    cfg = load_config(config_path(Study("compact", "drive", "medsam")))
    out = apply_params(
        cfg,
        {
            "optimizer": "torch.optim.SGD",
            "lr": 0.01,
            "weight_decay": 0.001,
            "patience": 99,
            "batch_size": 8,
            "patch_size": 8,
            "embed_dim": 256,
        },
    )
    assert out["training_params"]["lr"] == 0.01 and out["dataset_params"]["batch_size"] == 8
    vit = out["model_params"]["medsam_params"]["vit_b_params"]
    assert vit["patch_size"] == 8 and vit["embed_dim"] == 256
    assert out["model_params"]["medsam_params"]["prompt_encoder_params"]["image_embedding_size"] == [32, 32]
    # optimizer and patience are sampled but never applied (see README, "Known deviations")
    assert out["training_params"]["optimizer"] == "AdamW"
    assert out["training_params"]["patience"] == cfg["training_params"]["patience"]
    assert "search_space" not in out and cfg["model_params"]["medsam_params"]["vit_b_params"]["patch_size"] == 4


def test_apply_params_cnn_keys() -> None:
    cfg = load_config(config_path(Study("cnn", "drive", "lwnet")))
    out = apply_params(cfg, {"layers": "[16, 32, 64]"})
    assert out["model_params"]["layers"] == "[16, 32, 64]"  # kept as sampled; LW-Net parses it
