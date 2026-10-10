import pytest
from mtp.data.grouping.base import get_grouper
from mtp.data.boundary_cache import build_cache, load_cache
from mtp.eval.layer_probes import features_for_sentence, run_layer_probes


def test_cache_roundtrip_and_guards(tiny_tok, tmp_path):
    path = tmp_path / "cache"
    meta = build_cache(
        ["जा रहा था।", "घर के लिए"],
        tiny_tok,
        get_grouper("hi_rules_v1"),
        path,
        max_length=16,
    )
    assert meta["n_rows"] == 2
    ds, saved = load_cache(path, tiny_tok.name_or_path, 16)
    assert meta == saved and len(ds) == 2
    assert ds[0]["group_id"] == [0, 0, 0]
    with pytest.raises(ValueError):
        load_cache(path, "wrong")
    with pytest.raises(FileExistsError):
        build_cache(["घर"], tiny_tok, get_grouper("hi_rules_v1"), path)


def test_probe_shift_and_frozen_forward(tiny_model_dir):
    pytest.importorskip("sklearn")  # requirements-jai.txt, not the core requirements
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    g = get_grouper("hi_rules_v1")
    layers, y = features_for_sentence(model, tok, g, "मैं जा रहा था।")
    assert y.tolist() == [1, 0, 0]
    assert len(layers) == 3 and all(x.shape == (3, 16) for x in layers)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = run_layer_probes(
        model,
        tok,
        g,
        ["मैं जा रहा था।", "वह घर से आया था"],
        ["बच्चे पार्क में खेल रहे हैं।"],
    )
    assert result["base_model_frozen"] and result["layer"] == [0, 1, 2]
    assert result["n_eval_positions"] == 5
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        run_layer_probes(model, tok, g, ["घर"], ["घर"])


def test_multi_label_probes_share_model_forwards(tiny_model_dir):
    pytest.importorskip("sklearn")  # requirements-jai.txt, not the core requirements
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mtp.eval.layer_probes import run_multi_layer_probes

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    results = run_multi_layer_probes(
        model,
        tok,
        [get_grouper("hi_rules_v1"), get_grouper("words")],
        ["मैं जा रहा था।", "वह घर से आया था"],
        ["बच्चे पार्क में खेल रहे हैं।"],
    )
    hook.remove()
    assert len(calls) == 3 and set(results) == {"hi_rules_v1", "words"}
    assert all(r["shared_forward_label_sets"] == 2 for r in results.values())
