"""JI-10: streaming CPU logistic probes over frozen causal-model layers.

Each feature at token t predicts whether token t+1 starts a group. Both token
positions must be real, unpadded tokens; EOS/BOS do not create labels. Train and
eval texts must be disjoint. Features are not held for the entire corpus.
"""

import numpy as np
from .group_statistics import sentence_digest
from mtp.data.grouping.align import label_tokens


def features_for_sentence(
    model, tokenizer, grouper, text, max_length=128, device="cpu"
):
    import torch

    labels = label_tokens(text, tokenizer, grouper, max_length)
    ids = torch.tensor([labels["input_ids"]], device=device)
    mask = torch.tensor([labels["attention_mask"]], device=device)
    valid = [
        i
        for i in range(len(labels["group_id"]) - 1)
        if labels["group_id"][i] >= 0
        and labels["group_id"][i + 1] >= 0
        and labels["attention_mask"][i]
        and labels["attention_mask"][i + 1]
    ]
    if not valid:
        return [], np.asarray([], dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        result = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
    y = np.asarray([labels["group_start"][i + 1] for i in valid], dtype=np.int64)
    layers = [
        h[0, valid].detach().float().cpu().numpy().copy() for h in result.hidden_states
    ]
    return layers, y


def run_multi_layer_probes(
    model,
    tokenizer,
    groupers,
    train_texts,
    eval_texts,
    device="cpu",
    max_length=128,
    max_positions=64,
    seed=42,
):
    """Share each expensive frozen-model forward across all label/control sets."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score, accuracy_score

    if not groupers or len({g.name for g in groupers}) != len(groupers):
        raise ValueError("require uniquely named groupers")
    if set(train_texts) & set(eval_texts):
        raise ValueError("train/eval sentences overlap")
    if max_positions <= 0 or max_length < 2:
        raise ValueError("invalid probe limits")
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(seed)
    probes = {}
    counts = {g.name: np.zeros(2, dtype=np.int64) for g in groupers}
    predictions = {}
    truth = {g.name: [] for g in groupers}
    n_train = 0

    def extract(text):
        layers, first = features_for_sentence(
            model, tokenizer, groupers[0], text, max_length, device
        )
        if not len(first):
            return [], {}
        reference = label_tokens(text, tokenizer, groupers[0], max_length)
        positions = [
            i
            for i in range(len(reference["group_id"]) - 1)
            if reference["group_id"][i] >= 0
            and reference["group_id"][i + 1] >= 0
            and reference["attention_mask"][i]
            and reference["attention_mask"][i + 1]
        ]
        targets = {groupers[0].name: first}
        for grouper in groupers[1:]:
            labels = label_tokens(text, tokenizer, grouper, max_length)
            if labels["input_ids"] != reference["input_ids"]:
                raise ValueError("groupers changed tokenization")
            targets[grouper.name] = np.asarray(
                [labels["group_start"][i + 1] for i in positions], dtype=np.int64
            )
        return layers, targets

    for text in train_texts:
        layers, targets = extract(text)
        if not targets:
            continue
        if not probes:
            for grouper in groupers:
                probes[grouper.name] = [
                    SGDClassifier(
                        loss="log_loss", alpha=0.0001, random_state=seed, average=True
                    )
                    for _ in layers
                ]
                predictions[grouper.name] = [[] for _ in layers]
        n = len(next(iter(targets.values())))
        selected = rng.choice(n, min(max_positions, n), replace=False)
        for name, y in targets.items():
            for probe, x in zip(probes[name], layers):
                probe.partial_fit(x[selected], y[selected], classes=np.array([0, 1]))
            counts[name] += np.bincount(y[selected], minlength=2)
        n_train += len(selected)
    if not probes:
        raise ValueError("no valid training token pairs")
    for text in eval_texts:
        layers, targets = extract(text)
        if not targets:
            continue
        for name, y in targets.items():
            truth[name].extend(y.tolist())
            for values, probe, x in zip(predictions[name], probes[name], layers):
                values.extend(probe.predict(x).tolist())
    if not any(truth.values()):
        raise ValueError("no valid evaluation token pairs")
    results = {}
    for grouper in groupers:
        name = grouper.name
        y = truth[name]
        majority = int(counts[name].argmax())
        base = [majority] * len(y)
        results[name] = {
            "layer": list(range(len(probes[name]))),
            "f1": [float(f1_score(y, p, zero_division=0)) for p in predictions[name]],
            "acc": [float(accuracy_score(y, p)) for p in predictions[name]],
            "majority_baseline": {
                "class": majority,
                "f1": float(f1_score(y, base, zero_division=0)),
                "acc": float(accuracy_score(y, base)),
            },
            "grouper": name,
            "tokenizer": tokenizer.name_or_path,
            "seed": seed,
            "n_train_sentences": len(train_texts),
            "n_eval_sentences": len(eval_texts),
            "n_train_positions": n_train,
            "n_eval_positions": len(y),
            "train_sentence_sha256": sentence_digest(train_texts),
            "eval_sentence_sha256": sentence_digest(eval_texts),
            "max_length": max_length,
            "max_train_positions_per_sentence": max_positions,
            "probe": "streaming SGD logistic, one epoch",
            "base_model_frozen": True,
            "shared_forward_label_sets": len(groupers),
        }
    return results


def run_layer_probes(
    model,
    tokenizer,
    grouper,
    train_texts,
    eval_texts,
    device="cpu",
    max_length=128,
    max_positions=64,
    seed=42,
):
    return run_multi_layer_probes(
        model,
        tokenizer,
        [grouper],
        train_texts,
        eval_texts,
        device,
        max_length,
        max_positions,
        seed,
    )[grouper.name]
