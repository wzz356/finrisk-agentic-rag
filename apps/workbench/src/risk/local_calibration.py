"""Small character-ngram classifier trained only on the declared development split."""

from __future__ import annotations

import json
import re
import unicodedata
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DIMENSION = 2048
STATES = ("risk", "no_clear_risk", "insufficient_information")


def features(text: str) -> NDArray[np.float32]:
    value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())
    result = np.zeros(DIMENSION, dtype=np.float32)
    for size in (2, 3):
        for start in range(max(0, len(value) - size + 1)):
            index = 1 + zlib.crc32(value[start : start + size].encode()) % (DIMENSION - 1)
            result[index] += 1
    np.log1p(result, out=result)
    norm = np.linalg.norm(result)
    if norm:
        result /= norm
    result[0] = 1
    return result


class LocalRiskCalibration:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.weights = data["weights"].astype(np.float32)
            self.labels = tuple(str(x) for x in data["labels"])
            self.example_features = (
                data["example_features"]
                if "example_features" in data
                else np.empty((0, DIMENSION), dtype=np.float32)
            )
            self.examples = (
                [json.loads(str(v)) for v in data["examples"]] if "examples" in data else []
            )
            self.version = (
                str(data["version"].item())
                if "version" in data
                else "char-ngram-development400-reviewed-v2"
            )
        if self.weights.shape != (DIMENSION, 3 + len(self.labels)):
            raise ValueError("本地候选模型维度与标签表不一致")
        if not np.isfinite(self.weights).all():
            raise ValueError("本地候选模型包含无效数值")

    def predict(self, text: str) -> tuple[float, list[tuple[str, float]]]:
        logits = features(text) @ self.weights
        state = np.exp(logits[:3] - np.max(logits[:3]))
        state /= state.sum()
        scores = 1 / (1 + np.exp(-np.clip(logits[3:], -20, 20)))
        ordered = sorted(zip(self.labels, (float(v) for v in scores)), key=lambda x: -x[1])
        return float(state[0]), ordered[:5]

    def few_shot_examples(self, text: str) -> list[dict[str, object]]:
        if not self.examples:
            return []
        scores = self.example_features[:, 1:] @ features(text)[1:]
        order = [int(i) for i in np.argsort(-scores) if scores[i] >= 0.15]
        if not order:
            return []
        first = order[0]
        # A relevant contrast prevents two similar positive examples from becoming
        # apparent evidence that a normal/unknown input must also be positive.
        contrast = next(
            (
                i
                for i in order[1:]
                if self.examples[i]["decision_state"] != self.examples[first]["decision_state"]
                and scores[i] >= max(0.15, float(scores[first]) * 0.65)
            ),
            None,
        )
        selected = [first, contrast] if contrast is not None else order[:2]
        return [self.examples[i] for i in selected]


def train_model(
    texts: list[str],
    states: list[str],
    label_sets: list[list[str]],
    labels: list[str],
    path: Path,
    *,
    example_metadata: list[dict[str, object]] | None = None,
) -> None:
    if not (len(texts) == len(states) == len(label_sets)) or not texts:
        raise ValueError("训练文本、状态和标签必须非空且逐条对应")
    if example_metadata is not None and len(example_metadata) != len(texts):
        raise ValueError("训练样本元数据未逐条对应")
    matrix = np.stack([features(text) for text in texts])
    targets = np.zeros((len(texts), 3 + len(labels)), dtype=np.float32)
    for index, (state, names) in enumerate(zip(states, label_sets, strict=True)):
        targets[index, STATES.index(state)] = 1
        for label in names:
            if label in labels:
                targets[index, 3 + labels.index(label)] = 1
    weights = np.zeros((DIMENSION, targets.shape[1]), dtype=np.float32)
    support = targets[:, 3:].sum(axis=0)
    balance = np.minimum(6, np.sqrt((len(texts) - support) / np.maximum(1, support)))
    momentum = np.zeros_like(weights)
    for _ in range(350):
        logits = matrix @ weights
        state = np.exp(logits[:, :3] - logits[:, :3].max(axis=1, keepdims=True))
        state /= state.sum(axis=1, keepdims=True)
        probabilities = 1 / (1 + np.exp(-np.clip(logits[:, 3:], -20, 20)))
        error = np.concatenate(
            [
                state - targets[:, :3],
                (probabilities - targets[:, 3:]) * (1 + targets[:, 3:] * (balance - 1)),
            ],
            axis=1,
        )
        gradient = matrix.T @ error / len(texts) + weights * 0.0005
        momentum = 0.8 * momentum + 0.2 * gradient
        weights -= 0.8 * momentum
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [i for i, text in enumerate(texts) if len(text) <= 500]
    examples = [
        {
            "text": texts[i],
            "decision_state": states[i],
            "risk_labels": label_sets[i],
            **(example_metadata[i] if example_metadata is not None else {}),
        }
        for i in indices
    ]
    np.savez_compressed(
        path,
        weights=weights,
        labels=np.array(labels),
        example_features=matrix[indices],
        examples=np.array([json.dumps(e, ensure_ascii=False) for e in examples]),
        version=np.array(f"char-ngram-development{len(texts)}-boundary-v3"),
    )
