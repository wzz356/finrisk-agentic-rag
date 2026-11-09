"""来源级中文BM25；显式本地profile，无外部embedding或重排调用。"""

import math
import re
from collections import Counter, defaultdict
from typing import Any

TOKENIZER_VERSION = "han-unigram-bigram-ascii/1.0"
PROFILE = "bm25-source-v1"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word)
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for idx, chunk in enumerate(chunks):
            terms = Counter(tokenize(chunk["quote"]))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((idx, count))
        self.average = sum(self.lengths) / max(1, len(chunks))

    def search(self, text: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        scores: dict[int, float] = defaultdict(float)
        total = len(self.chunks)
        for term in set(tokenize(text)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for idx, count in posting:
                denominator = count + 1.5 * (0.25 + 0.75 * self.lengths[idx] / self.average)
                scores[idx] += idf * count * 2.5 / denominator
        results: list[tuple[dict[str, Any], float]] = []
        seen: set[str] = set()
        for idx in sorted(scores, key=lambda i: (-scores[i], i)):
            chunk = self.chunks[idx]
            if chunk["source_id"] in seen:
                continue
            seen.add(chunk["source_id"])
            results.append((chunk, scores[idx]))
            if len(results) >= top_k:
                break
        return results
