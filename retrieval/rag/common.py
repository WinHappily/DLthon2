from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "config" / "versions.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


TOKEN_RE = re.compile(r"부칙\s*제\s*\d+\s*조|제\s*\d+\s*조|표\s*\d+|\d+(?:[.,]\d+)?|[A-Za-z]+\d*|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """BM25용 토큰. 조항 · 표 번호는 통째로, 한글은 낱말과 두 글자 조각(bigram)을 함께 쓴다."""
    out = []
    for t in TOKEN_RE.findall(text):
        t = re.sub(r"\s+", "", t)
        if re.fullmatch(r"[가-힣]+", t):
            out.append(t)
            out.extend(t[i:i + 2] for i in range(len(t) - 1))
        else:
            out.append(t.lower())
    return out
