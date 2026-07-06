from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_real_fake_sft import FAKE_NEWS_PROMPT


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Refresh prompt text in existing FakeTT swift jsonl files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakett",
        help="Directory containing fakett_train/val/test.jsonl",
    )
    return parser.parse_args()


def refresh_file(path: Path) -> int:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if not messages:
                rows.append(row)
                continue
            row["messages"][0]["content"] = build_prompt_from_text(messages[0]["content"])
            rows.append(row)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_prompt_from_text(text: str) -> str:
    fields = parse_observation_fields(text)
    parts = ["<video>", FAKE_NEWS_PROMPT, "", "Auxiliary observations:"]

    claim = fields.get("news claim to verify", "")
    if claim:
        parts.append(f"- news claim to verify: {claim}")

    caption = fields.get("uploader caption", "")
    if caption:
        parts.append(f"- uploader caption: {caption}")

    profile = fields.get("uploader profile", "")
    if profile:
        parts.append(f"- uploader profile: {profile}")

    publish_time = fields.get("publish time metadata", "")
    if publish_time:
        parts.append(f"- publish time metadata: {publish_time}")

    return "\n".join(parts)


def parse_observation_fields(text: str) -> dict[str, str]:
    lines = text.splitlines()
    in_observations = False
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_value
        if current_key is not None:
            fields[current_key] = "\n".join(current_value).strip()
        current_key = None
        current_value = []

    for line in lines:
        if not in_observations:
            if line.strip() == "Auxiliary observations:":
                in_observations = True
            continue

        if line.startswith("- "):
            flush()
            body = line[2:]
            if ":" in body:
                key, value = body.split(":", 1)
                current_key = key.strip()
                current_value = [value.strip()]
            else:
                current_key = body.strip()
                current_value = []
        else:
            if current_key is not None:
                current_value.append(line.rstrip())

    flush()
    return fields


def main() -> None:
    args = parse_args()
    for split in ("train", "val", "test"):
        path = args.input_dir / f"fakett_{split}.jsonl"
        count = refresh_file(path)
        print(json.dumps({"file": str(path), "count": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
