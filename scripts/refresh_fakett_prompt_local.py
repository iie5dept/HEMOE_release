from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_real_fake_sft import FAKE_NEWS_PROMPT


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Write FakeTT jsonl files with the latest prompt to a separate directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakett",
        help="Directory containing fakett_train/val/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakett_prompt_v2",
        help="Separate output directory; it must not be the input directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in output-dir. Input files are never overwritten.",
    )
    return parser.parse_args()


def refresh_file(input_path: Path, output_path: Path, overwrite: bool) -> int:
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if not messages or messages[0].get("role") != "user":
                raise ValueError(f"Missing first user message in {input_path}:{line_number}")
            row["messages"][0]["content"] = build_prompt_from_text(messages[0]["content"])
            rows.append(row)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_prompt_from_text(text: str) -> str:
    fields = parse_observation_fields(text)
    parts = ["<video>", FAKE_NEWS_PROMPT, "", "Auxiliary observations:"]

    claim = fields.get("news claim to verify", "")
    if claim:
        parts.append(f"- news claim to verify: <claim>{strip_claim_tags(claim)}</claim>")
    else:
        raise ValueError("Prompt does not contain a 'news claim to verify' field.")

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


def strip_claim_tags(value: str) -> str:
    value = value.strip()
    while value.startswith("<claim>") and value.endswith("</claim>"):
        value = value[len("<claim>"):-len("</claim>")].strip()
    if not value:
        raise ValueError("The news claim is empty after removing <claim> tags.")
    return value


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
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("--output-dir must differ from --input-dir to protect the source data.")

    paths = [
        (
            args.input_dir / f"fakett_{split}.jsonl",
            args.output_dir / f"fakett_{split}.jsonl",
        )
        for split in ("train", "val", "test")
    ]
    missing_inputs = [str(input_path) for input_path, _ in paths if not input_path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing_inputs)}")
    existing_outputs = [str(output_path) for _, output_path in paths if output_path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: " + ", ".join(existing_outputs)
        )

    for input_path, output_path in paths:
        count = refresh_file(input_path, output_path, overwrite=args.overwrite)
        print(
            json.dumps(
                {"input": str(input_path), "output": str(output_path), "count": count},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
