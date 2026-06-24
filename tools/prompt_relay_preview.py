#!/usr/bin/env python3
"""Preview Prompt Relay Smart prompt parsing without launching ComfyUI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCALE_FACTOR = 100000.0


def _load_parse_smart_prompt(repo_root: Path):
    parser_path = repo_root / "parser.py"
    spec = importlib.util.spec_from_file_location("prompt_relay_parser", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser module from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_smart_prompt


def _read_prompt(args: argparse.Namespace) -> str:
    sources = [bool(args.prompt), args.file is not None, args.stdin]
    if sum(sources) > 1:
        raise ValueError("Use only one prompt source: argument, --file, or --stdin")

    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide a prompt argument, --file, or --stdin")


def _valid_segments(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [segment for segment in parsed if segment["text"].strip()]
    return valid or [{"text": " ", "weight": 1.0}]


def _convert_to_latent_lengths(
    pixel_lengths: list[int], temporal_stride: int, latent_frames: int
) -> list[int]:
    """Mirror nodes._convert_to_latent_lengths without importing ComfyUI modules."""
    if not pixel_lengths:
        return []

    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(value) for value in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for index in range(diff):
            result[order[index % len(order)]] += 1

    for i, value in enumerate(result):
        if value >= 1:
            continue
        max_idx = max(range(len(result)), key=lambda j: result[j])
        if result[max_idx] > 1:
            result[max_idx] -= 1
            result[i] = 1

    return result


def _distribute_segment_lengths(
    num_segments: int, latent_frames: int, specified_lengths: list[int] | None = None
) -> list[int]:
    """Mirror prompt_relay.distribute_segment_lengths for preview output."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for length in lengths:
        end = min(cursor + length, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


def build_preview(
    prompt: str,
    *,
    global_prompt: str,
    latent_frames: int | None,
    temporal_stride: int,
    repo_root: Path,
) -> dict[str, Any]:
    if temporal_stride <= 0:
        raise ValueError("--temporal-stride must be greater than zero")
    if latent_frames is not None and latent_frames <= 0:
        raise ValueError("--latent-frames must be greater than zero")

    parse_smart_prompt = _load_parse_smart_prompt(repo_root)
    segments = _valid_segments(parse_smart_prompt(prompt))
    encoded_lengths = [int(segment["weight"] * SCALE_FACTOR) for segment in segments]

    preview_segments: list[dict[str, Any]] = []
    latent_lengths = None
    if latent_frames is not None:
        converted = _convert_to_latent_lengths(
            encoded_lengths, temporal_stride, latent_frames
        )
        latent_lengths = _distribute_segment_lengths(
            len(segments), latent_frames, converted
        )

    for index, segment in enumerate(segments):
        item = {
            "index": index + 1,
            "text": segment["text"],
            "weight": segment["weight"],
            "encoded_length": encoded_lengths[index],
        }
        if latent_lengths is not None:
            item["estimated_latent_frames"] = latent_lengths[index]
        preview_segments.append(item)

    global_text = global_prompt.strip() or segments[0]["text"]

    return {
        "global_prompt": global_text,
        "local_prompts": " | ".join(segment["text"] for segment in segments),
        "segment_lengths": ", ".join(str(length) for length in encoded_lengths),
        "temporal_stride": temporal_stride,
        "latent_frames": latent_frames,
        "segments": preview_segments,
    }


def _print_text(preview: dict[str, Any]) -> None:
    print("global_prompt:")
    print(preview["global_prompt"])
    print()
    print("local_prompts:")
    print(preview["local_prompts"])
    print()
    print("segment_lengths:")
    print(preview["segment_lengths"])
    print()
    print("segments:")
    for segment in preview["segments"]:
        line = (
            f"{segment['index']}. weight={segment['weight']} "
            f"encoded_length={segment['encoded_length']}"
        )
        if "estimated_latent_frames" in segment:
            line += f" estimated_latent_frames={segment['estimated_latent_frames']}"
        print(line)
        print(f"   {segment['text']}")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Preview the Smart prompt parser and encoded Prompt Relay segment lengths "
            "without loading ComfyUI."
        )
    )
    parser.add_argument("prompt", nargs="?", help="Smart prompt text to preview")
    parser.add_argument("--file", type=Path, help="Read Smart prompt text from a file")
    parser.add_argument("--stdin", action="store_true", help="Read Smart prompt text from stdin")
    parser.add_argument(
        "--global-prompt",
        default="",
        help="Optional global prompt; defaults to the first parsed segment",
    )
    parser.add_argument(
        "--latent-frames",
        type=int,
        help="Optional latent frame count for estimating final segment allocation",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=1,
        help="Pixel-to-latent temporal stride used for allocation estimates",
    )
    parser.add_argument("--json", action="store_true", help="Write machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        prompt = _read_prompt(args)
        preview = build_preview(
            prompt,
            global_prompt=args.global_prompt,
            latent_frames=args.latent_frames,
            temporal_stride=args.temporal_stride,
            repo_root=repo_root,
        )
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.json:
        print(json.dumps(preview, indent=2))
    else:
        _print_text(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
