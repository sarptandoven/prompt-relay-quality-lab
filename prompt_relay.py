import logging
import math
import os
import torch

log = logging.getLogger(__name__)


DEFAULT_MAX_MASK_ELEMENTS = 1_073_741_824


def _dtype_bytes(dtype):
    text = str(dtype).lower()
    if "64" in text or "double" in text:
        return 8
    if "16" in text or "half" in text or "bfloat" in text:
        return 2
    return 4


def _max_mask_elements():
    raw = os.environ.get("PROMPT_RELAY_MAX_MASK_ELEMENTS", str(DEFAULT_MAX_MASK_ELEMENTS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_MASK_ELEMENTS
    return max(0, value)


def _check_mask_size(Lq, Lk, dtype, mode):
    elements = int(Lq) * int(Lk)
    cap = _max_mask_elements()
    if cap and elements > cap:
        est_gib = elements * _dtype_bytes(dtype) / (1024 ** 3)
        cap_gib = cap * _dtype_bytes(dtype) / (1024 ** 3)
        max_query_tokens = max(cap // max(int(Lk), 1), 1)
        raise RuntimeError(
            "PromptRelay: refusing to build a {mode} attention mask with "
            "Lq={Lq:,}, Lk={Lk:,} ({elements:,} elements, ~{est_gib:.2f} GiB at {dtype}). "
            "This usually means the workflow is trying to Prompt Relay a long video as one full attention pass. "
            "For 2500+ frame generation, split the video into temporal chunks/windows and apply Prompt Relay per chunk. "
            "At the current text-token length, the safety cap allows about {max_query_tokens:,} query tokens per chunk. "
            "Use plan_temporal_chunks(...) in this module to size latent-frame chunks from tokens/frame, or set "
            "PROMPT_RELAY_MAX_MASK_ELEMENTS=0 only if you have deliberately budgeted the VRAM/RAM. "
            "Current safety cap is {cap:,} elements (~{cap_gib:.2f} GiB at this dtype)."
            .format(
                mode=mode,
                Lq=Lq,
                Lk=Lk,
                elements=elements,
                est_gib=est_gib,
                dtype=dtype,
                max_query_tokens=max_query_tokens,
                cap=cap,
                cap_gib=cap_gib,
            )
        )
    return elements


def estimate_max_chunk_frames(tokens_per_frame, text_tokens, max_mask_elements=None, safety_margin=0.9):
    """Estimate the largest latent-frame chunk that fits the additive mask budget.

    Prompt Relay's additive temporal mask is shaped ``[query_tokens, text_tokens]``.
    For video attention, ``query_tokens = latent_frames * tokens_per_frame``. This
    helper is intentionally pure and runtime-neutral so long-form schedulers can
    plan finite windows without changing the default one-pass Comfy node behavior.
    """
    if max_mask_elements is None:
        max_mask_elements = _max_mask_elements()
    if max_mask_elements == 0:
        return math.inf
    try:
        tokens_per_frame = int(tokens_per_frame)
        text_tokens = int(text_tokens)
        safety_margin = float(safety_margin)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: chunk planner inputs must be numeric.") from exc
    if tokens_per_frame <= 0 or text_tokens <= 0:
        raise ValueError("PromptRelay: tokens_per_frame and text_tokens must be positive to plan chunks.")
    if not math.isfinite(safety_margin) or safety_margin <= 0:
        raise ValueError("PromptRelay: safety_margin must be positive and finite.")

    budget = max(1, int(max_mask_elements * min(safety_margin, 1.0)))
    elements_per_frame = tokens_per_frame * text_tokens
    if elements_per_frame > budget:
        raise ValueError(
            "PromptRelay: even one latent frame would exceed the mask budget "
            f"({elements_per_frame:,} elements/frame > {budget:,} budgeted elements). "
            "Reduce tokens_per_frame/text_tokens, raise PROMPT_RELAY_MAX_MASK_ELEMENTS, or disable the cap only after VRAM budgeting."
        )
    return max(1, budget // elements_per_frame)


def plan_temporal_chunks(latent_frames, tokens_per_frame, text_tokens, max_mask_elements=None, overlap_frames=4, safety_margin=0.9):
    """Return overlapping latent-frame chunks for long-form Prompt Relay windows.

    Chunks use half-open ``[start, end)`` latent-frame ranges. The plan is safe for
    arbitrarily long timelines because each chunk is bounded by the configured
    mask element cap; callers can stitch windows with overlap/crossfade outside
    this module.
    """
    if latent_frames <= 0:
        return {"max_chunk_frames": 0, "overlap_frames": 0, "chunks": []}

    max_chunk_frames = estimate_max_chunk_frames(
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        safety_margin=safety_margin,
    )
    if math.isinf(max_chunk_frames) or max_chunk_frames >= latent_frames:
        return {
            "max_chunk_frames": latent_frames,
            "overlap_frames": 0,
            "chunks": [{"start": 0, "end": int(latent_frames), "length": int(latent_frames)}],
        }

    max_chunk_frames = int(max_chunk_frames)
    overlap = min(max(int(overlap_frames), 0), max_chunk_frames - 1)
    step = max_chunk_frames - overlap
    chunks = []
    start = 0
    while start < latent_frames:
        end = min(start + max_chunk_frames, int(latent_frames))
        chunks.append({"start": start, "end": end, "length": end - start})
        if end >= latent_frames:
            break
        start += step

    return {"max_chunk_frames": max_chunk_frames, "overlap_frames": overlap, "chunks": chunks}


def plan_quality_temporal_chunks(
    latent_frames,
    tokens_per_frame,
    text_tokens,
    max_mask_elements=None,
    target_chunk_frames=257,
    overlap_frames=16,
    safety_margin=0.9,
):
    """Return quality-oriented long-form chunks that still respect mask budget.

    ``plan_temporal_chunks`` answers the hard memory question: how large can one
    Prompt Relay attention window be? For 2500+ frame videos that cap can still
    be far larger than a visually useful LTXV window. This wrapper chooses the
    smaller of the memory-bounded maximum and a quality target so ComfyUI
    schedulers can start from bounded 129-257 frame-style windows without
    accidentally launching one giant semantically brittle pass.
    """
    try:
        latent_frames = int(latent_frames)
        target_chunk_frames = int(target_chunk_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: quality chunk planner inputs must be integers.") from exc
    if target_chunk_frames <= 0:
        raise ValueError("PromptRelay: target_chunk_frames must be positive.")

    if latent_frames <= 0:
        return {
            "max_chunk_frames": 0,
            "budget_max_chunk_frames": 0,
            "quality_target_frames": target_chunk_frames,
            "overlap_frames": 0,
            "chunks": [],
        }

    budget_max = estimate_max_chunk_frames(
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        safety_margin=safety_margin,
    )
    if math.isinf(budget_max):
        chosen = target_chunk_frames
        budget_report = math.inf
    else:
        budget_report = int(budget_max)
        chosen = min(int(budget_max), target_chunk_frames)

    plan = plan_temporal_chunks(
        latent_frames,
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        overlap_frames=overlap_frames,
        safety_margin=safety_margin,
    )
    if chosen < plan["max_chunk_frames"]:
        overlap = min(max(int(overlap_frames), 0), chosen - 1)
        step = chosen - overlap
        chunks = []
        start = 0
        while start < latent_frames:
            end = min(start + chosen, latent_frames)
            chunks.append({"start": start, "end": end, "length": end - start})
            if end >= latent_frames:
                break
            start += step
        plan = {"max_chunk_frames": chosen, "overlap_frames": overlap, "chunks": chunks}

    plan["budget_max_chunk_frames"] = budget_report
    plan["quality_target_frames"] = target_chunk_frames
    return plan


def plan_temporal_chunk_page(
    latent_frames,
    tokens_per_frame,
    text_tokens,
    max_mask_elements=None,
    overlap_frames=4,
    safety_margin=0.9,
    start_chunk=0,
    max_chunks=16,
):
    """Return a bounded page of temporal chunks without materializing a huge plan.

    ``plan_temporal_chunks`` is convenient for normal long-form renders, but very
    long timelines should be schedulable as a resumable stream. This helper uses
    the same budget math and chunk geometry while returning only ``max_chunks``
    windows plus a cursor for the next page. That keeps planning memory bounded
    for hour-scale or effectively unbounded generation queues.
    """
    try:
        latent_frames = int(latent_frames)
        start_chunk = int(start_chunk)
        max_chunks = int(max_chunks)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: paged chunk planner inputs must be integers.") from exc
    if start_chunk < 0:
        raise ValueError("PromptRelay: start_chunk must be non-negative.")
    if max_chunks <= 0:
        raise ValueError("PromptRelay: max_chunks must be positive.")
    if latent_frames <= 0:
        return {
            "max_chunk_frames": 0,
            "overlap_frames": 0,
            "start_chunk": start_chunk,
            "next_chunk": None,
            "complete": True,
            "chunks": [],
        }

    max_chunk_frames = estimate_max_chunk_frames(
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        safety_margin=safety_margin,
    )
    if math.isinf(max_chunk_frames) or max_chunk_frames >= latent_frames:
        chunks = [] if start_chunk else [{"start": 0, "end": latent_frames, "length": latent_frames}]
        return {
            "max_chunk_frames": latent_frames,
            "overlap_frames": 0,
            "start_chunk": start_chunk,
            "next_chunk": None,
            "complete": True,
            "chunks": chunks,
        }

    max_chunk_frames = int(max_chunk_frames)
    overlap = min(max(int(overlap_frames), 0), max_chunk_frames - 1)
    step = max_chunk_frames - overlap
    total_chunks = math.ceil(max(latent_frames - max_chunk_frames, 0) / step) + 1
    if start_chunk >= total_chunks:
        return {
            "max_chunk_frames": max_chunk_frames,
            "overlap_frames": overlap,
            "start_chunk": start_chunk,
            "next_chunk": None,
            "complete": True,
            "total_chunks": total_chunks,
            "chunks": [],
        }

    first_start = start_chunk * step
    chunks = []
    start = first_start
    while start < latent_frames and len(chunks) < max_chunks:
        end = min(start + max_chunk_frames, latent_frames)
        chunks.append({"start": start, "end": end, "length": end - start})
        if end >= latent_frames:
            break
        start += step

    complete = not chunks or chunks[-1]["end"] >= latent_frames
    next_chunk = None if complete else start_chunk + len(chunks)
    return {
        "max_chunk_frames": max_chunk_frames,
        "overlap_frames": overlap,
        "start_chunk": start_chunk,
        "next_chunk": next_chunk,
        "complete": complete,
        "total_chunks": total_chunks,
        "chunks": chunks,
    }


def plan_quality_temporal_chunk_page(
    latent_frames,
    tokens_per_frame,
    text_tokens,
    max_mask_elements=None,
    target_chunk_frames=257,
    overlap_frames=16,
    safety_margin=0.9,
    start_chunk=0,
    max_chunks=16,
):
    """Return a resumable page of quality-bounded long-form chunks.

    This is the streaming counterpart to ``plan_quality_temporal_chunks``. It
    keeps quality windows small enough for stable long-form semantics while only
    materializing the next page, so hour-scale timelines do not require building
    a giant chunk list before scheduling can start.
    """
    try:
        latent_frames = int(latent_frames)
        target_chunk_frames = int(target_chunk_frames)
        start_chunk = int(start_chunk)
        max_chunks = int(max_chunks)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: quality paged chunk planner inputs must be integers.") from exc
    if target_chunk_frames <= 0:
        raise ValueError("PromptRelay: target_chunk_frames must be positive.")
    if start_chunk < 0:
        raise ValueError("PromptRelay: start_chunk must be non-negative.")
    if max_chunks <= 0:
        raise ValueError("PromptRelay: max_chunks must be positive.")

    if latent_frames <= 0:
        return {
            "max_chunk_frames": 0,
            "budget_max_chunk_frames": 0,
            "quality_target_frames": target_chunk_frames,
            "overlap_frames": 0,
            "start_chunk": start_chunk,
            "next_chunk": None,
            "complete": True,
            "total_chunks": 0,
            "chunks": [],
        }

    budget_max = estimate_max_chunk_frames(
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        safety_margin=safety_margin,
    )
    if math.isinf(budget_max):
        budget_report = math.inf
        max_chunk_frames = min(target_chunk_frames, latent_frames)
    else:
        budget_report = int(budget_max)
        max_chunk_frames = min(int(budget_max), target_chunk_frames, latent_frames)

    overlap = min(max(int(overlap_frames), 0), max_chunk_frames - 1)
    step = max_chunk_frames - overlap
    total_chunks = math.ceil(max(latent_frames - max_chunk_frames, 0) / step) + 1
    if start_chunk >= total_chunks:
        return {
            "max_chunk_frames": max_chunk_frames,
            "budget_max_chunk_frames": budget_report,
            "quality_target_frames": target_chunk_frames,
            "overlap_frames": overlap,
            "start_chunk": start_chunk,
            "next_chunk": None,
            "complete": True,
            "total_chunks": total_chunks,
            "chunks": [],
        }

    first_start = start_chunk * step
    chunks = []
    start = first_start
    while start < latent_frames and len(chunks) < max_chunks:
        end = min(start + max_chunk_frames, latent_frames)
        chunks.append({"start": start, "end": end, "length": end - start})
        if end >= latent_frames:
            break
        start += step

    complete = not chunks or chunks[-1]["end"] >= latent_frames
    next_chunk = None if complete else start_chunk + len(chunks)
    return {
        "max_chunk_frames": max_chunk_frames,
        "budget_max_chunk_frames": budget_report,
        "quality_target_frames": target_chunk_frames,
        "overlap_frames": overlap,
        "start_chunk": start_chunk,
        "next_chunk": next_chunk,
        "complete": complete,
        "total_chunks": total_chunks,
        "chunks": chunks,
    }


def format_temporal_chunk_plan_diagnostics(plan, handoffs=None, max_chunks=8):
    """Return a bounded one-line summary of a long-form chunk plan.

    This is intended for ComfyUI validation logs before a costly 2500+ frame
    render. It reports the effective timeline span, configured mask-bounded
    chunk size, overlap, stitched output coverage, a worst seam status, and a
    head/tail sample of chunk windows without dumping every chunk in an
    infinite/very-long plan.
    """
    chunks = list((plan or {}).get("chunks", []))
    max_chunk_frames = (plan or {}).get("max_chunk_frames", 0)
    overlap_frames = (plan or {}).get("overlap_frames", 0)

    if not chunks:
        return (
            "chunk_plan: frames=0 chunks=0 max_chunk={max_chunk} overlap={overlap} "
            "stitched_frames=0 worst_handoff=none"
        ).format(max_chunk=max_chunk_frames, overlap=overlap_frames)

    starts = [int(chunk["start"]) for chunk in chunks]
    ends = [int(chunk["end"]) for chunk in chunks]
    span_start = min(starts)
    span_end = max(ends)

    try:
        stitched_frames = sum(chunk["keep_length"] for chunk in plan_chunk_stitch_ranges(chunks))
    except ValueError:
        stitched_frames = "invalid"

    status_rank = {"hard_cut": 3, "short_overlap": 2, "ok": 1}
    worst_handoff = "none"
    if handoffs:
        worst_handoff = max(
            (handoff.get("status", "unknown") for handoff in handoffs),
            key=lambda status: status_rank.get(status, 0),
        )

    total = len(chunks)
    indexed_chunks = list(enumerate(chunks))
    if max_chunks and total > max_chunks:
        head_count = max_chunks // 2
        tail_count = max_chunks - head_count
        indexed_chunks = indexed_chunks[:head_count] + indexed_chunks[-tail_count:]

    parts = [
        "chunk_plan: frames={frames} chunks={chunks} max_chunk={max_chunk} overlap={overlap} "
        "stitched_frames={stitched} worst_handoff={worst}".format(
            frames=span_end - span_start,
            chunks=total,
            max_chunk=max_chunk_frames,
            overlap=overlap_frames,
            stitched=stitched_frames,
            worst=worst_handoff,
        )
    ]

    previous_idx = None
    for idx, chunk in indexed_chunks:
        if previous_idx is not None and idx != previous_idx + 1:
            parts.append(f"... {idx - previous_idx - 1} chunk(s) omitted ...")
        parts.append("chunk{idx}=[{start}:{end}] len={length}".format(
            idx=idx,
            start=int(chunk["start"]),
            end=int(chunk["end"]),
            length=int(chunk.get("length", int(chunk["end"]) - int(chunk["start"]))),
        ))
        previous_idx = idx

    return "; ".join(parts)


def evaluate_temporal_chunk_plan(plan, tokens_per_frame, text_tokens, max_mask_elements=None, min_context_frames=8):
    """Evaluate whether a temporal chunk plan is safe enough to render.

    This is a pure preflight gate for long-form schedulers: it combines mask
    budget checks, stitch coverage, and seam/continuity status into a compact
    machine-readable record before a costly render is launched.
    """
    chunks = list((plan or {}).get("chunks", []))
    if max_mask_elements is None:
        max_mask_elements = _max_mask_elements()
    try:
        tokens_per_frame = int(tokens_per_frame)
        text_tokens = int(text_tokens)
        max_mask_elements = int(max_mask_elements)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: chunk evaluation inputs must be numeric.") from exc
    if tokens_per_frame <= 0 or text_tokens <= 0:
        raise ValueError("PromptRelay: tokens_per_frame and text_tokens must be positive to evaluate chunks.")

    mask_elements = []
    over_budget_chunks = []
    for idx, chunk in enumerate(chunks):
        if "length" in chunk:
            length = int(chunk["length"])
        else:
            length = int(chunk["end"]) - int(chunk["start"])
        elements = length * tokens_per_frame * text_tokens
        mask_elements.append(elements)
        if max_mask_elements and elements > max_mask_elements:
            over_budget_chunks.append(idx)

    error = None
    stitched_frames = 0
    handoffs = []
    continuity = []
    try:
        stitched = plan_chunk_stitch_ranges(chunks)
        stitched_frames = sum(chunk["keep_length"] for chunk in stitched)
        handoffs = plan_chunk_handoffs(chunks, min_context_frames=min_context_frames)
        continuity = plan_chunk_continuity_windows(chunks, context_frames=min_context_frames)
    except ValueError as exc:
        error = str(exc)

    def _counts(rows):
        counts = {}
        for row in rows:
            status = row.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    handoff_counts = _counts(handoffs)
    continuity_counts = _counts(continuity)
    handoff_rank = {"hard_cut": 3, "short_overlap": 2, "ok": 1}
    continuity_rank = {"missing": 3, "partial": 2, "ok": 1}
    worst_handoff = max(handoff_counts, key=lambda status: handoff_rank.get(status, 0)) if handoff_counts else "none"
    worst_continuity = max(continuity_counts, key=lambda status: continuity_rank.get(status, 0)) if continuity_counts else "none"
    safe = (
        error is None
        and not over_budget_chunks
        and worst_handoff not in {"hard_cut", "short_overlap"}
        and worst_continuity not in {"missing", "partial"}
    )

    return {
        "chunks": len(chunks),
        "stitched_frames": stitched_frames,
        "peak_mask_elements": max(mask_elements, default=0),
        "max_mask_elements": max_mask_elements,
        "over_budget_chunks": over_budget_chunks,
        "handoff_status_counts": handoff_counts,
        "continuity_status_counts": continuity_counts,
        "worst_handoff": worst_handoff,
        "worst_continuity": worst_continuity,
        "safe": safe,
        "error": error,
    }


def evaluate_temporal_chunk_page(
    page,
    tokens_per_frame,
    text_tokens,
    max_mask_elements=None,
    min_context_frames=8,
    previous_tail_chunk=None,
):
    """Evaluate one resumable chunk page plus its boundary to the prior page.

    Infinite/very-long schedulers usually plan a few chunks at a time. Evaluating
    only the current page can miss a gap or weak overlap between the last chunk
    from the previous page and the first chunk in this page. This helper keeps
    the budget check scoped to the current page while preflighting that carry-in
    seam when ``previous_tail_chunk`` is supplied.
    """
    chunks = list((page or {}).get("chunks", []))
    evaluation = evaluate_temporal_chunk_plan(
        {"chunks": chunks},
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        max_mask_elements=max_mask_elements,
        min_context_frames=min_context_frames,
    )

    boundary_error = None
    boundary_handoffs = []
    boundary_continuity = []
    boundary_safe = True
    boundary_error_code = None
    if previous_tail_chunk is not None and chunks:
        try:
            seam_chunks = [previous_tail_chunk, chunks[0]]
            boundary_handoffs = plan_chunk_handoffs(seam_chunks, min_context_frames=min_context_frames)
            boundary_continuity = plan_chunk_continuity_windows(seam_chunks, context_frames=min_context_frames)
        except ValueError as exc:
            error_text = str(exc)
            boundary_error = error_text
            boundary_error_code = "gap_from_previous_page" if "contiguous or overlapping" in error_text else "boundary_error"
            boundary_safe = False

    def _worst(rows, rank, empty="none"):
        if not rows:
            return empty
        return max((row.get("status", "unknown") for row in rows), key=lambda status: rank.get(status, 0))

    worst_boundary_handoff = _worst(boundary_handoffs, {"hard_cut": 3, "short_overlap": 2, "ok": 1})
    worst_boundary_continuity = _worst(boundary_continuity, {"missing": 3, "partial": 2, "ok": 1})
    if worst_boundary_handoff in {"hard_cut", "short_overlap"} or worst_boundary_continuity in {"missing", "partial"}:
        boundary_safe = False

    return {
        **evaluation,
        "safe": evaluation["safe"] and boundary_safe,
        "page_start_chunk": (page or {}).get("start_chunk"),
        "next_chunk": (page or {}).get("next_chunk"),
        "complete": bool((page or {}).get("complete", True)),
        "total_chunks": (page or {}).get("total_chunks"),
        "boundary_error": boundary_error,
        "boundary_error_code": boundary_error_code,
        "boundary_handoff_status": worst_boundary_handoff,
        "boundary_continuity_status": worst_boundary_continuity,
        "last_chunk": chunks[-1] if chunks else previous_tail_chunk,
    }


def _chunk_stream_prompt_fingerprint(token_ranges=None, segment_lengths=None, min_visible_frames=1):
    """Return compact prompt-schedule inputs that affect resumable stream safety."""
    if token_ranges is None and segment_lengths is None:
        return None
    if token_ranges is None or segment_lengths is None:
        return {"incomplete": True}
    ranges = [[int(start), int(end)] for start, end in token_ranges]
    lengths = [int(length) for length in segment_lengths]
    return {
        "range_count": len(ranges),
        "range_first": ranges[0] if ranges else None,
        "range_last": ranges[-1] if ranges else None,
        "range_checksum": sum((idx + 1) * (start * 1_000_003 + end) for idx, (start, end) in enumerate(ranges)),
        "length_count": len(lengths),
        "length_sum": sum(lengths),
        "length_checksum": sum((idx + 1) * length for idx, length in enumerate(lengths)),
        "min_visible_frames": int(min_visible_frames),
    }


def _chunk_stream_config_fingerprint(
    latent_frames,
    tokens_per_frame,
    text_tokens,
    max_mask_elements,
    target_chunk_frames,
    overlap_frames,
    safety_margin,
    crossfade_curve,
    min_context_frames=8,
    anchors_per_chunk=3,
    max_anchors=24,
    token_ranges=None,
    segment_lengths=None,
    min_visible_frames=1,
):
    """Return the resume-critical scheduler inputs for drift checks."""
    effective_cap = _max_mask_elements() if max_mask_elements is None else int(max_mask_elements)
    return {
        "latent_frames": int(latent_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "text_tokens": int(text_tokens),
        "max_mask_elements": effective_cap,
        "target_chunk_frames": int(target_chunk_frames),
        "overlap_frames": int(overlap_frames),
        "safety_margin": float(safety_margin),
        "crossfade_curve": str(crossfade_curve),
        "min_context_frames": int(min_context_frames),
        "anchors_per_chunk": int(anchors_per_chunk),
        "max_anchors": int(max_anchors),
        "prompt_schedule": _chunk_stream_prompt_fingerprint(
            token_ranges=token_ranges,
            segment_lengths=segment_lengths,
            min_visible_frames=min_visible_frames,
        ),
    }


def _validate_chunk_stream_resume_state(state):
    """Reject resume cursors that would rewind behind verified clean output.

    Close-to-infinite schedulers persist compact state between queue items. If a
    stale or manually-edited checkpoint moves ``next_chunk`` behind the last
    clean tail, the next page can re-render old chunks while still conditioning
    from newer anchors. Fail early instead of silently corrupting continuity.
    """
    if not state:
        return

    def optional_int(key):
        value = state.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PromptRelay: chunk stream resume state {key} must be an integer.") from exc

    next_chunk = optional_int("next_chunk")
    last_chunk_index = optional_int("last_chunk_index")
    total_chunks = optional_int("total_chunks")
    if next_chunk is not None and next_chunk < 0:
        raise ValueError("PromptRelay: chunk stream resume state next_chunk must be non-negative.")
    if last_chunk_index is not None and last_chunk_index < 0:
        raise ValueError("PromptRelay: chunk stream resume state last_chunk_index must be non-negative.")
    if total_chunks is not None and total_chunks < 0:
        raise ValueError("PromptRelay: chunk stream resume state total_chunks must be non-negative.")

    if last_chunk_index is not None and next_chunk is not None and next_chunk < last_chunk_index + 1:
        raise ValueError(
            "PromptRelay: chunk stream resume cursor is behind the verified clean tail; "
            "restart from a coherent checkpoint or retry from the rejected chunk before advancing."
        )
    if state.get("complete") and total_chunks is not None and next_chunk is not None and next_chunk < total_chunks:
        raise ValueError(
            "PromptRelay: completed chunk stream state cannot resume before total_chunks; "
            "use the final checkpoint cursor or clear the completed state."
        )


def _select_chunk_stream_tail(chunks, previous_tail, start_chunk, accepted_chunk_indices=None, rejected_chunk_indices=None):
    """Return the newest verified-clean tail chunk for paged resume state.

    Feedback-driven infinite schedulers may reject a rendered chunk after visual
    or metric checks. Such chunks must not become the continuity tail for the
    next page, otherwise the following boundary preflight would treat bad output
    as clean conditioning state.
    """
    chunks = list(chunks or [])
    try:
        start_chunk = int(start_chunk or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: start_chunk must be an integer for stream tail selection.") from exc

    accepted_set, rejected_set = _chunk_stream_feedback_sets(accepted_chunk_indices, rejected_chunk_indices)
    rejected_current = _chunk_stream_rejected_current_indices(chunks, start_chunk, rejected_set)
    retry_from = min(rejected_current) if rejected_current else None

    for local_idx in range(len(chunks) - 1, -1, -1):
        absolute_idx = start_chunk + local_idx
        if retry_from is not None and absolute_idx >= retry_from:
            continue
        if accepted_set is not None and absolute_idx not in accepted_set:
            continue
        if absolute_idx in rejected_set:
            continue
        return chunks[local_idx], absolute_idx
    return previous_tail, None


def _chunk_stream_feedback_sets(accepted_chunk_indices=None, rejected_chunk_indices=None):
    """Normalize absolute chunk feedback from a verifier loop."""
    accepted_set = None if accepted_chunk_indices is None else {int(idx) for idx in accepted_chunk_indices}
    rejected_set = set() if rejected_chunk_indices is None else {int(idx) for idx in rejected_chunk_indices}
    if accepted_set is not None and accepted_set & rejected_set:
        raise ValueError("PromptRelay: accepted and rejected chunk indices must not overlap.")
    return accepted_set, rejected_set


def _chunk_stream_rejected_current_indices(chunks, start_chunk, rejected_set):
    """Return rejected absolute indices that fall inside the currently planned page."""
    if not chunks or not rejected_set:
        return []
    current = {int(start_chunk) + idx for idx, _chunk in enumerate(chunks)}
    return sorted(current & set(rejected_set))


def _unsafe_crossfade_evaluation(error):
    return {
        "windows": 0,
        "min_blend_frames": 0,
        "bad_weight_frames": 0,
        "status_counts": {"gap": 1},
        "worst_status": "gap",
        "safe": False,
        "error": str(error),
        "windows_detail": [],
    }


def plan_quality_chunk_stream_step(
    latent_frames,
    tokens_per_frame,
    text_tokens,
    max_mask_elements=None,
    target_chunk_frames=257,
    overlap_frames=16,
    safety_margin=0.9,
    previous_state=None,
    max_chunks=16,
    min_context_frames=8,
    anchors_per_chunk=3,
    max_anchors=24,
    token_ranges=None,
    segment_lengths=None,
    min_visible_frames=1,
    crossfade_curve="cosine",
    accepted_chunk_indices=None,
    rejected_chunk_indices=None,
):
    """Plan and preflight one resumable quality chunk page.

    This combines the pure pieces needed by a close-to-infinite scheduler:
    quality-bounded chunk paging, prior-page seam evaluation, continuity windows,
    and a capped memory-anchor bank. When authored prompt ranges are supplied,
    it also clips global prompt beats into the current page so schedulers can
    avoid rendering chunks that would build an empty Prompt Relay mask. The
    returned ``state`` is deliberately small and serializable so callers can
    persist it between ComfyUI queue items rather than materializing an entire
    hour-scale timeline.
    """
    state = dict(previous_state or {})
    config_fingerprint = _chunk_stream_config_fingerprint(
        latent_frames=latent_frames,
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        max_mask_elements=max_mask_elements,
        target_chunk_frames=target_chunk_frames,
        overlap_frames=overlap_frames,
        safety_margin=safety_margin,
        crossfade_curve=crossfade_curve,
        min_context_frames=min_context_frames,
        anchors_per_chunk=anchors_per_chunk,
        max_anchors=max_anchors,
        token_ranges=token_ranges,
        segment_lengths=segment_lengths,
        min_visible_frames=min_visible_frames,
    )
    previous_fingerprint = state.get("config_fingerprint")
    if previous_fingerprint is not None and previous_fingerprint != config_fingerprint:
        raise ValueError(
            "PromptRelay: chunk stream resume state was produced with different scheduler inputs; "
            "restart the stream or keep latent_frames/tokens_per_frame/text_tokens/mask cap/target/overlap/safety unchanged."
        )
    _validate_chunk_stream_resume_state(state)

    start_chunk = state.get("next_chunk")
    if start_chunk is None:
        start_chunk = 0

    page = plan_quality_temporal_chunk_page(
        latent_frames=latent_frames,
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        max_mask_elements=max_mask_elements,
        target_chunk_frames=target_chunk_frames,
        overlap_frames=overlap_frames,
        safety_margin=safety_margin,
        start_chunk=start_chunk,
        max_chunks=max_chunks,
    )
    previous_tail = state.get("last_chunk")
    evaluation = evaluate_temporal_chunk_page(
        page,
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        max_mask_elements=max_mask_elements,
        min_context_frames=min_context_frames,
        previous_tail_chunk=previous_tail,
    )

    chunks = list(page.get("chunks", []))
    continuity_chunks = ([previous_tail] if previous_tail is not None and chunks else []) + chunks
    try:
        continuity = plan_chunk_continuity_windows(continuity_chunks, context_frames=min_context_frames)
    except ValueError:
        continuity = []

    try:
        crossfade_windows = plan_chunk_crossfade_windows(
            continuity_chunks,
            min_context_frames=min_context_frames,
            curve=crossfade_curve,
        )
        crossfade_evaluation = evaluate_chunk_crossfade_windows(
            crossfade_windows,
            min_blend_frames=min_context_frames,
        )
    except ValueError as exc:
        crossfade_windows = []
        crossfade_evaluation = _unsafe_crossfade_evaluation(exc)

    if chunks:
        conditioning_anchors = select_chunk_conditioning_anchors(
            state.get("anchors", []),
            chunks[0],
            max_anchors=min(max_anchors, max(1, anchors_per_chunk * 2)),
        )
        anchor_bank = extend_chunk_memory_anchor_bank(
            state.get("anchors", []),
            chunks,
            anchors_per_chunk=anchors_per_chunk,
            max_anchors=max_anchors,
            chunk_index_offset=page.get("start_chunk") or 0,
            accepted_chunk_indices=accepted_chunk_indices,
            rejected_chunk_indices=rejected_chunk_indices,
        )
    else:
        conditioning_anchors = select_chunk_conditioning_anchors(
            state.get("anchors", []),
            state.get("last_chunk"),
            max_anchors=min(max_anchors, max(1, anchors_per_chunk * 2)),
        ) if state.get("last_chunk") is not None else {"anchors": [], "dropped_anchors": 0, "max_anchors": max_anchors}
        existing = list(state.get("anchors", []))[-max_anchors:]
        anchor_bank = {
            "anchors": existing,
            "dropped_anchors": max(0, len(state.get("anchors", [])) - len(existing)),
            "max_anchors": max_anchors,
            "added_anchors": 0,
            "chunk_index_offset": page.get("start_chunk") or 0,
        }

    prompt_windows = None
    prompt_schedule_required = token_ranges is not None or segment_lengths is not None
    if prompt_schedule_required:
        if token_ranges is None or segment_lengths is None:
            raise ValueError(
                "PromptRelay: token_ranges and segment_lengths must both be provided for chunk stream prompt scheduling."
            )
        prompt_windows = plan_chunk_prompt_windows(
            chunks,
            token_ranges=token_ranges,
            segment_lengths=segment_lengths,
            min_visible_frames=min_visible_frames,
        )
    renderable_prompt_windows = None
    prompt_schedule_evaluation = None
    prompt_seams = None
    prompt_seam_evaluation = None
    prompt_seam_shift_candidates = None
    prompt_safe_stitch_ranges = None
    if prompt_windows is not None:
        renderable_prompt_windows = sum(1 for window in prompt_windows if window.get("source_indices"))
        prompt_schedule_evaluation = evaluate_chunk_prompt_schedule(prompt_windows)
        prompt_seams = plan_chunk_prompt_seams(
            continuity_chunks,
            segment_lengths=segment_lengths,
            margin_frames=min_context_frames,
        )
        prompt_seam_evaluation = evaluate_chunk_prompt_seams(prompt_seams)
        prompt_seam_shift_candidates = plan_chunk_prompt_seam_shift_candidates(
            continuity_chunks,
            segment_lengths=segment_lengths,
            margin_frames=min_context_frames,
        )
        prompt_safe_stitch_ranges = plan_prompt_safe_chunk_stitch_ranges(
            continuity_chunks,
            prompt_seam_shift_candidates,
        )
        if prompt_safe_stitch_ranges.get("safe"):
            try:
                crossfade_windows = plan_chunk_crossfade_windows(
                    continuity_chunks,
                    min_context_frames=min_context_frames,
                    curve=crossfade_curve,
                    stitch_ranges=prompt_safe_stitch_ranges.get("ranges"),
                )
                crossfade_evaluation = evaluate_chunk_crossfade_windows(
                    crossfade_windows,
                    min_blend_frames=min_context_frames,
                )
            except ValueError as exc:
                crossfade_windows = []
                crossfade_evaluation = _unsafe_crossfade_evaluation(exc)

    try:
        stitch_ranges = (
            prompt_safe_stitch_ranges.get("ranges")
            if prompt_safe_stitch_ranges and prompt_safe_stitch_ranges.get("safe")
            else None
        )
        page_output_ranges = plan_chunk_page_output_ranges(
            chunks,
            previous_tail=previous_tail,
            stitch_ranges=stitch_ranges,
        )
    except ValueError as exc:
        page_output_ranges = {
            "ranges": [],
            "emitted_frames": 0,
            "append_start": None,
            "append_end": None,
            "safe": False,
            "error": str(exc),
        }

    page_complete = bool(page.get("complete"))
    state_next_chunk = page.get("next_chunk")
    if page_complete:
        # Keep the resume cursor monotonic after the final page.  A follow-up
        # scheduler tick should return an empty completed page, not restart at
        # chunk zero because the planner's public page cursor is None at EOF.
        state_next_chunk = page.get("total_chunks", state_next_chunk)

    _accepted_set, rejected_set = _chunk_stream_feedback_sets(accepted_chunk_indices, rejected_chunk_indices)
    rejected_current_indices = _chunk_stream_rejected_current_indices(
        chunks,
        page.get("start_chunk") or 0,
        rejected_set,
    )
    if rejected_current_indices:
        # A rejected chunk is the next retry cursor. Do not advance beyond it or
        # carry anchors from later chunks into future conditioning state; that
        # would silently skip a failed seam in a feedback-driven infinite run.
        state_next_chunk = rejected_current_indices[0]
        page_complete = False
        kept_anchors = [
            anchor for anchor in anchor_bank.get("anchors", [])
            if int(anchor.get("chunk_index", -1)) < rejected_current_indices[0]
        ]
        anchor_bank = {
            **anchor_bank,
            "anchors": kept_anchors,
            "dropped_anchors": anchor_bank.get("dropped_anchors", 0)
            + max(0, len(anchor_bank.get("anchors", [])) - len(kept_anchors)),
            "retry_from_chunk": rejected_current_indices[0],
        }

    clean_tail, clean_tail_index = _select_chunk_stream_tail(
        chunks,
        previous_tail,
        page.get("start_chunk") or 0,
        accepted_chunk_indices=accepted_chunk_indices,
        rejected_chunk_indices=rejected_chunk_indices,
    )

    next_state = {
        "next_chunk": state_next_chunk,
        "last_chunk": clean_tail,
        "last_chunk_index": clean_tail_index if clean_tail_index is not None else state.get("last_chunk_index"),
        "anchors": anchor_bank.get("anchors", []),
        "complete": page_complete,
        "total_chunks": page.get("total_chunks"),
        "config_fingerprint": config_fingerprint,
    }
    progress = summarize_quality_chunk_stream_progress(page, next_state)

    result = {
        "page": page,
        "evaluation": evaluation,
        "progress": progress,
        "continuity": continuity,
        "anchor_bank": anchor_bank,
        "conditioning_anchors": conditioning_anchors,
        "prompt_windows": prompt_windows,
        "renderable_prompt_windows": renderable_prompt_windows,
        "prompt_schedule_evaluation": prompt_schedule_evaluation,
        "prompt_seams": prompt_seams,
        "prompt_seam_evaluation": prompt_seam_evaluation,
        "prompt_seam_shift_candidates": prompt_seam_shift_candidates,
        "prompt_safe_stitch_ranges": prompt_safe_stitch_ranges,
        "crossfade_windows": crossfade_windows,
        "crossfade_evaluation": crossfade_evaluation,
        "page_output_ranges": page_output_ranges,
        "rejected_current_indices": rejected_current_indices,
        "state": next_state,
        "safe_to_render": (
            bool(chunks)
            and not rejected_current_indices
            and bool(evaluation.get("safe"))
            and bool(crossfade_evaluation.get("safe"))
            and bool(page_output_ranges.get("safe"))
            and (prompt_schedule_evaluation is None or bool(prompt_schedule_evaluation.get("safe")))
            and (prompt_seam_evaluation is None or bool(prompt_seam_evaluation.get("safe")))
        ),
    }
    result["render_decision"] = decide_quality_chunk_stream_render(result)
    return result


def _quality_metric_passes(value, rule):
    """Return whether one candidate quality metric satisfies a threshold rule."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    if "min" in rule and value < float(rule["min"]):
        return False
    if "max" in rule and value > float(rule["max"]):
        return False
    return True


def _quality_metric_score(value, rule):
    """Normalize one metric so higher is better for candidate ranking."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if not math.isfinite(value):
        return float("-inf")
    if "max" in rule and "min" not in rule:
        return -value
    return value


def evaluate_quality_chunk_candidates(candidates, thresholds=None, weights=None):
    """Evaluate generated chunk candidates before they can become memory.

    This is intentionally metric-agnostic. Callers can feed simple offline scores
    now, such as sharpness, flicker, transition similarity, or identity
    similarity, and later swap in CLIP/DINO/face/object metrics without changing
    the stream planner. A chunk is accepted only when at least one candidate for
    that chunk passes every configured threshold; chunks with no passing candidate
    are returned as rejected retry cursors so bad output does not poison future
    conditioning state.
    """
    thresholds = thresholds or {}
    weights = weights or {}
    rows = []
    passed_by_chunk = {}
    all_chunks = set()

    for idx, candidate in enumerate(candidates or []):
        metrics = dict(candidate.get("metrics") or {})
        try:
            chunk_index = int(candidate.get("chunk_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: candidate chunk_index must be an integer.") from exc
        all_chunks.add(chunk_index)

        failed = []
        score = 0.0
        for metric_name, rule in thresholds.items():
            rule = rule or {}
            if metric_name not in metrics or not _quality_metric_passes(metrics.get(metric_name), rule):
                failed.append(metric_name)
            metric_score = _quality_metric_score(metrics.get(metric_name), rule)
            if math.isfinite(metric_score):
                score += float(weights.get(metric_name, 1.0)) * metric_score

        status = "pass" if not failed else "fail"
        row = {
            "index": idx,
            "chunk_index": chunk_index,
            "candidate_id": candidate.get("candidate_id", idx),
            "metrics": metrics,
            "status": status,
            "score": score,
            "failed_metrics": failed,
        }
        rows.append(row)
        if status == "pass":
            current = passed_by_chunk.get(chunk_index)
            if current is None or (row["score"], -row["index"]) > (current["score"], -current["index"]):
                passed_by_chunk[chunk_index] = row

    rejected_chunks = sorted(chunk for chunk in all_chunks if chunk not in passed_by_chunk)
    accepted_candidates = [passed_by_chunk[idx] for idx in sorted(passed_by_chunk)]
    accepted_candidate = max(accepted_candidates, key=lambda row: (row["score"], -row["index"]), default=None)
    decision = "accept" if accepted_candidate is not None else ("regenerate" if rows else "no_candidates")

    return {
        "decision": decision,
        "accepted_candidate": accepted_candidate,
        "accepted_candidates": accepted_candidates,
        "accepted_chunk_indices": sorted(passed_by_chunk),
        "rejected_chunk_indices": rejected_chunks,
        "retry_from_chunk": min(rejected_chunks) if rejected_chunks else None,
        "candidates": rows,
        "thresholds": thresholds,
    }


def format_quality_chunk_candidate_diagnostics(report, max_candidates=6):
    """Return a bounded one-line quality gate summary for ComfyUI logs."""
    report = report or {}
    accepted = report.get("accepted_candidate")
    accepted_text = "none"
    chunk_text = "none"
    if accepted is not None:
        accepted_text = str(accepted.get("candidate_id"))
        chunk_text = str(accepted.get("chunk_index"))

    parts = [
        "quality_gate: decision={decision} accepted={accepted} chunk={chunk} rejected={rejected} retry_from={retry}".format(
            decision=report.get("decision", "unknown"),
            accepted=accepted_text,
            chunk=chunk_text,
            rejected=report.get("rejected_chunk_indices", []),
            retry=report.get("retry_from_chunk"),
        )
    ]

    candidates = list(report.get("candidates") or [])
    indexed = list(enumerate(candidates))
    total = len(indexed)
    if max_candidates and total > max_candidates:
        head_count = max_candidates // 2
        tail_count = max_candidates - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]

    previous_pos = None
    for pos, candidate in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} candidate(s) omitted ...")
        failed = candidate.get("failed_metrics") or []
        parts.append(
            "candidate{idx} id={candidate_id} chunk={chunk} status={status} failed={failed} score={score:.4g}".format(
                idx=pos,
                candidate_id=candidate.get("candidate_id"),
                chunk=candidate.get("chunk_index"),
                status=candidate.get("status"),
                failed=",".join(failed) if failed else "none",
                score=float(candidate.get("score", 0.0)),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def decide_quality_chunk_stream_render(step):
    """Return a small render/skip decision for a chunk stream scheduler tick."""
    step = step or {}
    page = step.get("page") or {}
    state = step.get("state") or {}
    chunks = page.get("chunks") or []
    checkpoint_next_chunk = state.get("next_chunk", page.get("next_chunk"))
    if not chunks:
        return {
            "status": "complete" if page.get("complete") else "empty_page",
            "reason": "no_chunks",
            "safe_to_render": False,
            "start_chunk": page.get("start_chunk"),
            "next_chunk": page.get("next_chunk"),
            "checkpoint_next_chunk": checkpoint_next_chunk,
        }

    rejected_current = list(step.get("rejected_current_indices") or [])
    if rejected_current:
        retry_from = min(int(idx) for idx in rejected_current)
        return {
            "status": "retry",
            "reason": "rejected_chunk_feedback",
            "safe_to_render": False,
            "start_chunk": page.get("start_chunk"),
            "next_chunk": retry_from,
            "checkpoint_next_chunk": retry_from,
            "retry_from_chunk": retry_from,
            "rejected_current_indices": sorted(int(idx) for idx in rejected_current),
        }

    checks = [
        (step.get("evaluation"), "chunk_page"),
        (step.get("crossfade_evaluation"), "crossfade"),
        (step.get("prompt_schedule_evaluation"), "prompt_schedule"),
        (step.get("prompt_seam_evaluation"), "prompt_seams"),
    ]
    for check, name in checks:
        if check is not None and not check.get("safe"):
            reason = "unsafe_" + name
            if name == "prompt_schedule" and step.get("renderable_prompt_windows") == 0:
                reason = "empty_prompt_window"
            return {
                "status": "skip",
                "reason": reason,
                "safe_to_render": False,
                "start_chunk": page.get("start_chunk"),
                "next_chunk": page.get("next_chunk"),
                "checkpoint_next_chunk": checkpoint_next_chunk,
            }

    safe = bool(step.get("safe_to_render"))
    return {
        "status": "render" if safe else "skip",
        "reason": "ready" if safe else "not_renderable",
        "safe_to_render": safe,
        "start_chunk": page.get("start_chunk"),
        "next_chunk": page.get("next_chunk"),
        "checkpoint_next_chunk": checkpoint_next_chunk,
    }


def build_quality_chunk_render_manifest(step):
    """Return the compact payload a long-form scheduler needs for this page.

    The stream step contains preflight data, diagnostics, and internal resume
    bookkeeping. A ComfyUI workflow runner only needs the render decision, input
    chunk ranges, prompt windows, stitch/crossfade metadata, clean conditioning
    anchors, and a safe checkpoint state to persist after the queue item.
    """
    step = step or {}
    page = step.get("page") or {}
    chunks = list(page.get("chunks") or [])
    decision = step.get("render_decision") or decide_quality_chunk_stream_render(step)
    state = dict(step.get("state") or {})
    # The fingerprint is useful for in-process validation, but it is an opaque
    # hash-like guard rather than operator-facing checkpoint state.
    state.pop("config_fingerprint", None)
    output_ranges = step.get("page_output_ranges") or {}
    return {
        "renderable": bool(decision.get("safe_to_render")),
        "decision": decision,
        "chunk_indices": [int(page.get("start_chunk") or 0) + idx for idx, _chunk in enumerate(chunks)],
        "input_ranges": [(int(chunk.get("start", 0)), int(chunk.get("end", 0))) for chunk in chunks],
        "output_ranges": list(output_ranges.get("ranges") or []),
        "append_range": (output_ranges.get("append_start"), output_ranges.get("append_end")),
        "prompt_windows": list(step.get("prompt_windows") or []),
        "conditioning_anchors": list((step.get("conditioning_anchors") or {}).get("anchors") or []),
        "crossfade_windows": list(step.get("crossfade_windows") or []),
        "state": state,
        "progress": dict(step.get("progress") or {}),
        "diagnostics": format_quality_chunk_stream_step_diagnostics(step),
    }


def build_quality_chunk_queue_plan(manifest):
    """Turn a render manifest into per-chunk queue work items.

    ``build_quality_chunk_render_manifest`` is page-level: it describes all chunks
    in one scheduler page plus the checkpoint to persist after that page. Queue
    runners still need chunk-local work items so they can render, trim, blend, and
    append each window deterministically. This helper keeps that contract pure and
    serializable without depending on ComfyUI internals.
    """
    manifest = manifest or {}
    decision = dict(manifest.get("decision") or {})
    if not manifest.get("renderable") or decision.get("status") != "render":
        return {
            "status": "skip",
            "decision": decision,
            "items": [],
            "append_range": manifest.get("append_range"),
            "checkpoint_state": dict(manifest.get("state") or {}),
            "diagnostics": manifest.get("diagnostics", ""),
        }

    chunk_indices = list(manifest.get("chunk_indices") or [])
    input_ranges = list(manifest.get("input_ranges") or [])
    output_ranges = list(manifest.get("output_ranges") or [])
    prompt_windows = list(manifest.get("prompt_windows") or [])
    crossfade_windows = list(manifest.get("crossfade_windows") or [])
    conditioning_anchors = list(manifest.get("conditioning_anchors") or [])

    items = []
    for position, chunk_index in enumerate(chunk_indices):
        input_range = input_ranges[position] if position < len(input_ranges) else (None, None)
        output = output_ranges[position] if position < len(output_ranges) else {}
        keep_start = output.get("keep_start")
        keep_end = output.get("keep_end")
        trim_start = output.get("trim_start")
        trim_end = output.get("trim_end")
        chunk_prompts = [
            window for window in prompt_windows
            if int(window.get("chunk_index", -1)) == int(chunk_index)
        ]
        chunk_crossfades = [
            window for window in crossfade_windows
            if int(window.get("prev_index", -1)) == int(chunk_index)
            or int(window.get("next_index", -1)) == int(chunk_index)
        ]
        items.append({
            "queue_index": position,
            "chunk_index": int(chunk_index),
            "input_range": tuple(input_range),
            "keep_range": (keep_start, keep_end),
            "local_trim": (trim_start, trim_end),
            "output_range": dict(output),
            "prompt_windows": chunk_prompts,
            "conditioning_anchors": conditioning_anchors,
            "crossfade_windows": chunk_crossfades,
        })

    return {
        "status": "ready",
        "decision": decision,
        "items": items,
        "append_range": manifest.get("append_range"),
        "checkpoint_state": dict(manifest.get("state") or {}),
        "diagnostics": manifest.get("diagnostics", ""),
    }



def summarize_quality_chunk_stream_progress(page, state=None):
    """Return bounded progress counters for a resumable chunk stream.

    Long-form schedulers often execute a few chunks per queue item. This helper
    reports cursor progress without materializing the full timeline, plus the
    furthest planned and clean-tail frame so callers can persist/checkpoint a
    compact resume state between ticks.
    """
    page = page or {}
    state = state or {}

    def to_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    total_chunks = max(to_int(page.get("total_chunks"), 0), 0)
    start_chunk = max(to_int(page.get("start_chunk"), 0), 0)
    next_chunk = state.get("next_chunk", page.get("next_chunk"))
    if next_chunk is None:
        next_chunk = total_chunks if (state.get("complete") or page.get("complete")) else start_chunk
    next_chunk = max(to_int(next_chunk, start_chunk), 0)
    if total_chunks:
        next_chunk = min(next_chunk, total_chunks)
    complete = bool(state.get("complete") or page.get("complete"))
    processed_chunks = total_chunks if complete and total_chunks else next_chunk
    remaining_chunks = max(total_chunks - processed_chunks, 0) if total_chunks else 0
    progress = (processed_chunks / total_chunks) if total_chunks else (1.0 if complete else 0.0)

    chunks = list(page.get("chunks") or [])
    last_planned = chunks[-1] if chunks else None
    last_clean = state.get("last_chunk") or last_planned or {}
    planned_until_frame = to_int((last_planned or {}).get("end"), 0)
    clean_until_frame = to_int((last_clean or {}).get("end"), planned_until_frame)

    return {
        "start_chunk": start_chunk,
        "next_chunk": next_chunk,
        "processed_chunks": processed_chunks,
        "remaining_chunks": remaining_chunks,
        "total_chunks": total_chunks,
        "progress": progress,
        "complete": complete,
        "planned_until_frame": planned_until_frame,
        "clean_until_frame": clean_until_frame,
    }


def format_quality_chunk_stream_progress(progress):
    """Return a compact checkpoint/progress line for chunk-stream diagnostics."""
    progress = progress or {}
    return (
        "chunk_progress: chunks={processed}/{total} remaining={remaining} "
        "progress={ratio:.6f} planned_until={planned_until} clean_until={clean_until} complete={complete}"
    ).format(
        processed=progress.get("processed_chunks", 0),
        total=progress.get("total_chunks", 0),
        remaining=progress.get("remaining_chunks", 0),
        ratio=float(progress.get("progress", 0.0)),
        planned_until=progress.get("planned_until_frame", 0),
        clean_until=progress.get("clean_until_frame", 0),
        complete=bool(progress.get("complete")),
    )


def format_quality_chunk_stream_step_diagnostics(step, max_chunks=4, max_anchors=4):
    """Return a bounded one-line status for a resumable quality chunk step."""
    page = (step or {}).get("page", {})
    evaluation = (step or {}).get("evaluation", {})
    anchor_bank = (step or {}).get("anchor_bank", {})
    conditioning_anchors = (step or {}).get("conditioning_anchors", {})
    prompt_windows = (step or {}).get("prompt_windows")
    crossfade_windows = (step or {}).get("crossfade_windows")
    decision = (step or {}).get("render_decision") or decide_quality_chunk_stream_render(step)
    parts = [
        "chunk_stream: safe_to_render={safe} start_chunk={start} render_status={status} reason={reason} next_chunk={next} checkpoint_next={checkpoint_next} complete={complete}".format(
            safe=bool((step or {}).get("safe_to_render")),
            status=decision.get("status"),
            reason=decision.get("reason"),
            start=page.get("start_chunk"),
            next=page.get("next_chunk"),
            checkpoint_next=decision.get("checkpoint_next_chunk"),
            complete=bool(page.get("complete")),
        ),
        format_temporal_chunk_plan_evaluation(evaluation),
        format_temporal_chunk_plan_diagnostics(page, max_chunks=max_chunks),
        format_quality_chunk_stream_progress((step or {}).get("progress")),
        format_chunk_conditioning_anchor_diagnostics(conditioning_anchors, max_anchors=max_anchors),
        format_chunk_memory_anchor_diagnostics(anchor_bank, max_anchors=max_anchors),
        format_chunk_crossfade_diagnostics(crossfade_windows, max_windows=max_chunks),
        format_chunk_crossfade_evaluation((step or {}).get("crossfade_evaluation")),
        format_chunk_page_output_diagnostics((step or {}).get("page_output_ranges"), max_chunks=max_chunks),
    ]
    if prompt_windows is not None:
        parts.append(format_chunk_prompt_schedule_diagnostics(prompt_windows, max_windows=max_chunks))
        parts.append(format_chunk_prompt_schedule_evaluation(
            (step or {}).get("prompt_schedule_evaluation"),
            max_windows=max_chunks,
        ))
        parts.append(format_chunk_prompt_seam_diagnostics(
            (step or {}).get("prompt_seams"),
            max_seams=max_chunks,
        ))
        parts.append(format_chunk_prompt_seam_shift_diagnostics(
            (step or {}).get("prompt_seam_shift_candidates"),
            max_candidates=max_chunks,
        ))
        parts.append(format_prompt_safe_chunk_stitch_diagnostics(
            (step or {}).get("prompt_safe_stitch_ranges"),
            max_chunks=max_chunks,
        ))
        parts.append(format_chunk_prompt_seam_evaluation(
            (step or {}).get("prompt_seam_evaluation"),
        ))
    return " | ".join(parts)


def format_temporal_chunk_plan_evaluation(evaluation):
    """Return a compact one-line preflight summary for chunk plans."""
    if evaluation is None:
        return "chunk_eval: safe=False error=missing_evaluation"

    page_bits = ""
    if "page_start_chunk" in evaluation or "boundary_handoff_status" in evaluation:
        page_bits = (
            " page_start={page_start} next_chunk={next_chunk} complete={complete} "
            "boundary_handoff={boundary_handoff} boundary_continuity={boundary_continuity} boundary_error={boundary_error}"
        ).format(
            page_start=evaluation.get("page_start_chunk"),
            next_chunk=evaluation.get("next_chunk"),
            complete=bool(evaluation.get("complete")),
            boundary_handoff=evaluation.get("boundary_handoff_status", "none"),
            boundary_continuity=evaluation.get("boundary_continuity_status", "none"),
            boundary_error=evaluation.get("boundary_error") or "none",
        )

    return (
        "chunk_eval: safe={safe} chunks={chunks} stitched_frames={stitched} "
        "peak_mask={peak} cap={cap} over_budget={over_budget} "
        "worst_handoff={handoff} worst_continuity={continuity}{page_bits} error={error}"
    ).format(
        safe=bool(evaluation.get("safe")),
        chunks=evaluation.get("chunks", 0),
        stitched=evaluation.get("stitched_frames", 0),
        peak=evaluation.get("peak_mask_elements", 0),
        cap=evaluation.get("max_mask_elements", 0),
        over_budget=len(evaluation.get("over_budget_chunks", [])),
        handoff=evaluation.get("worst_handoff", "none"),
        continuity=evaluation.get("worst_continuity", "none"),
        page_bits=page_bits,
        error=evaluation.get("error") or "none",
    )


def plan_chunk_stitch_ranges(chunks):
    """Return deterministic keep/trim ranges for overlapping chunk outputs.

    The chunk planner intentionally emits overlapping input windows so the caller
    can carry context across a boundary. This helper converts those windows into
    non-overlapping half-open output ranges by splitting each overlap at its
    midpoint. The returned rows keep the original chunk range plus global
    ``keep_start``/``keep_end`` and local ``trim_start``/``trim_end`` counts.

    It is pure bookkeeping: it does not blend frames or alter Prompt Relay masks.
    Use the discarded overlap tails/heads for crossfade or continuity checks in
    the stitching layer.
    """
    if not chunks:
        return []

    normalized = []
    for idx, chunk in enumerate(chunks):
        try:
            start = int(chunk["start"])
            end = int(chunk["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: chunk stitch ranges require start/end integers.") from exc
        if end <= start:
            raise ValueError("PromptRelay: chunk stitch ranges require positive-length chunks.")
        if normalized and start < normalized[-1]["start"]:
            raise ValueError("PromptRelay: chunks must be ordered by start frame.")
        if normalized and start > normalized[-1]["end"]:
            raise ValueError("PromptRelay: chunks must be contiguous or overlapping to stitch.")
        normalized.append({"index": idx, "start": start, "end": end, "length": end - start})

    boundaries = [normalized[0]["start"]]
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] >= previous["end"]:
            boundary = previous["end"]
        else:
            boundary = (current["start"] + previous["end"]) // 2
        boundaries.append(boundary)
    boundaries.append(normalized[-1]["end"])

    stitched = []
    for idx, chunk in enumerate(normalized):
        keep_start = max(chunk["start"], boundaries[idx])
        keep_end = min(chunk["end"], boundaries[idx + 1])
        if keep_end <= keep_start:
            raise ValueError("PromptRelay: overlap is too large to assign non-empty stitch ranges.")
        stitched.append({
            **chunk,
            "keep_start": keep_start,
            "keep_end": keep_end,
            "keep_length": keep_end - keep_start,
            "trim_start": keep_start - chunk["start"],
            "trim_end": chunk["end"] - keep_end,
        })

    return stitched


def plan_chunk_continuity_windows(chunks, context_frames=8):
    """Return per-chunk continuity windows around stitched keep ranges.

    Chunked long-form generation should not only trim overlaps; the discarded
    overlap head/tail is useful state for seam scoring, latent/noise carry, and
    image conditioning of the next window. This helper turns stitch ranges into
    bounded incoming/outgoing context windows without touching tensors.
    """
    stitched = plan_chunk_stitch_ranges(chunks)
    if not stitched:
        return []

    try:
        context = int(context_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: context_frames must be an integer.") from exc
    context = max(context, 0)

    windows = []
    last_idx = len(stitched) - 1
    for idx, chunk in enumerate(stitched):
        incoming_start = max(chunk["start"], chunk["keep_start"] - context)
        incoming_end = chunk["keep_start"]
        outgoing_start = chunk["keep_end"]
        outgoing_end = min(chunk["end"], chunk["keep_end"] + context)

        incoming_length = incoming_end - incoming_start
        outgoing_length = outgoing_end - outgoing_start
        expected_incoming = idx > 0 and context > 0
        expected_outgoing = idx < last_idx and context > 0

        if (expected_incoming and incoming_length <= 0) or (expected_outgoing and outgoing_length <= 0):
            status = "missing"
        elif (expected_incoming and incoming_length < context) or (expected_outgoing and outgoing_length < context):
            status = "partial"
        else:
            status = "ok"

        windows.append({
            "index": chunk["index"],
            "chunk_start": chunk["start"],
            "chunk_end": chunk["end"],
            "keep_start": chunk["keep_start"],
            "keep_end": chunk["keep_end"],
            "incoming_start": incoming_start,
            "incoming_end": incoming_end,
            "incoming_length": incoming_length,
            "outgoing_start": outgoing_start,
            "outgoing_end": outgoing_end,
            "outgoing_length": outgoing_length,
            "status": status,
        })

    return windows


def format_chunk_continuity_diagnostics(windows, max_chunks=8):
    """Return a bounded summary of per-chunk continuity carry windows."""
    if not windows:
        return "chunk_continuity: chunks=0"

    status_rank = {"missing": 3, "partial": 2, "ok": 1}
    worst = max((window.get("status", "unknown") for window in windows), key=lambda status: status_rank.get(status, 0))
    total = len(windows)
    indexed_windows = list(enumerate(windows))
    if max_chunks and total > max_chunks:
        head_count = max_chunks // 2
        tail_count = max_chunks - head_count
        indexed_windows = indexed_windows[:head_count] + indexed_windows[-tail_count:]

    parts = [f"chunk_continuity: chunks={total} worst={worst}"]
    previous_pos = None
    for pos, window in indexed_windows:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} chunk(s) omitted ...")
        parts.append(
            "chunk{idx}: keep=[{keep_start}:{keep_end}] in=[{in_start}:{in_end}]({in_len}) "
            "out=[{out_start}:{out_end}]({out_len}) status={status}".format(
                idx=window["index"],
                keep_start=window["keep_start"],
                keep_end=window["keep_end"],
                in_start=window["incoming_start"],
                in_end=window["incoming_end"],
                in_len=window["incoming_length"],
                out_start=window["outgoing_start"],
                out_end=window["outgoing_end"],
                out_len=window["outgoing_length"],
                status=window["status"],
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def plan_chunk_memory_anchors(chunks, anchors_per_chunk=3, max_anchors=24):
    """Select bounded clean-memory anchor frames from stitched chunk output.

    Infinite/closed-loop schedulers should not feed every previous generated
    frame forward. This pure planner picks deterministic frame indices from the
    stitched keep ranges only, then caps the bank to the most recent anchors so
    identity/style memory stays bounded and rejected overlap tails never become
    permanent conditioning state.
    """
    try:
        anchors_per_chunk = int(anchors_per_chunk)
        max_anchors = int(max_anchors)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: anchors_per_chunk and max_anchors must be integers.") from exc
    if anchors_per_chunk <= 0:
        raise ValueError("PromptRelay: anchors_per_chunk must be positive.")
    if max_anchors <= 0:
        raise ValueError("PromptRelay: max_anchors must be positive.")

    stitched = plan_chunk_stitch_ranges(chunks)
    anchors = []
    for chunk in stitched:
        keep_start = int(chunk["keep_start"])
        keep_end = int(chunk["keep_end"])
        keep_length = keep_end - keep_start
        if keep_length <= 0:
            continue

        if anchors_per_chunk == 1:
            frame_indices = [keep_start + keep_length // 2]
        else:
            frame_indices = []
            denom = anchors_per_chunk - 1
            for idx in range(anchors_per_chunk):
                offset = round(idx * (keep_length - 1) / denom)
                frame_indices.append(keep_start + offset)

        for frame_idx in dict.fromkeys(frame_indices):
            anchors.append({
                "chunk_index": chunk["index"],
                "frame": frame_idx,
                "keep_start": keep_start,
                "keep_end": keep_end,
            })

    dropped = max(0, len(anchors) - max_anchors)
    if dropped:
        anchors = anchors[-max_anchors:]
    return {"anchors": anchors, "dropped_anchors": dropped, "max_anchors": max_anchors}


def extend_chunk_memory_anchor_bank(
    existing_anchors,
    chunks,
    anchors_per_chunk=3,
    max_anchors=24,
    chunk_index_offset=0,
    accepted_chunk_indices=None,
    rejected_chunk_indices=None,
):
    """Append one chunk page to a bounded rolling memory-anchor bank.

    Paged/infinite schedulers cannot call ``plan_chunk_memory_anchors`` on the
    whole timeline forever. This helper keeps only the caller's existing anchor
    bank plus anchors from the next bounded chunk page, normalizes local page
    chunk indices into absolute chunk indices, and drops oldest anchors first.

    ``accepted_chunk_indices`` and ``rejected_chunk_indices`` are absolute chunk
    indices from a verification loop. When provided, only accepted chunks become
    future conditioning anchors. This keeps failed or manually rejected chunks
    from poisoning close-to-infinite runs.
    """
    try:
        chunk_index_offset = int(chunk_index_offset)
        anchors_per_chunk = int(anchors_per_chunk)
        max_anchors = int(max_anchors)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: chunk_index_offset, anchors_per_chunk, and max_anchors must be integers.") from exc
    if chunk_index_offset < 0:
        raise ValueError("PromptRelay: chunk_index_offset must be non-negative.")
    if anchors_per_chunk <= 0:
        raise ValueError("PromptRelay: anchors_per_chunk must be positive.")
    if max_anchors <= 0:
        raise ValueError("PromptRelay: max_anchors must be positive.")

    chunks = list(chunks or [])
    accepted_set = None if accepted_chunk_indices is None else {int(idx) for idx in accepted_chunk_indices}
    rejected_set = set() if rejected_chunk_indices is None else {int(idx) for idx in rejected_chunk_indices}
    if accepted_set is not None and accepted_set & rejected_set:
        raise ValueError("PromptRelay: accepted and rejected chunk indices must not overlap.")

    allowed_absolute_indices = []
    skipped_chunks = 0
    for local_idx, _chunk in enumerate(chunks):
        absolute_idx = chunk_index_offset + local_idx
        if accepted_set is not None and absolute_idx not in accepted_set:
            skipped_chunks += 1
            continue
        if absolute_idx in rejected_set:
            skipped_chunks += 1
            continue
        allowed_absolute_indices.append(absolute_idx)
    allowed_absolute_indices = set(allowed_absolute_indices)

    bank = []
    for anchor in existing_anchors or []:
        try:
            bank.append({
                "chunk_index": int(anchor["chunk_index"]),
                "frame": int(anchor["frame"]),
                "keep_start": int(anchor["keep_start"]),
                "keep_end": int(anchor["keep_end"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: existing anchors require chunk_index/frame/keep_start/keep_end integers.") from exc

    page_plan = plan_chunk_memory_anchors(
        chunks,
        anchors_per_chunk=anchors_per_chunk,
        max_anchors=max(max_anchors, max(1, len(chunks) * anchors_per_chunk)),
    )
    page_anchors = []
    for anchor in page_plan["anchors"]:
        absolute_idx = int(anchor["chunk_index"]) + chunk_index_offset
        if absolute_idx not in allowed_absolute_indices:
            continue
        page_anchors.append({
            **anchor,
            "chunk_index": absolute_idx,
        })

    combined = sorted(bank + page_anchors, key=lambda anchor: (anchor["frame"], anchor["chunk_index"]))
    dropped = max(0, len(combined) - max_anchors) + int(page_plan.get("dropped_anchors", 0))
    if len(combined) > max_anchors:
        combined = combined[-max_anchors:]

    return {
        "anchors": combined,
        "dropped_anchors": dropped,
        "max_anchors": max_anchors,
        "added_anchors": len(page_anchors),
        "chunk_index_offset": chunk_index_offset,
        "skipped_chunks": skipped_chunks,
        "accepted_chunk_indices": sorted(accepted_set) if accepted_set is not None else None,
        "rejected_chunk_indices": sorted(rejected_set),
    }


def format_chunk_memory_anchor_diagnostics(plan, max_anchors=8):
    """Return a bounded summary of clean-memory anchor selection."""
    anchors = list((plan or {}).get("anchors", []))
    if not anchors:
        return "chunk_memory: anchors=0 dropped=0"

    total = len(anchors)
    indexed = list(enumerate(anchors))
    if max_anchors and total > max_anchors:
        head_count = max_anchors // 2
        tail_count = max_anchors - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]

    parts = [
        "chunk_memory: anchors={count} dropped={dropped} cap={cap}".format(
            count=total,
            dropped=(plan or {}).get("dropped_anchors", 0),
            cap=(plan or {}).get("max_anchors", total),
        )
    ]
    previous_pos = None
    for pos, anchor in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} anchor(s) omitted ...")
        parts.append(
            "chunk{chunk}@frame{frame} keep=[{start}:{end}]".format(
                chunk=anchor["chunk_index"],
                frame=anchor["frame"],
                start=anchor["keep_start"],
                end=anchor["keep_end"],
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def select_chunk_conditioning_anchors(anchor_bank, chunk, max_anchors=4, max_age_frames=None):
    """Select recent clean anchors available before rendering one chunk.

    The rolling memory bank stores stitched frames from completed pages. A
    close-to-infinite scheduler should feed only a tiny, deterministic subset
    into the next chunk, otherwise identity/style conditioning grows without
    bound. Anchors before the chunk end are eligible so overlap carry-in frames
    from the previous rendered chunk can condition the current seam.
    """
    try:
        max_anchors = int(max_anchors)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: max_anchors must be an integer for conditioning anchors.") from exc
    if max_anchors <= 0:
        raise ValueError("PromptRelay: max_anchors must be positive for conditioning anchors.")
    if not chunk:
        return {"anchors": [], "dropped_anchors": 0, "max_anchors": max_anchors}
    try:
        chunk_start = int(chunk["start"])
        chunk_end = int(chunk["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: conditioning chunk requires integer start/end.") from exc
    if chunk_end <= chunk_start:
        raise ValueError("PromptRelay: conditioning chunk end must be greater than start.")

    max_age = None
    if max_age_frames is not None:
        try:
            max_age = int(max_age_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: max_age_frames must be an integer when provided.") from exc
        if max_age < 0:
            raise ValueError("PromptRelay: max_age_frames must be non-negative.")

    eligible = []
    for anchor in anchor_bank or []:
        try:
            normalized = {
                "chunk_index": int(anchor["chunk_index"]),
                "frame": int(anchor["frame"]),
                "keep_start": int(anchor["keep_start"]),
                "keep_end": int(anchor["keep_end"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: conditioning anchors require chunk_index/frame/keep_start/keep_end integers.") from exc
        if normalized["frame"] >= chunk_end:
            continue
        if max_age is not None and normalized["frame"] < chunk_start - max_age:
            continue
        eligible.append(normalized)

    eligible.sort(key=lambda anchor: (anchor["frame"], anchor["chunk_index"]))
    dropped = max(0, len(eligible) - max_anchors)
    if dropped:
        eligible = eligible[-max_anchors:]
    return {
        "anchors": eligible,
        "dropped_anchors": dropped,
        "max_anchors": max_anchors,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "max_age_frames": max_age,
    }


def format_chunk_conditioning_anchor_diagnostics(plan, max_anchors=4):
    """Return a bounded summary of anchors selected for the next render."""
    anchors = list((plan or {}).get("anchors", []))
    if not anchors:
        return "chunk_conditioning: anchors=0 dropped=0"
    indexed = list(enumerate(anchors))
    if max_anchors and len(indexed) > max_anchors:
        indexed = indexed[-max_anchors:]
    parts = [
        "chunk_conditioning: anchors={count} dropped={dropped} window=[{start}:{end}]".format(
            count=len(anchors),
            dropped=(plan or {}).get("dropped_anchors", 0),
            start=(plan or {}).get("chunk_start", "?"),
            end=(plan or {}).get("chunk_end", "?"),
        )
    ]
    for _pos, anchor in indexed:
        parts.append("chunk{chunk}@frame{frame}".format(chunk=anchor["chunk_index"], frame=anchor["frame"]))
    return "; ".join(parts)


def clip_segments_to_chunk(token_ranges, segment_lengths, chunk_start, chunk_end, min_visible_frames=1):
    """Clip global Prompt Relay segment timing to one temporal chunk.

    Long-form schedulers render bounded latent windows, but prompt beats are
    usually authored on the global timeline. This helper preserves token ranges
    and returns only the parts of positive-length segments visible inside a
    chunk, with lengths shifted to chunk-local time for ``build_segments``.
    Half-open ranges are used throughout: global segments ``[start, end)`` and
    chunks ``[chunk_start, chunk_end)``.
    """
    if len(token_ranges) != len(segment_lengths):
        raise ValueError(
            "PromptRelay: token_ranges and segment_lengths must have the same length to clip a chunk."
        )

    try:
        chunk_start = int(chunk_start)
        chunk_end = int(chunk_end)
        min_visible = int(min_visible_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: chunk_start, chunk_end, and min_visible_frames must be integers.") from exc
    if chunk_end <= chunk_start:
        raise ValueError("PromptRelay: chunk_end must be greater than chunk_start.")
    min_visible = max(min_visible, 1)

    clipped_token_ranges = []
    clipped_lengths = []
    source_indices = []
    global_ranges = []
    local_ranges = []
    cursor = 0

    for idx, (token_range, raw_length) in enumerate(zip(token_ranges, segment_lengths)):
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: segment_lengths must be integers before chunk clipping.") from exc
        if length <= 0:
            continue

        segment_start = cursor
        segment_end = cursor + length
        cursor = segment_end

        visible_start = max(segment_start, chunk_start)
        visible_end = min(segment_end, chunk_end)
        visible_length = visible_end - visible_start
        if visible_length < min_visible:
            continue

        clipped_token_ranges.append(token_range)
        clipped_lengths.append(visible_length)
        source_indices.append(idx)
        global_ranges.append((visible_start, visible_end))
        local_ranges.append((visible_start - chunk_start, visible_end - chunk_start))

    return {
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "token_ranges": clipped_token_ranges,
        "segment_lengths": clipped_lengths,
        "source_indices": source_indices,
        "global_ranges": global_ranges,
        "local_ranges": local_ranges,
    }


def format_chunk_prompt_window_diagnostics(window, max_segments=8):
    """Return a bounded summary of global prompt beats retained in a chunk."""
    source_indices = list(window.get("source_indices", []))
    global_ranges = list(window.get("global_ranges", []))
    local_ranges = list(window.get("local_ranges", []))
    lengths = list(window.get("segment_lengths", []))
    if not source_indices:
        return "chunk_prompt_window: segments=0"

    total = len(source_indices)
    indexed = list(enumerate(zip(source_indices, global_ranges, lengths)))
    if max_segments and total > max_segments:
        head_count = max_segments // 2
        tail_count = max_segments - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]

    parts = [
        "chunk_prompt_window: chunk=[{start}:{end}] segments={count}".format(
            start=window.get("chunk_start"),
            end=window.get("chunk_end"),
            count=total,
        )
    ]
    previous_pos = None
    for pos, (source_idx, global_range, length) in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} segment(s) omitted ...")
        local_range = local_ranges[pos] if pos < len(local_ranges) else (None, None)
        parts.append(
            "src{idx}=global[{start}:{end}] local[{local_start}:{local_end}] local_len={length}".format(
                idx=source_idx,
                start=global_range[0],
                end=global_range[1],
                local_start=local_range[0],
                local_end=local_range[1],
                length=length,
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def plan_chunk_prompt_windows(chunks, token_ranges, segment_lengths, min_visible_frames=1, max_windows=None):
    """Attach chunk-local Prompt Relay prompt windows to a chunk page.

    Long-form schedulers need a per-window view of the authored global prompt
    beats before they render each chunk. This helper clips global segment timing
    for each chunk and keeps the result bounded when called on a resumable page.
    It is intentionally pure metadata: callers still decide whether to render,
    crossfade, or skip chunks with no visible prompt beats.
    """
    try:
        min_visible = int(min_visible_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: min_visible_frames must be an integer.") from exc
    min_visible = max(min_visible, 1)
    if max_windows is not None:
        try:
            max_windows = int(max_windows)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: max_windows must be an integer when provided.") from exc
        if max_windows <= 0:
            raise ValueError("PromptRelay: max_windows must be positive when provided.")

    windows = []
    for idx, chunk in enumerate(chunks or []):
        if max_windows is not None and len(windows) >= max_windows:
            break
        try:
            chunk_start = int(chunk["start"])
            chunk_end = int(chunk["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: chunk prompt windows require chunk start/end integers.") from exc

        window = clip_segments_to_chunk(
            token_ranges,
            segment_lengths,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            min_visible_frames=min_visible,
        )
        window["chunk_index"] = int(chunk.get("index", idx)) if isinstance(chunk, dict) else idx
        window["chunk_length"] = chunk_end - chunk_start
        windows.append(window)
    return windows


def evaluate_chunk_prompt_schedule(windows, min_coverage_ratio=0.95):
    """Evaluate whether chunk-local prompt beats cover each render window.

    Long-form schedulers need more than a count of visible prompt segments: a
    page with a tiny sliver of a prompt at one edge is technically renderable
    but semantically under-conditioned. This helper merges each window's
    chunk-local prompt ranges, computes coverage, and flags empty or partial
    windows before a resumable render is launched.
    """
    try:
        min_coverage = float(min_coverage_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: min_coverage_ratio must be numeric.") from exc
    if not math.isfinite(min_coverage) or min_coverage < 0:
        raise ValueError("PromptRelay: min_coverage_ratio must be non-negative and finite.")

    rows = []
    total_visible = 0
    total_frames = 0
    for idx, window in enumerate(windows or []):
        try:
            chunk_length = int(window.get("chunk_length", int(window["chunk_end"]) - int(window["chunk_start"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: prompt schedule windows require chunk_length or chunk_start/chunk_end.") from exc
        if chunk_length <= 0:
            raise ValueError("PromptRelay: prompt schedule chunk lengths must be positive.")

        ranges = []
        for start, end in window.get("local_ranges", []):
            start = max(0, int(start))
            end = min(chunk_length, int(end))
            if end > start:
                ranges.append((start, end))
        ranges.sort()

        merged = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        visible = sum(end - start for start, end in merged)
        coverage = visible / chunk_length
        if visible == 0:
            status = "empty"
        elif coverage < min_coverage:
            status = "partial"
        else:
            status = "covered"

        rows.append({
            "chunk_index": window.get("chunk_index", idx),
            "chunk_start": window.get("chunk_start"),
            "chunk_end": window.get("chunk_end"),
            "chunk_length": chunk_length,
            "visible_frames": visible,
            "coverage_ratio": coverage,
            "status": status,
        })
        total_visible += visible
        total_frames += chunk_length

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    worst = "none"
    if rows:
        rank = {"empty": 3, "partial": 2, "covered": 1}
        worst = max((row["status"] for row in rows), key=lambda status: rank.get(status, 0))
    return {
        "windows": len(rows),
        "visible_frames": total_visible,
        "chunk_frames": total_frames,
        "coverage_ratio": (total_visible / total_frames) if total_frames else 0.0,
        "status_counts": counts,
        "empty_windows": counts.get("empty", 0),
        "partial_windows": counts.get("partial", 0),
        "worst_status": worst,
        "safe": worst in {"none", "covered"},
        "windows_detail": rows,
    }


def format_chunk_prompt_schedule_evaluation(evaluation, max_windows=4):
    """Return a bounded prompt coverage preflight summary."""
    if evaluation is None:
        return "chunk_prompt_eval: safe=False error=missing_evaluation"
    rows = list(evaluation.get("windows_detail", []))
    parts = [
        "chunk_prompt_eval: safe={safe} windows={windows} visible_frames={visible}/{frames} "
        "coverage={coverage:.3f} empty={empty} partial={partial} worst={worst}".format(
            safe=bool(evaluation.get("safe")),
            windows=evaluation.get("windows", 0),
            visible=evaluation.get("visible_frames", 0),
            frames=evaluation.get("chunk_frames", 0),
            coverage=float(evaluation.get("coverage_ratio", 0.0)),
            empty=evaluation.get("empty_windows", 0),
            partial=evaluation.get("partial_windows", 0),
            worst=evaluation.get("worst_status", "none"),
        )
    ]
    indexed = list(enumerate(rows))
    if max_windows and len(indexed) > max_windows:
        indexed = indexed[:max_windows // 2] + indexed[-(max_windows - max_windows // 2):]
    previous_pos = None
    for pos, row in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} window(s) omitted ...")
        parts.append(
            "chunk{idx}: coverage={coverage:.3f} visible={visible}/{length} status={status}".format(
                idx=row.get("chunk_index"),
                coverage=float(row.get("coverage_ratio", 0.0)),
                visible=row.get("visible_frames", 0),
                length=row.get("chunk_length", 0),
                status=row.get("status", "unknown"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def format_chunk_prompt_schedule_diagnostics(windows, max_windows=4, max_segments=4):
    """Return a bounded summary of prompt-beat coverage across chunk windows."""
    windows = list(windows or [])
    if not windows:
        return "chunk_prompt_schedule: windows=0 segments=0 empty_windows=0"

    total_segments = sum(len(window.get("source_indices", [])) for window in windows)
    empty_windows = sum(1 for window in windows if not window.get("source_indices"))
    total = len(windows)
    indexed_windows = list(enumerate(windows))
    if max_windows and total > max_windows:
        head_count = max_windows // 2
        tail_count = max_windows - head_count
        indexed_windows = indexed_windows[:head_count] + indexed_windows[-tail_count:]

    parts = [
        "chunk_prompt_schedule: windows={windows} segments={segments} empty_windows={empty}".format(
            windows=total,
            segments=total_segments,
            empty=empty_windows,
        )
    ]
    previous_pos = None
    for pos, window in indexed_windows:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} window(s) omitted ...")
        chunk_index = window.get("chunk_index", pos)
        segment_count = len(window.get("source_indices", []))
        parts.append(
            "chunk{idx}: segments={count} | {summary}".format(
                idx=chunk_index,
                count=segment_count,
                summary=format_chunk_prompt_window_diagnostics(window, max_segments=max_segments),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def plan_chunk_handoffs(chunks, min_context_frames=8):
    """Return seam diagnostics for adjacent long-form chunks.

    Long videos need more than non-overlapping stitch ranges: each boundary should
    have enough shared temporal context to compare/crossfade tail and head frames.
    This helper is pure diagnostics. It validates the chunk order through
    ``plan_chunk_stitch_ranges`` and then reports the overlap and context windows
    around every seam so a scheduler can reject or flag weak handoffs before a
    costly render.
    """
    stitched = plan_chunk_stitch_ranges(chunks)
    if len(stitched) < 2:
        return []

    try:
        min_context = int(min_context_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: min_context_frames must be an integer.") from exc
    min_context = max(min_context, 0)

    handoffs = []
    for previous, current in zip(stitched, stitched[1:]):
        seam = previous["keep_end"]
        overlap_start = max(previous["start"], current["start"])
        overlap_end = min(previous["end"], current["end"])
        overlap_length = max(overlap_end - overlap_start, 0)

        if overlap_length == 0:
            status = "hard_cut"
        elif overlap_length < min_context:
            status = "short_overlap"
        else:
            status = "ok"

        handoffs.append({
            "prev_index": previous["index"],
            "next_index": current["index"],
            "seam_frame": seam,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_length": overlap_length,
            "prev_context_start": max(previous["start"], seam - min_context),
            "prev_context_end": min(previous["end"], seam),
            "next_context_start": max(current["start"], seam),
            "next_context_end": min(current["end"], seam + min_context),
            "status": status,
        })

    return handoffs


def format_chunk_handoff_diagnostics(handoffs, max_handoffs=8):
    """Return a bounded, Comfy-log-friendly summary of long-form chunk seams."""
    if not handoffs:
        return "no chunk handoffs"

    total = len(handoffs)
    indexed_handoffs = list(enumerate(handoffs))
    if max_handoffs and total > max_handoffs:
        head_count = max_handoffs // 2
        tail_count = max_handoffs - head_count
        indexed_handoffs = indexed_handoffs[:head_count] + indexed_handoffs[-tail_count:]

    parts = []
    previous_idx = None
    for idx, handoff in indexed_handoffs:
        if previous_idx is not None and idx != previous_idx + 1:
            parts.append(f"... {idx - previous_idx - 1} handoff(s) omitted ...")
        parts.append(
            "handoff{idx}: chunk{prev}->{next} seam={seam} overlap=[{overlap_start}:{overlap_end}] "
            "overlap_len={overlap_len} prev_ctx=[{prev_ctx_start}:{prev_ctx_end}] "
            "next_ctx=[{next_ctx_start}:{next_ctx_end}] status={status}".format(
                idx=idx,
                prev=handoff["prev_index"],
                next=handoff["next_index"],
                seam=handoff["seam_frame"],
                overlap_start=handoff["overlap_start"],
                overlap_end=handoff["overlap_end"],
                overlap_len=handoff["overlap_length"],
                prev_ctx_start=handoff["prev_context_start"],
                prev_ctx_end=handoff["prev_context_end"],
                next_ctx_start=handoff["next_context_start"],
                next_ctx_end=handoff["next_context_end"],
                status=handoff["status"],
            )
        )
        previous_idx = idx
    return "; ".join(parts)


def plan_chunk_prompt_seams(chunks, segment_lengths, margin_frames=8):
    """Flag chunk stitch seams that land on or near prompt beat boundaries.

    Prompt Relay's local masks are strongest when semantic prompt changes are not
    hidden inside a crossfade seam. This pure preflight lets long-form schedulers
    notice when the deterministic stitch seam is too close to an authored prompt
    transition, so they can shift chunk starts or increase overlap before an
    expensive render.
    """
    try:
        margin = int(margin_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: margin_frames must be an integer.") from exc
    margin = max(margin, 0)

    boundaries = []
    cursor = 0
    for length in segment_lengths or []:
        try:
            length = int(length)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: segment_lengths must be integers for prompt seam diagnostics.") from exc
        if length <= 0:
            continue
        cursor += length
        boundaries.append(cursor)
    if boundaries:
        boundaries = boundaries[:-1]

    stitched = plan_chunk_stitch_ranges(chunks)
    seams = []
    for previous, current in zip(stitched, stitched[1:]):
        seam = int(previous["keep_end"])
        if boundaries:
            nearest = min(boundaries, key=lambda boundary: abs(boundary - seam))
            distance = abs(nearest - seam)
        else:
            nearest = None
            distance = None
        if nearest is None:
            status = "no_boundaries"
        elif distance == 0:
            status = "on_boundary"
        elif distance <= margin:
            status = "near_boundary"
        else:
            status = "clear"
        seams.append({
            "prev_index": previous["index"],
            "next_index": current["index"],
            "seam_frame": seam,
            "nearest_prompt_boundary": nearest,
            "distance_frames": distance,
            "margin_frames": margin,
            "status": status,
        })
    return seams


def evaluate_chunk_prompt_seams(seams, unsafe_statuses=("on_boundary", "near_boundary")):
    """Evaluate whether stitch seams avoid authored prompt transitions.

    Prompt changes landing inside overlap/crossfade regions are a high-risk
    source of semantic flicker in long-form chunking. Treat near/on-boundary
    seams as unsafe by default so resumable schedulers can shift the page or
    increase overlap before spending a render.
    """
    rows = list(seams or [])
    unsafe = set(unsafe_statuses or [])
    counts = {}
    for seam in rows:
        status = seam.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    rank = {"on_boundary": 5, "near_boundary": 4, "unknown": 3, "clear": 2, "no_boundaries": 1}
    worst = "none"
    if rows:
        worst = max((seam.get("status", "unknown") for seam in rows), key=lambda status: rank.get(status, 0))
    unsafe_count = sum(1 for seam in rows if seam.get("status", "unknown") in unsafe)
    return {
        "seams": len(rows),
        "status_counts": counts,
        "unsafe_seams": unsafe_count,
        "worst_status": worst,
        "safe": unsafe_count == 0,
    }


def plan_chunk_prompt_seam_shift_candidates(chunks, segment_lengths, margin_frames=8):
    """Suggest prompt-safe stitch seam positions inside chunk overlaps.

    A long-form scheduler does not have to split every overlap exactly at the
    midpoint. When that default seam lands on or near an authored prompt change,
    this helper searches the available overlap for the nearest frame that clears
    the configured prompt-boundary margin. It is pure metadata: callers can use
    ``recommended_seam_frame`` to trim/crossfade differently before rerendering
    or accepting a page.
    """
    seams = plan_chunk_prompt_seams(chunks, segment_lengths, margin_frames=margin_frames)
    if not seams:
        return []

    stitched = plan_chunk_stitch_ranges(chunks)
    boundaries = []
    cursor = 0
    for length in segment_lengths or []:
        try:
            length = int(length)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: segment_lengths must be integers for prompt seam shift planning.") from exc
        if length <= 0:
            continue
        cursor += length
        boundaries.append(cursor)
    boundaries = boundaries[:-1]

    try:
        margin = int(margin_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: margin_frames must be an integer.") from exc
    margin = max(margin, 0)

    def distance_to_nearest_boundary(frame):
        if not boundaries:
            return None, None
        nearest = min(boundaries, key=lambda boundary: abs(boundary - frame))
        return nearest, abs(nearest - frame)

    candidates = []
    for seam, previous, current in zip(seams, stitched, stitched[1:]):
        default_frame = int(seam["seam_frame"])
        overlap_start = max(previous["start"], current["start"])
        overlap_end = min(previous["end"], current["end"])
        if overlap_end <= overlap_start:
            candidates.append({
                **seam,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "recommended_seam_frame": default_frame,
                "shift_frames": 0,
                "recommended_nearest_prompt_boundary": seam.get("nearest_prompt_boundary"),
                "recommended_distance_frames": seam.get("distance_frames"),
                "recommendation_status": "no_overlap",
            })
            continue

        safe_frames = []
        for frame in range(overlap_start, overlap_end + 1):
            nearest, distance = distance_to_nearest_boundary(frame)
            if nearest is None or distance > margin:
                safe_frames.append((frame, nearest, distance))

        if not safe_frames:
            status = "blocked"
            recommended_frame = default_frame
            recommended_nearest = seam.get("nearest_prompt_boundary")
            recommended_distance = seam.get("distance_frames")
        else:
            recommended_frame, recommended_nearest, recommended_distance = min(
                safe_frames,
                key=lambda item: (abs(item[0] - default_frame), item[0]),
            )
            status = "clear" if recommended_frame == default_frame else "shifted_clear"

        candidates.append({
            **seam,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "recommended_seam_frame": recommended_frame,
            "shift_frames": recommended_frame - default_frame,
            "recommended_nearest_prompt_boundary": recommended_nearest,
            "recommended_distance_frames": recommended_distance,
            "recommendation_status": status,
        })
    return candidates


def plan_prompt_safe_chunk_stitch_ranges(chunks, seam_shift_candidates=None):
    """Return stitch ranges using safe prompt-seam recommendations when possible.

    ``plan_chunk_stitch_ranges`` splits overlap at the midpoint. When a prompt
    transition lands on that seam, ``plan_chunk_prompt_seam_shift_candidates``
    can recommend a nearby seam inside the same overlap. This helper converts
    those recommendations into concrete keep/trim ranges while falling back to
    the default seam for blocked recommendations. It stays metadata-only so
    renderers can apply the trim/crossfade plan without touching tensors here.
    """
    stitched = plan_chunk_stitch_ranges(chunks)
    if len(stitched) < 2:
        return {"ranges": stitched, "applied_shifts": 0, "blocked_shifts": 0, "safe": True}

    candidates = list(seam_shift_candidates or [])
    candidate_by_pair = {
        (candidate.get("prev_index"), candidate.get("next_index")): candidate
        for candidate in candidates
    }
    boundaries = [stitched[0]["start"]]
    applied = 0
    blocked = 0
    for previous, current in zip(stitched, stitched[1:]):
        candidate = candidate_by_pair.get((previous["index"], current["index"]))
        seam = int(previous["keep_end"])
        if candidate is not None:
            status = candidate.get("recommendation_status")
            if status in {"clear", "shifted_clear"}:
                recommended = int(candidate.get("recommended_seam_frame", seam))
                overlap_start = max(previous["start"], current["start"])
                overlap_end = min(previous["end"], current["end"])
                if overlap_start <= recommended <= overlap_end:
                    seam = recommended
                    if int(candidate.get("shift_frames", 0)) != 0:
                        applied += 1
                else:
                    blocked += 1
            elif status not in {None, "clear"}:
                blocked += 1
        boundaries.append(seam)
    boundaries.append(stitched[-1]["end"])

    ranges = []
    for idx, chunk in enumerate(stitched):
        keep_start = max(chunk["start"], boundaries[idx])
        keep_end = min(chunk["end"], boundaries[idx + 1])
        if keep_end <= keep_start:
            raise ValueError("PromptRelay: prompt-safe stitch shifts produced an empty keep range.")
        ranges.append({
            **chunk,
            "keep_start": keep_start,
            "keep_end": keep_end,
            "keep_length": keep_end - keep_start,
            "trim_start": keep_start - chunk["start"],
            "trim_end": chunk["end"] - keep_end,
        })

    return {
        "ranges": ranges,
        "applied_shifts": applied,
        "blocked_shifts": blocked,
        "safe": blocked == 0,
    }


def format_prompt_safe_chunk_stitch_diagnostics(plan, max_chunks=6):
    """Return bounded diagnostics for prompt-safe stitch ranges."""
    if plan is None:
        return "prompt_safe_stitch: safe=False error=missing_plan"
    ranges = list(plan.get("ranges", []))
    parts = [
        "prompt_safe_stitch: safe={safe} chunks={chunks} applied_shifts={applied} blocked_shifts={blocked}".format(
            safe=bool(plan.get("safe")),
            chunks=len(ranges),
            applied=plan.get("applied_shifts", 0),
            blocked=plan.get("blocked_shifts", 0),
        )
    ]
    indexed = list(enumerate(ranges))
    if max_chunks and len(indexed) > max_chunks:
        head_count = max_chunks // 2
        tail_count = max_chunks - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]
    previous_pos = None
    for pos, row in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} chunk(s) omitted ...")
        parts.append(
            "chunk{idx}: keep=[{start}:{end}] trim=({trim_start},{trim_end})".format(
                idx=row.get("index", pos),
                start=row.get("keep_start"),
                end=row.get("keep_end"),
                trim_start=row.get("trim_start"),
                trim_end=row.get("trim_end"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def format_chunk_prompt_seam_shift_diagnostics(candidates, max_candidates=6):
    """Return bounded diagnostics for prompt-safe stitch seam shifts."""
    candidates = list(candidates or [])
    if not candidates:
        return "chunk_prompt_seam_shifts: candidates=0 worst=none"

    rank = {"no_overlap": 5, "blocked": 4, "shifted_clear": 2, "clear": 1}
    worst = max(
        (candidate.get("recommendation_status", "unknown") for candidate in candidates),
        key=lambda status: rank.get(status, 3),
    )
    indexed = list(enumerate(candidates))
    if max_candidates and len(indexed) > max_candidates:
        head_count = max_candidates // 2
        tail_count = max_candidates - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]

    parts = [f"chunk_prompt_seam_shifts: candidates={len(candidates)} worst={worst}"]
    previous_pos = None
    for pos, candidate in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} candidate(s) omitted ...")
        parts.append(
            "shift{idx}: chunk{prev}->{next} seam={seam} recommended={recommended} "
            "shift={shift} overlap=[{start}:{end}] status={status}".format(
                idx=pos,
                prev=candidate.get("prev_index"),
                next=candidate.get("next_index"),
                seam=candidate.get("seam_frame"),
                recommended=candidate.get("recommended_seam_frame"),
                shift=candidate.get("shift_frames"),
                start=candidate.get("overlap_start"),
                end=candidate.get("overlap_end"),
                status=candidate.get("recommendation_status", "unknown"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def format_chunk_prompt_seam_evaluation(evaluation):
    """Return a compact prompt seam safety summary."""
    if evaluation is None:
        return "chunk_prompt_seam_eval: safe=False error=missing_evaluation"
    counts = evaluation.get("status_counts", {})
    return (
        "chunk_prompt_seam_eval: safe={safe} seams={seams} unsafe={unsafe} "
        "on_boundary={on_boundary} near_boundary={near_boundary} worst={worst}"
    ).format(
        safe=bool(evaluation.get("safe")),
        seams=evaluation.get("seams", 0),
        unsafe=evaluation.get("unsafe_seams", 0),
        on_boundary=counts.get("on_boundary", 0),
        near_boundary=counts.get("near_boundary", 0),
        worst=evaluation.get("worst_status", "none"),
    )


def format_chunk_prompt_seam_diagnostics(seams, max_seams=6):
    """Return a bounded summary of stitch seams vs prompt transitions."""
    seams = list(seams or [])
    if not seams:
        return "chunk_prompt_seams: seams=0 worst=none"

    rank = {"on_boundary": 4, "near_boundary": 3, "clear": 2, "no_boundaries": 1}
    worst = max((seam.get("status", "unknown") for seam in seams), key=lambda status: rank.get(status, 0))
    indexed = list(enumerate(seams))
    if max_seams and len(indexed) > max_seams:
        head_count = max_seams // 2
        tail_count = max_seams - head_count
        indexed = indexed[:head_count] + indexed[-tail_count:]

    parts = [f"chunk_prompt_seams: seams={len(seams)} worst={worst}"]
    previous_pos = None
    for pos, seam in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} seam(s) omitted ...")
        parts.append(
            "seam{idx}: chunk{prev}->{next} frame={frame} nearest_boundary={boundary} distance={distance} status={status}".format(
                idx=pos,
                prev=seam.get("prev_index"),
                next=seam.get("next_index"),
                frame=seam.get("seam_frame"),
                boundary=seam.get("nearest_prompt_boundary"),
                distance=seam.get("distance_frames"),
                status=seam.get("status", "unknown"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def plan_chunk_crossfade_windows(chunks, min_context_frames=8, curve="cosine", stitch_ranges=None):
    """Return deterministic overlap blend weights for adjacent chunk seams.

    Stitch ranges decide which frames survive. Close-to-infinite schedulers also
    need a bounded recipe for blending the discarded overlap tails/heads so seam
    handling is reproducible across queue items. This helper emits per-overlap
    frame weights: ``prev_weight`` fades out, ``next_weight`` fades in, and both
    sum to one. No tensors are touched; callers can apply the schedule to RGB,
    latents, optical-flow warped frames, or seam metrics.
    """
    if curve not in {"linear", "cosine"}:
        raise ValueError("PromptRelay: crossfade curve must be 'linear' or 'cosine'.")

    handoffs = plan_chunk_handoffs(chunks, min_context_frames=min_context_frames)
    seam_by_pair = {}
    if stitch_ranges is not None:
        ranges = list(stitch_ranges or [])
        for previous, current in zip(ranges, ranges[1:]):
            seam_by_pair[(previous.get("index"), current.get("index"))] = int(previous.get("keep_end"))

    windows = []
    for handoff in handoffs:
        pair = (handoff["prev_index"], handoff["next_index"])
        seam_frame = seam_by_pair.get(pair, handoff["seam_frame"])
        length = int(handoff.get("overlap_length", 0))
        frame_weights = []
        if length > 0:
            if length == 1:
                ramps = [0.5]
            else:
                ramps = []
                denom = length - 1
                for idx in range(length):
                    t = idx / denom
                    if curve == "cosine":
                        t = 0.5 - 0.5 * math.cos(math.pi * t)
                    ramps.append(t)
            for offset, next_weight in enumerate(ramps):
                prev_weight = 1.0 - next_weight
                frame_weights.append({
                    "frame": int(handoff["overlap_start"]) + offset,
                    "prev_weight": prev_weight,
                    "next_weight": next_weight,
                })

        windows.append({
            "prev_index": handoff["prev_index"],
            "next_index": handoff["next_index"],
            "seam_frame": seam_frame,
            "overlap_start": handoff["overlap_start"],
            "overlap_end": handoff["overlap_end"],
            "overlap_length": length,
            "curve": curve,
            "status": "blendable" if handoff["status"] == "ok" else handoff["status"],
            "frame_weights": frame_weights,
        })
    return windows


def format_chunk_crossfade_diagnostics(windows, max_windows=6, max_weights=3):
    """Return a bounded one-line summary of planned seam blend weights."""
    if not windows:
        return "chunk_crossfade: windows=0"

    status_rank = {"hard_cut": 3, "short_overlap": 2, "blendable": 1}
    worst = max((window.get("status", "unknown") for window in windows), key=lambda status: status_rank.get(status, 0))
    total = len(windows)
    indexed_windows = list(enumerate(windows))
    if max_windows and total > max_windows:
        head_count = max_windows // 2
        tail_count = max_windows - head_count
        indexed_windows = indexed_windows[:head_count] + indexed_windows[-tail_count:]

    parts = [f"chunk_crossfade: windows={total} worst={worst}"]
    previous_pos = None
    for pos, window in indexed_windows:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} crossfade(s) omitted ...")
        weights = list(window.get("frame_weights", []))
        sample = weights[:max_weights] if max_weights else []
        if max_weights and len(weights) > max_weights:
            sample = weights[:max_weights // 2] + weights[-(max_weights - max_weights // 2):]
        weight_text = ",".join(
            "f{frame}:{prev:.3f}/{next:.3f}".format(
                frame=row["frame"],
                prev=row["prev_weight"],
                next=row["next_weight"],
            )
            for row in sample
        ) or "none"
        parts.append(
            "xfade{idx}: chunk{prev}->{next} overlap=[{start}:{end}] len={length} curve={curve} status={status} weights={weights}".format(
                idx=pos,
                prev=window["prev_index"],
                next=window["next_index"],
                start=window["overlap_start"],
                end=window["overlap_end"],
                length=window["overlap_length"],
                curve=window["curve"],
                status=window["status"],
                weights=weight_text,
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def evaluate_chunk_crossfade_windows(windows, min_blend_frames=1, weight_tolerance=1e-6):
    """Evaluate whether planned overlap blends are safe to hand to a stitcher.

    Handoff checks catch gaps and too-short overlaps before rendering. This
    stricter crossfade preflight also verifies that each blendable seam has a
    long-enough weight ramp and that every per-frame pair is normalized. It is
    metadata-only, but gives infinite schedulers a cheap guard against accepting
    malformed seam recipes into rolling state.
    """
    try:
        min_blend = int(min_blend_frames)
        tolerance = float(weight_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: crossfade evaluation inputs must be numeric.") from exc
    min_blend = max(min_blend, 0)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("PromptRelay: weight_tolerance must be non-negative and finite.")

    rows = []
    status_counts = {}
    bad_weight_frames = 0
    for idx, window in enumerate(windows or []):
        status = window.get("status", "unknown")
        weights = list(window.get("frame_weights", []))
        overlap_length = int(window.get("overlap_length", len(weights)))
        normalized = True
        monotonic = True
        previous_next = None
        for row in weights:
            prev_weight = float(row.get("prev_weight", 0.0))
            next_weight = float(row.get("next_weight", 0.0))
            if abs((prev_weight + next_weight) - 1.0) > tolerance:
                normalized = False
                bad_weight_frames += 1
            if previous_next is not None and next_weight + tolerance < previous_next:
                monotonic = False
            previous_next = next_weight

        if status != "blendable":
            eval_status = status
        elif overlap_length < min_blend or len(weights) < min_blend:
            eval_status = "short_blend"
        elif not normalized:
            eval_status = "bad_weights"
        elif not monotonic:
            eval_status = "non_monotonic"
        else:
            eval_status = "ok"
        status_counts[eval_status] = status_counts.get(eval_status, 0) + 1
        rows.append({
            "index": idx,
            "prev_index": window.get("prev_index"),
            "next_index": window.get("next_index"),
            "overlap_length": overlap_length,
            "weights": len(weights),
            "status": eval_status,
        })

    rank = {"hard_cut": 6, "short_overlap": 5, "short_blend": 4, "bad_weights": 3, "non_monotonic": 2, "ok": 1}
    worst = "none"
    if rows:
        worst = max((row["status"] for row in rows), key=lambda status: rank.get(status, 0))
    return {
        "windows": len(rows),
        "min_blend_frames": min_blend,
        "bad_weight_frames": bad_weight_frames,
        "status_counts": status_counts,
        "worst_status": worst,
        "safe": worst in {"none", "ok"} and bad_weight_frames == 0,
        "windows_detail": rows,
    }


def format_chunk_crossfade_evaluation(evaluation, max_windows=4):
    """Return a compact safety summary for overlap blend plans."""
    if evaluation is None:
        return "chunk_crossfade_eval: safe=False error=missing_evaluation"
    rows = list(evaluation.get("windows_detail", []))
    parts = [
        "chunk_crossfade_eval: safe={safe} windows={windows} min_blend={min_blend} bad_weight_frames={bad} worst={worst}".format(
            safe=bool(evaluation.get("safe")),
            windows=evaluation.get("windows", 0),
            min_blend=evaluation.get("min_blend_frames", 0),
            bad=evaluation.get("bad_weight_frames", 0),
            worst=evaluation.get("worst_status", "none"),
        )
    ]
    indexed = list(enumerate(rows))
    if max_windows and len(indexed) > max_windows:
        indexed = indexed[:max_windows // 2] + indexed[-(max_windows - max_windows // 2):]
    previous_pos = None
    for pos, row in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} crossfade(s) omitted ...")
        parts.append(
            "xfade{idx}: chunk{prev}->{next} len={length} weights={weights} status={status}".format(
                idx=row.get("index", pos),
                prev=row.get("prev_index"),
                next=row.get("next_index"),
                length=row.get("overlap_length", 0),
                weights=row.get("weights", 0),
                status=row.get("status", "unknown"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def plan_chunk_page_output_ranges(chunks, previous_tail=None, stitch_ranges=None):
    """Return append ranges for the current page, excluding prior-tail context.

    Paged long-form renderers pass the previous clean tail into seam checks, but
    must not append that tail again. This maps stitch ranges from the combined
    ``[previous_tail] + chunks`` view back onto only the current page so a queue
    item can crop outputs deterministically before appending frames.
    """
    chunks = list(chunks or [])
    if not chunks:
        return {"ranges": [], "emitted_frames": 0, "append_start": None, "append_end": None, "safe": False}

    combined = ([previous_tail] if previous_tail is not None else []) + chunks
    ranges = list(stitch_ranges) if stitch_ranges is not None else plan_chunk_stitch_ranges(combined)
    offset = 1 if previous_tail is not None else 0
    if len(ranges) < offset + len(chunks):
        raise ValueError("PromptRelay: page output ranges require stitch ranges for every current chunk.")

    outputs = []
    emitted = 0
    for page_index, chunk in enumerate(chunks):
        row = dict(ranges[offset + page_index])
        keep_start = int(row["keep_start"])
        keep_end = int(row["keep_end"])
        source_start = int(chunk["start"])
        source_end = int(chunk["end"])
        if keep_end <= keep_start:
            raise ValueError("PromptRelay: page output range has no frames to append.")
        outputs.append({
            "page_index": page_index,
            "chunk_index": row.get("index", page_index),
            "chunk_start": source_start,
            "chunk_end": source_end,
            "keep_start": keep_start,
            "keep_end": keep_end,
            "keep_length": keep_end - keep_start,
            "trim_start": keep_start - source_start,
            "trim_end": source_end - keep_end,
        })
        emitted += keep_end - keep_start

    return {
        "ranges": outputs,
        "emitted_frames": emitted,
        "append_start": outputs[0]["keep_start"],
        "append_end": outputs[-1]["keep_end"],
        "safe": emitted > 0,
    }


def format_chunk_page_output_diagnostics(plan, max_chunks=4):
    """Return bounded diagnostics for current-page append/crop ranges."""
    if plan is None:
        return "chunk_page_output: safe=False error=missing_plan"
    ranges = list(plan.get("ranges") or [])
    parts = [
        "chunk_page_output: safe={safe} chunks={chunks} emitted={emitted} append=[{start}:{end}]".format(
            safe=bool(plan.get("safe")),
            chunks=len(ranges),
            emitted=plan.get("emitted_frames", 0),
            start=plan.get("append_start"),
            end=plan.get("append_end"),
        )
    ]
    indexed = list(enumerate(ranges))
    if max_chunks and len(indexed) > max_chunks:
        indexed = indexed[:max_chunks // 2] + indexed[-(max_chunks - max_chunks // 2):]
    previous_pos = None
    for pos, row in indexed:
        if previous_pos is not None and pos != previous_pos + 1:
            parts.append(f"... {pos - previous_pos - 1} chunk(s) omitted ...")
        parts.append(
            "page_chunk{idx}: keep=[{start}:{end}] trim=({trim_start},{trim_end})".format(
                idx=row.get("page_index", pos),
                start=row.get("keep_start"),
                end=row.get("keep_end"),
                trim_start=row.get("trim_start"),
                trim_end=row.get("trim_end"),
            )
        )
        previous_pos = pos
    return "; ".join(parts)


def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention (integer frame indexing)."""
    _check_mask_size(Lq, Lk, dtype, "video")
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    """Penalty matrix for queries that don't map to integer frames (e.g. LTXAV audio tokens)."""
    _check_mask_size(Lq, Lk, dtype, "scaled/audio")
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        audio_midpoint = seg.get("midpoint_audio", seg["midpoint"])
        d = (query_frames[:, None] - audio_midpoint).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def _as_int(value):
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _flatten_grid_sizes(grid_sizes):
    """Return one [T,H,W]-like row from common Comfy/LTX grid_sizes shapes."""
    if grid_sizes is None:
        return None

    if hasattr(grid_sizes, "tolist"):
        grid_sizes = grid_sizes.tolist()

    while isinstance(grid_sizes, (list, tuple)) and grid_sizes and isinstance(grid_sizes[0], (list, tuple)):
        grid_sizes = grid_sizes[0]

    return grid_sizes


def _grid_tokens_per_frame(grid_sizes, fallback_tokens_per_frame):
    """Return video tokens per frame from Comfy transformer grid metadata.

    Comfy can pass `grid_sizes` as `[frames, height, width]`, as batched
    `[[frames, height, width], ...]`, or as a tensor/array shaped `[B,3]`.
    Prompt Relay only needs the spatial token count for the current batch
    shape. Falling back is safer than misrouting LTXV AV attention when a custom
    node omits or truncates metadata.
    """
    grid = _flatten_grid_sizes(grid_sizes)
    if grid is None or len(grid) < 3:
        return fallback_tokens_per_frame

    try:
        tokens_per_frame = _as_int(grid[-2]) * _as_int(grid[-1])
    except (TypeError, ValueError):
        return fallback_tokens_per_frame
    return tokens_per_frame if tokens_per_frame > 0 else fallback_tokens_per_frame


def classify_attention_route(Lq, Lk, max_token_idx, latent_frames, video_tokens_per_frame):
    """Classify whether Prompt Relay should mask an attention call.

    Returns:
        None: skip the attention call.
        "video": video text cross-attention with integer frame mapping.
        "scaled": non-video text cross-attention, used by LTXV audio text attention.
    """
    if Lq == Lk:
        return None

    video_lq = latent_frames * video_tokens_per_frame
    if Lk == video_lq or Lk < max_token_idx:
        return None

    return "video" if Lq == video_lq else "scaled"


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """Closure: mask_fn(q, k, transformer_options) -> additive mask or None.

    The returned function carries a small ``prompt_relay_diagnostics`` dict so
    tests and Comfy logs can prove whether LTX video attention and scaled AV
    attention actually received Prompt Relay masks.
    """
    if not q_token_idx:
        raise ValueError(
            "PromptRelay: no positive-length relay segments were built. "
            "Check that the latent has frames and segment_lengths leaves at least one segment in range."
        )

    cache = {}
    diagnostics = {
        "calls": 0,
        "applied": {"video": 0, "scaled": 0},
        "skipped": {"uncond": 0, "route": 0},
        "last": None,
    }
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(q, k, transformer_options):
        diagnostics["calls"] += 1
        Lq, Lk = q.shape[1], k.shape[1]

        # Only apply on conditional pass — not unconditional (negative prompt)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            diagnostics["skipped"]["uncond"] += 1
            diagnostics["last"] = {"mode": None, "reason": "uncond", "Lq": Lq, "Lk": Lk}
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = _grid_tokens_per_frame(grid_sizes, fallback_tokens_per_frame)
        mode = classify_attention_route(Lq, Lk, max_token_idx, latent_frames, video_tpf)
        if mode is None:
            diagnostics["skipped"]["route"] += 1
            diagnostics["last"] = {"mode": None, "reason": "route", "Lq": Lq, "Lk": Lk, "video_tpf": video_tpf}
            return None

        key = (Lq, Lk, mode, video_tpf, q.device, q.dtype)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            log.info(
                "[PromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, latent_frames=%d, video_tpf=%d, nonzero=%d/%d",
                mode, Lq, Lk, latent_frames, video_tpf, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost

        diagnostics["applied"][mode] += 1
        diagnostics["last"] = {"mode": mode, "Lq": Lq, "Lk": Lk, "video_tpf": video_tpf}
        return cache[key].to(q.dtype)

    mask_fn.prompt_relay_diagnostics = diagnostics
    return mask_fn


SIGMA_MODE_UPSTREAM = "upstream_compat"
SIGMA_MODE_PAPER = "paper_boundary"


def calculate_prompt_relay_sigma(segment_length, free_window, epsilon, sigma_mode=SIGMA_MODE_UPSTREAM):
    """Return temporal decay sigma for a segment.

    The default `upstream_compat` mode preserves the current ComfyUI-PromptRelay
    behavior. `paper_boundary` is opt-in and follows Prompt Relay Eq. 4 so the
    retained prior at the segment endpoint equals epsilon:
        exp(-((L/2 - w)^2) / (2 sigma^2)) = epsilon
    """
    if not 0 < epsilon < 1:
        return 0.1448

    if sigma_mode == SIGMA_MODE_PAPER:
        endpoint_distance = max(float(segment_length) / 2.0, 0.0)
        penalty_distance = max(endpoint_distance - float(free_window), 0.0)
        if penalty_distance <= 0:
            return 1e-6
        return penalty_distance / math.sqrt(2.0 * math.log(1.0 / epsilon))

    return 1.0 / math.log(1.0 / epsilon)


def _local_token_span(local_token_idx):
    """Return a compact inclusive/exclusive token span for tensors or test doubles."""
    if hasattr(local_token_idx, "tolist"):
        values = local_token_idx.tolist()
    else:
        values = list(local_token_idx)
    if not values:
        return "[]"
    return f"[{int(values[0])}:{int(values[-1]) + 1}]"


def format_segment_diagnostics(q_token_idx, max_segments=12):
    """Return a concise, log-friendly summary of Prompt Relay segment timing.

    This is intentionally plain text so a ComfyUI console log is enough to verify
    segment timing, paper-boundary sigma, and any LTX AV audio frame offset. For
    long-form runs with many prompt beats, the summary is bounded to avoid noisy
    multi-kilobyte Comfy logs while still showing the head/tail timing shape.
    """
    parts = []

    total = len(q_token_idx)
    indexed_segments = list(enumerate(q_token_idx))
    if max_segments and total > max_segments:
        head_count = max_segments // 2
        tail_count = max_segments - head_count
        indexed_segments = indexed_segments[:head_count] + indexed_segments[-tail_count:]

    previous_idx = None
    for idx, seg in indexed_segments:
        if previous_idx is not None and idx != previous_idx + 1:
            parts.append(f"... {idx - previous_idx - 1} segment(s) omitted ...")
        parts.append(
            "seg{idx}: tokens={tokens} mid={mid:.3f} win={win:.3f} sigma={sigma:.6g} str={strength:.3f} "
            "audio_mid={audio_mid:.3f} audio_win={audio_win:.3f} audio_sigma={audio_sigma:.6g} audio_str={audio_strength:.3f}".format(
                idx=idx,
                tokens=_local_token_span(seg["local_token_idx"]),
                mid=float(seg["midpoint"]),
                win=float(seg["window"]),
                sigma=float(seg["sigma"]),
                strength=float(seg.get("strength", 1.0)),
                audio_mid=float(seg.get("midpoint_audio", seg["midpoint"])),
                audio_win=float(seg.get("window_audio", seg["window"])),
                audio_sigma=float(seg.get("sigma_audio", seg["sigma"])),
                audio_strength=float(seg.get("strength_audio", 1.0)),
            )
        )
        previous_idx = idx
    return "; ".join(parts)


def _segment_relay_knobs(relay_options):
    opts = relay_options or {}
    return {
        "video_strength": opts.get("video_strength", 1.0),
        "video_window_scale": opts.get("video_window_scale", 1.0),
        "audio_epsilon": opts.get("audio_epsilon"),
        "audio_strength": opts.get("audio_strength", 1.0),
        "audio_window_scale": opts.get("audio_window_scale", 1.0),
        "audio_frame_offset_frames": opts.get("audio_frame_offset_frames", 0.0),
        "sigma_mode": opts.get("sigma_mode", SIGMA_MODE_UPSTREAM),
    }


def _build_segment_metadata(tok_start, tok_end, frame_start, length, epsilon, knobs):
    midpoint = (2 * int(frame_start) + int(length)) // 2
    base_window = max(int(length) // 2 - 2, 0)
    video_window = max(base_window * knobs["video_window_scale"], 0.0)
    audio_window = max(base_window * knobs["audio_window_scale"], 0.0)
    sigma = calculate_prompt_relay_sigma(length, video_window, epsilon, knobs["sigma_mode"])
    sigma_audio = (
        calculate_prompt_relay_sigma(length, audio_window, knobs["audio_epsilon"], knobs["sigma_mode"])
        if knobs["audio_epsilon"] is not None and 0 < knobs["audio_epsilon"] < 1
        else sigma
    )
    return {
        "local_token_idx": torch.arange(tok_start, tok_end),
        "midpoint": midpoint,
        "midpoint_audio": midpoint + knobs["audio_frame_offset_frames"],
        "window": video_window,
        "sigma": sigma,
        "strength": knobs["video_strength"],
        "window_audio": audio_window,
        "sigma_audio": sigma_audio,
        "strength_audio": knobs["audio_strength"],
    }


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    """Per-segment metadata for the temporal penalty.

    relay_options (optional dict) overrides per-stream knobs:
        video_strength, video_window_scale,
        audio_epsilon, audio_strength, audio_window_scale, audio_frame_offset_frames, sigma_mode
    Audio knobs only affect architectures whose cross-attention takes the scaled
    (non-integer-frame) path — currently LTX audio_attn2. The audio frame offset
    shifts only that scaled path, useful when LTX AV speech onset leads/lags the
    visual segment boundary.
    """
    knobs = _segment_relay_knobs(relay_options)

    if relay_options:
        log.info(
            "[PromptRelay] Advanced options active — sigma_mode=%s | "
            "video: strength=%.3f window_scale=%.3f | "
            "audio: epsilon=%s strength=%.3f window_scale=%.3f frame_offset=%.3f",
            knobs["sigma_mode"],
            knobs["video_strength"], knobs["video_window_scale"],
            f"{knobs['audio_epsilon']:.4f}" if knobs["audio_epsilon"] is not None else "inherit",
            knobs["audio_strength"], knobs["audio_window_scale"], knobs["audio_frame_offset_frames"],
        )

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            continue
        q_token_idx.append(_build_segment_metadata(tok_start, tok_end, frame_cursor, L, epsilon, knobs))
        frame_cursor += L

    return q_token_idx


def build_chunk_segments(chunk_window, epsilon=1e-3, relay_options=None):
    """Build segment metadata from ``clip_segments_to_chunk`` while preserving gaps.

    Chunked schedulers render a local window of the global timeline. A segment
    that starts midway through that chunk must keep its chunk-local midpoint;
    otherwise the attention anchor drifts to frame zero and seams become harder
    to diagnose. ``clip_segments_to_chunk`` provides explicit ``local_ranges``
    for that case, and this helper converts them to normal Prompt Relay segment
    metadata.
    """
    token_ranges = list((chunk_window or {}).get("token_ranges", []))
    local_ranges = list((chunk_window or {}).get("local_ranges", []))
    if len(token_ranges) != len(local_ranges):
        raise ValueError("PromptRelay: chunk_window must include one local range per token range.")

    knobs = _segment_relay_knobs(relay_options)
    q_token_idx = []
    for (tok_start, tok_end), (local_start, local_end) in zip(token_ranges, local_ranges):
        try:
            local_start = int(local_start)
            local_end = int(local_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: chunk local ranges must be integer pairs.") from exc
        length = local_end - local_start
        if length <= 0:
            continue
        q_token_idx.append(_build_segment_metadata(tok_start, tok_end, local_start, length, epsilon, knobs))
    return q_token_idx


def get_raw_tokenizer(clip):
    """Extract the raw SPiece/HF tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """Tokenize global + space-prefixed locals; return (full_prompt, per-local token ranges).

    Uses incremental tokenization to avoid SentencePiece context-dependency issues.
    """
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """Validate or auto-distribute segment frame counts, capped to fit within latent_frames."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        # ceil division — matches reference implementation
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective
