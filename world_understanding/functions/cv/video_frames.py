# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract still frames from a video for VLM consumption."""

import hashlib
import logging
import math
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

MAX_FRAMES = 64
DEFAULT_FRAMES = 8


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    n: int = DEFAULT_FRAMES,
    stride_ms: int | None = None,
) -> list[Path]:
    """Sample up to ``n`` frames from a video and write them as PNGs.

    Frames are sampled either evenly across the video duration (when
    ``stride_ms`` is None) or every ``stride_ms`` starting at ``t=0``, capped
    at ``n`` frames in either case. Output filenames embed the frame index
    and the *actual* post-decode timestamp in milliseconds (truncated to int)
    so callers can correlate frames back to the video.

    When the container does not report usable ``frame_count`` and ``fps``
    metadata (e.g. fragmented or partially-downloaded MP4, some VFR webm),
    or when the seek path can't prove progress because the backend's
    position counters stay stuck, the function falls back to a sequential
    read of up to ``n`` frames from the start instead of raising. In
    fallback even-mode (no ``stride_ms``) the result is the first ``n``
    decodable frames clustered at the head of the stream rather than
    spread across the unknown duration. Very short clips
    (``frame_count <= n`` in even mode) also take the sequential path
    because midpoint seeks land past the only frame's PTS on a
    1-frame source and degrade quickly on short 2- or 4-frame sources.

    Args:
        video_path: Path to the input video file (.mp4, .mov, .webm, ...).
        output_dir: Directory where PNG frames are written. Created if missing.
            Partial files written before a mid-loop failure are removed; if the
            directory was created by this call and ends up empty, it is
            removed too.
        n: Maximum number of frames to extract. Must be in [1, 64]. The
            actual return count may be lower than ``n``: even-spacing mode
            caps at the source ``frame_count`` (very short clips return all
            their frames sequentially); stride mode drops timestamps that
            fall at or beyond the video duration; content-hash dedup
            coalesces byte-identical decoded frames so sub-frame-period
            strides and static-scene sources can collapse to as few as one
            frame; and both modes break early if the decoder runs out of
            frames.
        stride_ms: Optional sampling interval in milliseconds, with the first
            sample at ``t=0`` and subsequent samples at ``stride_ms``,
            ``2*stride_ms``, ... . When None, frames are spread evenly across
            the duration.

    Returns:
        List of frame paths in chronological order. The caller owns the files
        and is responsible for cleanup on the success path.

    Raises:
        ValueError: If ``n`` is outside [1, 64] or ``stride_ms`` is non-positive.
        FileNotFoundError: If ``video_path`` does not exist.
        RuntimeError: With one of:
            - "Could not open video ..." — corrupt or unsupported codec.
            - "Could not reopen video ..." — the seek path returned no
              frames and the defensive close-and-reopen for the sequential
              fallback also failed (rare; e.g. file was deleted between
              the original open and the reopen).
            - "No frames could be decoded ..." — open succeeded but every read
              returned no frame.
            - "Failed to write frame ..." — ``cv2.imwrite`` returned False
              (full disk, permissions, codec).
    """
    if not 1 <= n <= MAX_FRAMES:
        raise ValueError(f"n must be in [1, {MAX_FRAMES}], got {n}")
    if stride_ms is not None and stride_ms <= 0:
        raise ValueError(f"stride_ms must be positive, got {stride_ms}")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    written: list[Path] = []
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video (corrupt or unsupported): {video_path}"
            )

        # Read raw metadata before casting — some backends report frame_count
        # as +inf or NaN, which would crash int() before the metadata_ok
        # check could route us to the sequential fallback.
        raw_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        metadata_ok = (
            math.isfinite(raw_frame_count)
            and raw_frame_count > 0
            and math.isfinite(fps)
            and fps > 0
        )

        if metadata_ok:
            frame_count = int(raw_frame_count)
            # Short-clip guard: when the source has at most ``n`` frames in
            # even mode, midpoint sampling is unsafe — on a 1-frame mp4 the
            # midpoint of the (single) segment lands past the only frame's
            # PTS, the seek silently advances the cursor past EOF, and on
            # some backends even ``CAP_PROP_POS_FRAMES=0`` cannot rewind
            # afterwards. Skip the seek path entirely and let the
            # sequential reader take the first ``n`` decoded frames from
            # the head — which is exactly what the docstring promises for
            # any clip whose unique frame count is <= ``n``.
            short_clip = stride_ms is None and frame_count <= n
            if short_clip:
                logger.info(
                    "Short clip (frame_count=%d <= n=%d); using sequential read: %s",
                    frame_count,
                    n,
                    video_path,
                )
                _read_sequential(
                    capture,
                    n=n,
                    stride_ms=stride_ms,
                    output_dir=output_dir,
                    paths=written,
                    fps_hint=fps,
                )
            else:
                duration_ms = (frame_count / fps) * 1000.0
                timestamps_ms = _pick_timestamps(
                    duration_ms=duration_ms, n=n, stride_ms=stride_ms
                )
                seek_completed = _seek_and_save(
                    capture, timestamps_ms, output_dir, written
                )
                # Trigger the sequential fallback when (a) the seek path
                # produced zero frames, or (b) it broke early due to a
                # backend seek-rejection AND we got fewer frames than the
                # caller requested. Case (b) covers the partial-success
                # backend that accepts ``set(POS_MSEC, 0)`` as a no-op
                # success but rejects every later non-zero seek: without
                # this, ``stride_ms=500, n=8`` on such a backend would
                # silently return only the t=0 frame instead of falling
                # back to a sequential read that can fill the rest.
                # Discard the partial seek output before retrying so the
                # final return is a coherent sequence of either
                # all-seek or all-sequential samples (mixing the two
                # produces a result the caller can't explain by either
                # the seek schedule or the sequential synth-clock).
                if not written or (not seek_completed and len(written) < n):
                    if written:
                        logger.info(
                            "Seek path partially completed (%d/%d frames) "
                            "before seek failure; discarding partial output "
                            "and retrying sequentially: %s",
                            len(written),
                            n,
                            video_path,
                        )
                        for p in written:
                            try:
                                p.unlink()
                            except OSError:
                                pass
                        written.clear()
                    else:
                        logger.info(
                            "Seek path produced no frames; reopening for "
                            "sequential read: %s",
                            video_path,
                        )
                    # Backend rejected ``CAP_PROP_POS_MSEC`` seeks (some
                    # FFmpeg builds return False on every set() even
                    # though sequential read() would still decode), or
                    # every seek landed past a usable frame. Reopen the
                    # capture and retry sequentially: ``CAP_PROP_POS_FRAMES
                    # = 0`` is unreliable after a seek-past-EOF on some
                    # backends, so a fresh handle is the only robust way
                    # to guarantee we start from frame 0.
                    capture.release()
                    capture = cv2.VideoCapture(str(video_path))
                    if not capture.isOpened():
                        # Re-open should not fail since the original open
                        # succeeded for the same path, but guard anyway.
                        raise RuntimeError(
                            "Could not reopen video for sequential fallback: "
                            f"{video_path}"
                        )
                    _read_sequential(
                        capture,
                        n=n,
                        stride_ms=stride_ms,
                        output_dir=output_dir,
                        paths=written,
                        fps_hint=fps,
                    )
        else:
            logger.warning(
                "Video metadata unavailable (frame_count=%r, fps=%r); "
                "falling back to sequential read: %s",
                raw_frame_count,
                fps,
                video_path,
            )
            # Pass fps through if it's individually valid, even when
            # frame_count was the unusable half. Otherwise stride
            # sampling falls back to the 30fps default and mis-spaces
            # samples on (e.g.) a 60fps capture with bad frame_count.
            fallback_fps_hint = fps if math.isfinite(fps) and fps > 0 else None
            _read_sequential(
                capture,
                n=n,
                stride_ms=stride_ms,
                output_dir=output_dir,
                paths=written,
                fps_hint=fallback_fps_hint,
            )

        if not written:
            raise RuntimeError(f"No frames could be decoded from: {video_path}")

        logger.debug("Extracted %d frames from %s", len(written), video_path.name)
        return written
    except BaseException:
        for p in written:
            try:
                p.unlink()
            except OSError:
                pass
        if not output_dir_existed:
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise
    finally:
        capture.release()


def _seek_and_save(
    capture: "cv2.VideoCapture",
    timestamps_ms: list[float],
    output_dir: Path,
    paths: list[Path],
) -> bool:
    """Seek to each timestamp, decode, write a PNG.

    Skips iterations whose decoded frame is byte-identical to the
    previous successful write — sub-frame-period ``stride_ms`` (e.g.
    ``stride_ms=1`` on a 30fps clip) and large-GOP keyframe-snap both
    produce repeated reads of the same source frame, and writing them
    all would waste the downstream VLM frame budget on byte-identical
    images.

    Content-hash dedup is robust to backends with stuck or unreliable
    ``CAP_PROP_POS_FRAMES`` / ``CAP_PROP_POS_MSEC`` counters: a decode
    that produces visually-distinct content is accepted even if the
    position counters report no progress, and a decode that produces
    byte-identical content is correctly skipped regardless of what the
    counters say. The earlier counter-based dedup misclassified both
    cases on certain FFmpeg builds.

    Appends each successfully written file to ``paths`` as it lands so
    the caller's reference reflects partial state if a later iteration
    raises.

    Returns True if every requested timestamp produced a written frame
    (or was correctly dedup-skipped), False if the loop broke early
    because the backend rejected a ``set(CAP_PROP_POS_MSEC, ...)`` call
    or a seek-then-read returned no frame. The caller can use the
    False return to decide whether to retry via the sequential fallback
    even when ``paths`` is non-empty (e.g. when only the leading
    ``set(POS_MSEC, 0)`` succeeded and every later seek was rejected,
    the schedule is incomplete and a sequential retry can fill in the
    rest).
    """
    last_hash: bytes | None = None
    # Initialize to 0.0 (not -inf) so a backend that pins POS_MSEC
    # at 0 from the very first read also fails the strict-advance
    # check on iteration 1 — otherwise the first decode would accept
    # raw_ms=0 even when the requested t_ms was non-zero, collapsing
    # frame_000__t<requested>.png to frame_000__t0.png.
    last_accepted_ms: float = 0.0
    for t_ms in timestamps_ms:
        if not capture.set(cv2.CAP_PROP_POS_MSEC, t_ms):
            logger.info("Seek to %.1fms failed; stopping early", t_ms)
            return False
        ok, frame = capture.read()
        if not ok or frame is None:
            logger.info("No frame at %.1fms; stopping early", t_ms)
            return False
        # Hash the decoded frame's raw pixel buffer. blake2b on
        # tobytes() costs roughly one decode equivalent at 1080p
        # (modest overhead, dominated by the actual decode for typical
        # short clips); bounded ``n <= 64`` keeps the total well below
        # a second even at 4K.
        frame_hash = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
        if frame_hash == last_hash:
            logger.debug("Skipping byte-identical decode at %.1fms", t_ms)
            continue
        last_hash = frame_hash

        raw_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        # Use the backend's reported timestamp only when it actually
        # advanced past the previously accepted decode. A backend that
        # pins POS_MSEC at 0 reports a finite, non-negative value on
        # every read, so the naive `is finite and >= 0` check accepts
        # 0.0 and we'd name every distinct frame `__t0`. Falling back
        # to the requested t_ms preserves the documented contract that
        # filenames embed the source timestamp.
        ms_advanced = math.isfinite(raw_ms) and raw_ms > last_accepted_ms
        actual_ms = raw_ms if ms_advanced else t_ms
        if math.isfinite(raw_ms) and raw_ms >= 0:
            last_accepted_ms = max(last_accepted_ms, raw_ms)
        # Use len(paths) for the index so dedup-skipped iterations don't
        # leave gaps in the output sequence; matches _read_sequential.
        out = output_dir / f"frame_{len(paths):03d}__t{int(actual_ms)}.png"
        if not cv2.imwrite(str(out), frame):
            # imwrite may have created a partial file before failing; the
            # outer cleanup only walks `paths`, which doesn't yet include
            # this entry, so unlink it ourselves before raising.
            try:
                out.unlink()
            except OSError:
                pass
            raise RuntimeError(f"Failed to write frame to {out}")
        paths.append(out)
    return True


_SYNTH_FPS_FALLBACK = 30.0


def _read_sequential(
    capture: "cv2.VideoCapture",
    *,
    n: int,
    stride_ms: int | None,
    output_dir: Path,
    paths: list[Path],
    fps_hint: float | None,
) -> None:
    """Decode the stream sequentially, picking up to ``n`` frames.

    Used when container metadata is unreliable or when the seek path
    rejects every CAP_PROP_POS_MSEC seek. In stride mode, frames are
    accepted only when the clock has advanced past the next
    ``next_target_ms`` boundary; in even mode (no stride), the first
    ``n`` decodable frames are taken from the start of the stream.

    Some metadata-poor streams report a constant ``CAP_PROP_POS_MSEC``
    of 0 for every read. Stride filtering against a stuck PTS would
    lock after the first accepted frame and silently return only one
    PNG, so we synthesize a frame-index clock whenever the backend's
    reported PTS doesn't advance. ``fps_hint`` (the container's
    reported FPS, when known) drives the synth-clock spacing — using a
    hard-coded 30fps default when the source is e.g. a 60fps camera
    capture would mis-space stride samples by 2x. Falls back to 30fps
    when ``fps_hint`` is None or non-positive.

    Appends each written file to ``paths`` as it lands so the caller's
    reference reflects partial state when a later iteration raises.
    """
    # Stride targets are anchored to the requested schedule
    # ``[0, stride, 2*stride, ...]`` rather than to whatever the last
    # accepted frame's PTS happened to be — for variable-rate streams
    # an overshoot frame (e.g. 1000ms accepted for the 750ms boundary)
    # would otherwise shift every later target by the overshoot and
    # silently skip valid scheduled samples (1500ms in that case).
    next_boundary_idx = 0
    if fps_hint is not None and math.isfinite(fps_hint) and fps_hint > 0:
        synth_step_ms = 1000.0 / fps_hint
    else:
        synth_step_ms = 1000.0 / _SYNTH_FPS_FALLBACK
    decoded_idx = 0
    prev_real_ms = -1.0
    last_hash: bytes | None = None
    # Bound work proportional to ``n`` so a static-scene or
    # repeated-content fallback can't decode-until-EOF on a long clip
    # — content-hash dedup keeps writing 1 PNG while the loop hits
    # ``continue`` for every subsequent identical frame, and without
    # this floor we'd hash an entire hour of duplicate frames just
    # to satisfy a request for n=8. Per-call budget: 30 frames per
    # requested output (~1s of duplicate decodes per slot at 30fps),
    # with a floor for very small ``n``.
    consecutive_dedup_skips = 0
    dedup_skip_budget = max(60, n * 30)
    while len(paths) < n:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        raw_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if math.isfinite(raw_ms) and raw_ms > prev_real_ms:
            t_ms = raw_ms
            prev_real_ms = raw_ms
        else:
            # Backend isn't advancing PTS — synthesize a clock from the
            # decoded-frame index so stride filtering still makes progress.
            t_ms = decoded_idx * synth_step_ms
        decoded_idx += 1

        if stride_ms is not None:
            target_ms = next_boundary_idx * stride_ms
            if t_ms < target_ms:
                continue

        # Match _seek_and_save's content-hash dedup so a static-scene
        # source produces the same number of PNGs whether it goes
        # through the seek path or the sequential fallback. Without
        # this, a malformed-metadata mp4 of the same static video
        # would yield n byte-identical PNGs while the well-formed
        # version yields 1.
        frame_hash = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
        if frame_hash == last_hash:
            logger.debug("Skipping byte-identical sequential decode at %.1fms", t_ms)
            consecutive_dedup_skips += 1
            if consecutive_dedup_skips > dedup_skip_budget:
                logger.info(
                    "Stopping sequential fallback after %d consecutive "
                    "duplicate decodes (budget=%d)",
                    consecutive_dedup_skips,
                    dedup_skip_budget,
                )
                break
            continue
        last_hash = frame_hash
        consecutive_dedup_skips = 0

        out = output_dir / f"frame_{len(paths):03d}__t{int(t_ms)}.png"
        if not cv2.imwrite(str(out), frame):
            try:
                out.unlink()
            except OSError:
                pass
            raise RuntimeError(f"Failed to write frame to {out}")
        paths.append(out)
        if stride_ms is not None:
            # Advance to the smallest boundary strictly past the accepted
            # frame's t_ms so a future decode still has to clear the
            # next genuine ``k * stride_ms`` boundary.
            next_boundary_idx = int(t_ms // stride_ms) + 1


def _pick_timestamps(
    *, duration_ms: float, n: int, stride_ms: int | None
) -> list[float]:
    """Return millisecond offsets to sample, in chronological order.

    Even mode (``stride_ms is None``): returns ``n`` interior offsets at the
    midpoint of each of ``n`` equal segments.

    Stride mode: returns offsets ``0, stride_ms, 2*stride_ms, ...`` up to
    ``n`` entries, dropping any that fall at or beyond ``duration_ms``.
    """
    if stride_ms is not None:
        timestamps = [float(stride_ms * i) for i in range(n)]
        return [t for t in timestamps if t < duration_ms]

    if n <= 0:
        return []
    step = duration_ms / n
    return [step * (i + 0.5) for i in range(n)]
