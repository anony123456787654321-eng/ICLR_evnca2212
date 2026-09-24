"""Memory ablations that change ONLY what they claim to change.

The first attempt set `enable_long_term=False`. In the pinned XMem that flag
does far more than disable long-term memory: `max_work_elements` is only
computed under it (memory_manager.py:160), and the bounding/compression block
is guarded by it (memory_manager.py:182). Turning it off therefore produced
UNBOUNDED working memory rather than a controlled removal.

Measured on a 52-frame DAVIS clip at frame 51:

    normal        work_mem  9,720 (compressed) + long_mem 128
    old ablation  work_mem 17,820 (still growing) + long_mem 0

So the "no long-term" arm was a different, larger computation -- and the
growth is the likely cause of the failure on MOSE event 9e85179b.

The corrected ablations keep `enable_long_term=True`, so bounding and
compression behave exactly as in `normal`, and intervene at the READOUT.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# Cache-key version. The corrected controls are a DIFFERENT computation from
# the old ones, so results produced under v1 must not be reused.
ABLATION_VERSION = 2


@dataclass
class ReadStats:
    """What the readout actually consumed, per call."""

    calls: int = 0
    calls_with_long_term_available: int = 0
    long_term_elements_excluded: int = 0     # max observed
    work_mem_elements: int = 0               # max observed
    work_mem_elements_final: int = 0
    long_mem_elements_final: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class LongTermReadAblation:
    """Exclude long-term keys/values from the prediction readout.

    Bounded memory management is PRESERVED: working memory is still capped at
    `max_mid_term_frames`, compression still runs, and features still migrate
    into the long-term store. The store is simply not consulted when
    predicting, so the model answers from recent memory alone while the
    memory system behaves identically to `normal`.

    This is the controlled removal the experiment intended. It is not the same
    as `enable_long_term=False`, which also removes the bound.
    """

    name = "no_long_term_read"
    semantics = (
        "Long-term keys and values are excluded from the readout. Working "
        "memory remains bounded and compressed exactly as in `normal`, and "
        "the long-term store is still written -- it is simply not read. "
        "Compare against `enable_long_term=False`, which additionally removes "
        "the working-memory bound and is therefore a different computation."
    )

    def __init__(self):
        self.stats = ReadStats()
        self._original = None
        self._memory = None

    def install(self, processor):
        mem = processor.memory
        self._memory = mem
        original = mem.match_memory
        self._original = original
        stats = self.stats

        def readout_without_long_term(query_key, selection):
            lm = getattr(mem, "long_mem", None)
            engaged = bool(lm is not None and mem.enable_long_term
                           and lm.engaged())
            stats.calls += 1
            if engaged:
                stats.calls_with_long_term_available += 1
                stats.long_term_elements_excluded = max(
                    stats.long_term_elements_excluded, int(lm.size))
            stats.work_mem_elements = max(stats.work_mem_elements,
                                          int(mem.work_mem.size))
            if not engaged:
                return original(query_key, selection)

            # Temporarily hide the long-term store for the duration of THIS
            # read. Management is untouched: the store keeps its contents and
            # is restored immediately afterwards.
            saved = mem.long_mem
            empty = type(saved)(count_usage=mem.enable_long_term_usage)
            mem.long_mem = empty
            try:
                return original(query_key, selection)
            finally:
                mem.long_mem = saved

        mem.match_memory = readout_without_long_term
        return self

    def finish(self):
        """Remove the hook and drop every reference we introduced.

        Exception-safe: a failure reading the final sizes must still leave the
        object clean, because a surviving hook would leak the processor into
        the next event. The closure captures `original` as a local rather than
        reaching through `self`, so removing the instance override breaks the
        only cycle this class creates.
        """
        mem, self._memory = self._memory, None
        original, self._original = self._original, None
        if mem is None:
            return self.stats
        try:
            if original is not None:
                # Delete the instance override rather than reassigning, so the
                # object is left exactly as found.
                try:
                    del mem.__dict__["match_memory"]
                except (KeyError, AttributeError):
                    try:
                        mem.match_memory = original
                    except Exception:
                        pass
            try:
                self.stats.work_mem_elements_final = int(mem.work_mem.size)
                lm = getattr(mem, "long_mem", None)
                self.stats.long_mem_elements_final = (
                    int(lm.size) if lm is not None else 0)
            except Exception:
                # Sizes are diagnostics; failing to read them must not leave
                # the hook installed.
                pass
        finally:
            del mem, original
        return self.stats

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return None


class RecentOnlyAblation(LongTermReadAblation):
    """Retain only the most recent frames, with the bound actually enforced.

    Two things the earlier version got wrong:

      * its `max_mid_term_frames` limit was never applied, because that
        setting is read only under `enable_long_term`;
      * it had no policy for the first annotated frame.

    Here `enable_long_term` stays on so the bound is enforced through XMem's
    own compression path, the long-term store is excluded from the readout as
    above, and the PROMPT frame is explicitly retained: dropping the only
    frame carrying ground truth would change the task rather than the memory,
    and the semi-supervised setting guarantees it.
    """

    name = "recent_only"
    semantics = (
        "Working memory is bounded to the configured recent window and the "
        "long-term store is excluded from the readout. The first annotated "
        "frame is RETAINED: it carries the only ground truth, and dropping it "
        "would change the task rather than the memory."
    )
    keeps_prompt_frame = True


def configure_control(xmem_config: dict, mode: str) -> dict:
    """Return the XMem configuration for one control.

    Every control keeps `enable_long_term=True` so that bounding and
    compression are identical to `normal`; the difference is made at the
    readout instead.
    """
    cfg = dict(xmem_config)
    applied = {"mode": mode, "ablation_version": ABLATION_VERSION}
    if mode == "normal":
        applied["semantics"] = "published configuration, unmodified"
    elif mode == "no_long_term_read":
        applied["semantics"] = LongTermReadAblation.semantics
    elif mode == "recent_only":
        # A genuinely short recent window, enforced through the SAME
        # compression path `normal` uses.
        cfg["max_mid_term_frames"] = 3
        cfg["min_mid_term_frames"] = 2
        applied["semantics"] = RecentOnlyAblation.semantics
        applied["max_mid_term_frames"] = 3
    elif mode == "disrupted_content":
        applied["semantics"] = (
            "memory content disruption; reported as not applicable where the "
            "pinned revision offers no hook")
    else:
        raise ValueError(f"unknown control {mode!r}")
    cfg["enable_long_term"] = True          # bounding must stay on
    applied["enable_long_term"] = True
    applied["bounded_management_preserved"] = True
    return cfg, applied


CONTROLS = ("normal", "no_long_term_read", "recent_only", "disrupted_content")


# The name callers use. `configure` is kept as an alias so any older caller
# keeps working, but `configure_control` is what the experiment imports.
configure = configure_control
