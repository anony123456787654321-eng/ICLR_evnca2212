"""Message delivery schedules and the node-gossip runner.

Everything the paper claims about order-invariance is tested here: the same
example is run under synchronous delivery, 0.5-probability asynchronous firing,
dropped messages, random delays, reordering, and duplicate delivery.  A method
whose fixed point depends on the schedule will show it as variance across
``jitter`` seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class DeliverySchedule:
    fire_prob: float = 1.0        # 1.0 = synchronous
    drop_prob: float = 0.0
    max_delay: int = 0            # extra steps a message may sit in flight
    repeat_prob: float = 0.0      # probability a message is delivered twice

    @staticmethod
    def named(name: str) -> "DeliverySchedule":
        return {
            "sync": DeliverySchedule(1.0, 0.0, 0, 0.0),
            "async": DeliverySchedule(0.5, 0.0, 0, 0.0),
            "lossy": DeliverySchedule(0.5, 0.2, 0, 0.0),
            "delayed": DeliverySchedule(0.5, 0.0, 3, 0.0),
            "repeating": DeliverySchedule(0.5, 0.0, 1, 0.3),
            "adversarial": DeliverySchedule(0.5, 0.2, 3, 0.3),
        }[name]


def run_node_gossip(
    states: List,
    adj: List[List[int]],
    steps: int,
    injections: Dict[int, list],
    rng: np.random.Generator,
    schedule: Optional[DeliverySchedule] = None,
    probe: Optional[Callable[[int, List], None]] = None,
) -> List:
    """Run merge-based gossip; ``states`` are mutated and returned.

    Messages carry a *snapshot* of the sender's state at send time, so delays
    and repeats really do deliver stale and duplicated evidence.
    """
    schedule = schedule or DeliverySchedule()
    n = len(states)
    inflight: Dict[int, List] = {}

    for t in range(steps):
        for atom in injections.get(t, []):
            states[atom.cell].insert(atom.root_id, atom.payload, atom.mass,
                                     getattr(atom, 'refinement', 0.0))
        for dst, msg in inflight.pop(t, []):
            states[dst] = states[dst].merge(msg)
        for i in range(n):
            if schedule.fire_prob < 1.0 and rng.random() >= schedule.fire_prob:
                continue
            snap = states[i].copy()
            for j in adj[i]:
                if schedule.drop_prob and rng.random() < schedule.drop_prob:
                    continue
                delay = int(rng.integers(0, schedule.max_delay + 1)) if schedule.max_delay else 0
                inflight.setdefault(t + 1 + delay, []).append((j, snap))
                if schedule.repeat_prob and rng.random() < schedule.repeat_prob:
                    extra = int(rng.integers(1, schedule.max_delay + 2))
                    inflight.setdefault(t + 1 + delay + extra, []).append((j, snap))
        if probe is not None:
            probe(t, states)
    return states
