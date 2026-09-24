"""Dynamic-sector set prediction for real ambiguous retrieval evidence."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_sectors import GrowingSectorController, SectorConfig


def _mlp(*sizes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def isotropic_set_nll(prediction, weight, target, target_mask, precision):
    """Proper isotropic-Gaussian mixture score per embedding dimension."""
    squared_distance = (
        target.unsqueeze(2) - prediction.unsqueeze(1)
    ).square().sum(-1)
    log_weight = weight.clamp(min=1e-9).log().unsqueeze(1)
    dimension = prediction.shape[-1]
    log_normalizer = 0.5 * dimension * (
        precision.clamp(min=1e-6).log() - torch.log(
            precision.new_tensor(2.0 * torch.pi)
        )
    )
    score = torch.logsumexp(
        log_weight + log_normalizer.view(-1, 1, 1)
        - 0.5 * precision.view(-1, 1, 1) * squared_distance,
        dim=-1,
    )
    return (-(score * target_mask).sum()
            / target_mask.sum().clamp(min=1) / dimension)


def component_recall_distance(prediction, weight, target, target_mask):
    similarity = torch.einsum("btd,bkd->btk", target, prediction)
    similarity = similarity.masked_fill(weight.unsqueeze(1) <= 0.05, -1.0)
    distance = (2.0 - 2.0 * similarity.max(-1).values).clamp(min=0.0).sqrt()
    return (distance * target_mask).sum() / target_mask.sum().clamp(min=1)


def interpretation_geometry_loss(sector_vector, chunk, chunk_mask,
                                 target, target_mask, temperature=0.1):
    """Supervised-contrastive geometry from weak chunk-to-target matching.

    A chunk's pseudo-label is its nearest valid human interpretation in frozen
    BGE space.  This teaches the vector geometry during training but is absent
    at inference; no target count or target embedding enters the controller.
    """
    losses = []
    raw_similarity = torch.einsum("bnd,btd->bnt", chunk, target)
    raw_similarity = raw_similarity.masked_fill(
        ~target_mask.bool().unsqueeze(1), -float("inf")
    )
    pseudo_label = raw_similarity.argmax(-1)
    for batch_index in range(len(chunk)):
        valid = chunk_mask[batch_index]
        z = sector_vector[batch_index, valid]
        label = pseudo_label[batch_index, valid]
        if len(z) < 2:
            continue
        logits = z @ z.T / temperature
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        positive = (label.unsqueeze(0) == label.unsqueeze(1)) & ~eye
        anchors = positive.any(-1)
        if not bool(anchors.any()):
            continue
        denominator = torch.logsumexp(logits.masked_fill(eye, -float("inf")), -1)
        numerator = torch.logsumexp(
            logits.masked_fill(~positive, -float("inf")), -1
        )
        losses.append((denominator[anchors] - numerator[anchors]).mean())
    return torch.stack(losses).mean() if losses else sector_vector.sum() * 0.0


class AmbigSectorNCA(nn.Module):
    """Set predictor whose sector count is state, never a fixed task label.

    The semantic projection is a deterministic Johnson--Lindenstrauss map of
    frozen BGE space.  The NCA may learn a bounded residual, but controller
    geometry is meaningful before training and cannot read target cardinality.
    """

    def __init__(self, embedding_dim: int, hidden: int = 128, sim_dim: int = 32,
                 k_max: int = 8, variant: str = "dynamic"):
        super().__init__()
        if variant not in {"dynamic", "fixed4", "no_sectors"}:
            raise ValueError(f"unknown variant: {variant}")
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.k_max = k_max
        self.variant = variant
        self.use_sectors = variant != "no_sectors"

        generator = torch.Generator().manual_seed(20260831)
        projection = torch.randn(embedding_dim, sim_dim, generator=generator)
        projection = torch.linalg.qr(projection, mode="reduced").Q
        self.register_buffer("semantic_projection", projection)

        self.input_proj = _mlp(2 * embedding_dim, hidden, hidden)
        self.sim_residual = nn.Linear(hidden, sim_dim, bias=False)
        nn.init.zeros_(self.sim_residual.weight)
        self.gru = nn.GRUCell(3 * hidden, hidden)
        self.sector_score = _mlp(hidden + 1, hidden, 1)
        self.output_residual = _mlp(hidden, hidden, embedding_dim)
        nn.init.zeros_(self.output_residual[-1].weight)
        nn.init.zeros_(self.output_residual[-1].bias)
        self.root_credit = _mlp(hidden, hidden, 1)

        config = SectorConfig(
            dim=sim_dim,
            k_max=k_max,
            initial_sectors=4 if variant == "fixed4" else 1,
            dynamic_ops=variant == "dynamic",
            density_birth=True,
            distance_birth=False,
            max_births_per_step=k_max,
        )
        self.controller = GrowingSectorController(config)

    def _root_claim(self, root_content, root_index, mask):
        """Credit each root once from its pre-message content representation.

        Reading credit from the recurrent state lets repeated delivery perturb
        the evidence claim through extra computation.  Root-wise means of the
        initial content are invariant to exact repetition while still allowing
        different articles to earn different bounded credit.
        """
        claims = []
        for batch_index in range(len(root_content)):
            valid_roots = torch.unique(root_index[batch_index][mask[batch_index]])
            summaries = []
            for root in valid_roots:
                members = mask[batch_index] & (root_index[batch_index] == root)
                summaries.append(root_content[batch_index, members].mean(0))
            root_state = torch.stack(summaries)
            claims.append(torch.sigmoid(self.root_credit(root_state)).squeeze(-1))
        max_roots = max(len(row) for row in claims)
        padded = root_content.new_zeros(len(claims), max_roots)
        root_mask = torch.zeros(
            len(claims), max_roots, dtype=torch.bool, device=root_content.device
        )
        ratios = []
        for index, row in enumerate(claims):
            padded[index, :len(row)] = row
            root_mask[index, :len(row)] = True
            ratios.append(row.mean())
        return torch.stack(ratios), padded, root_mask

    def forward(self, chunk, question, root_index, mask, steps: int = 4):
        batch, cells, _ = chunk.shape
        question_cells = question.unsqueeze(1).expand(-1, cells, -1)
        local = self.input_proj(torch.cat([chunk, question_cells], dim=-1))
        state = local
        controller_states = None

        for _ in range(steps):
            # Fresh tensors are required at each recurrent step: autograd saves
            # q for the state update, so mutating one shared buffer later would
            # invalidate the saved forward value.
            q_full = chunk.new_zeros(batch, cells, self.k_max)
            active_full = torch.zeros(
                batch, self.k_max, dtype=torch.bool, device=chunk.device
            )
            semantic = F.normalize(chunk @ self.semantic_projection, dim=-1)
            residual = torch.tanh(self.sim_residual(state))
            sector_vector = F.normalize(semantic + residual, dim=-1)
            if self.use_sectors:
                if controller_states is None:
                    controller_states = [
                        (self.controller.init_grouped_state(
                            sector_vector[b, mask[b]].detach(),
                            root_index[b, mask[b]],
                        ) if self.variant == "dynamic" else
                         self.controller.init_state(sector_vector[b, mask[b]].detach()))
                        for b in range(batch)
                    ]
                for b in range(batch):
                    _, controller_states[b] = self.controller.step(
                        sector_vector[b, mask[b]].detach(), controller_states[b]
                    )
                    # detach() alone still aliases the mutable controller
                    # state; clone the snapshot saved by autograd for matmul.
                    prototype = controller_states[b].prototypes.detach().clone()
                    active = controller_states[b].active.clone()
                    similarity = sector_vector[b, mask[b]] @ prototype.T
                    q = (similarity / self.controller.cfg.temperature).masked_fill(
                        ~active.unsqueeze(0), -float("inf")
                    ).softmax(-1)
                    q_full[b, mask[b]] = q
                    active_full[b] = controller_states[b].active
            else:
                q_full[..., 0] = mask.to(chunk.dtype)
                active_full[:, 0] = True

            denominator = q_full.sum(1).unsqueeze(-1).clamp(min=1e-6)
            summary = torch.einsum("bnk,bnh->bkh", q_full, state) / denominator
            support = q_full.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            score = self.sector_score(
                torch.cat([summary, support.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
            score = score.masked_fill(~active_full, -float("inf"))
            alpha = score.softmax(-1)
            local_context = torch.einsum("bnk,bkh->bnh", q_full, summary)
            global_context = torch.einsum("bk,bkh->bh", alpha, summary)
            update = torch.cat([
                local,
                local_context,
                global_context.unsqueeze(1).expand(-1, cells, -1),
            ], dim=-1)
            state = self.gru(
                update.reshape(batch * cells, -1),
                state.reshape(batch * cells, -1),
            ).view(batch, cells, self.hidden)
            state = state * mask.unsqueeze(-1)

        denominator = q_full.sum(1).unsqueeze(-1).clamp(min=1e-6)
        semantic_summary = torch.einsum("bnk,bnd->bkd", q_full, chunk) / denominator
        hidden_summary = torch.einsum("bnk,bnh->bkh", q_full, state) / denominator
        prediction = F.normalize(
            semantic_summary + self.output_residual(hidden_summary), dim=-1
        )
        claim_ratio, root_credit, root_mask = self._root_claim(local, root_index, mask)
        # Confidence is monotone in credited roots but capped regardless of
        # chunk multiplicity.  Repeated chunks can refine prediction, not credit.
        concentration = 2.0 + 14.0 * claim_ratio
        return {
            "prediction": prediction,
            "weight": alpha,
            "active": active_full,
            "q": q_full,
            "claim_ratio": claim_ratio,
            "root_credit": root_credit,
            "root_mask": root_mask,
            "concentration": concentration,
            "allocated_sectors": active_full.sum(-1).to(chunk.dtype),
            "sector_vector": sector_vector,
        }
