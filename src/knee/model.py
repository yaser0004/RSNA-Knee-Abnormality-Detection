import timm
import torch
import torch.nn as nn


class KneeModel(nn.Module):
    """Phase 1 baseline: single 2D backbone applied per slice, mean-pooled
    across slices, 12 sigmoid heads (BCE loss applied outside the model on
    the returned logits). Input: [batch, n_slices, 3, H, W]."""

    def __init__(self, backbone_name: str = "efficientnet_b0", num_labels: int = 12,
                 pretrained: bool = True, head: str = "mean", **backbone_kwargs):
        super().__init__()
        if head not in ("mean", "slot_attention"):
            raise ValueError(f"unknown head {head!r}")
        # kwargs go straight to timm: a ViT needs img_size to build its position
        # embedding for the resolution it will actually see, and only timm knows
        # what a given architecture accepts
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, **backbone_kwargs)
        self.head_kind = head
        self.num_labels = num_labels
        n_features = self.backbone.num_features
        self.head = nn.Linear(n_features, num_labels)
        if head == "slot_attention":
            # one attention score per (slot, diagnosis), and a per-diagnosis
            # readout so each label reads its own pooled feature rather than a
            # shared one
            self.slot_scores = nn.Linear(n_features, num_labels)
            self.label_readout = nn.Parameter(torch.empty(num_labels, n_features))
            nn.init.normal_(self.label_readout, std=n_features ** -0.5)
            self.label_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is either [batch, n_images, 3, H, W] (the Phase 2 flat volume) or
        [batch, n_slots, n_groups, 3, H, W] (PreppedSlotDataset). The slot and
        group axes are folded here rather than in the loader so `mask` stays
        addressable per slot.

        `mask` is [batch, n_slots] of bool. An absent slot arrives as zeros, and
        an unmasked mean would average real anatomy with a black square -- the
        more slots a study is missing, the harder its features get pulled toward
        whatever the backbone emits for black. Since slot fill runs from 100%
        (AX_FLUID) down to 19% (AX_STRUCT), that bias would track missingness,
        which is not a property of the knee."""
        if self.head_kind == "slot_attention":
            slot_features = self._slot_features(x)
            return self._attend(slot_features, mask)[0]

        if x.dim() == 6:
            batch, n_slots, n_groups = x.shape[:3]
            x = x.reshape(batch, n_slots * n_groups, *x.shape[3:])
        else:
            n_groups = 1

        batch, n_images, channels, height, width = x.shape
        features = self.backbone(x.reshape(batch * n_images, channels, height, width))
        features = features.view(batch, n_images, -1)

        if mask is None:
            return self.head(features.mean(dim=1))

        weights = mask.to(features.dtype).repeat_interleave(n_groups, dim=1).unsqueeze(-1)
        # a study with no present slot would divide by zero; it is already
        # destined for build_submission's 0.5 fallback, but the model must not
        # be what raises first
        denominator = weights.sum(dim=1).clamp(min=1.0)
        return self.head((features * weights).sum(dim=1) / denominator)

    def _slot_features(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, n_slots, n_groups, 3, H, W] -> one feature per slot, averaged
        over that slot's groups. Groups within a slot are the same acquisition at
        different depths, so averaging them is a within-slot summary; the axis
        worth attending over is the slot, which is the plane and weighting."""
        if x.dim() != 6:
            raise ValueError(
                "the slot_attention head needs [batch, n_slots, n_groups, 3, H, W]; "
                f"got {tuple(x.shape)}. The flat volume has no slot axis to attend over."
            )
        batch, n_slots, n_groups, channels, height, width = x.shape
        flat = x.reshape(batch * n_slots * n_groups, channels, height, width)
        features = self.backbone(flat).view(batch, n_slots, n_groups, -1)
        return features.mean(dim=2)

    def _attend(self, slot_features: torch.Tensor,
                mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-diagnosis masked attention over slots. Returns (logits, weights)."""
        scores = self.slot_scores(slot_features)               # [B, S, L]
        if mask is not None:
            present = mask.unsqueeze(-1)
            # a study with no present slot has nothing to normalise over and
            # softmax would return NaN; fall back to attending uniformly, whose
            # features are zeros anyway, so only the bias survives
            any_present = mask.any(dim=1).view(-1, 1, 1)
            scores = scores.masked_fill(~present & any_present, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * (present | ~any_present).to(weights.dtype)

        pooled = torch.einsum("bsf,bsl->blf", slot_features, weights)
        logits = torch.einsum("blf,lf->bl", pooled, self.label_readout) + self.label_bias
        return logits, weights

    def attention_weights(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """[batch, n_slots, num_labels] -- which plane each diagnosis reads.
        Exposed because it is the claim c2 makes: a mean-pool head cannot say
        "MCL lives in the coronal plane", and this one should be checkable
        against the anatomy rather than trusted."""
        return self._attend(self._slot_features(x), mask)[1]
