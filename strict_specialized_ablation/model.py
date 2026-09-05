import copy

import torch

from isolated_ablation.model import IsolatedAblationModel


class StrictOriginalAblationModel(IsolatedAblationModel):
    """Single-component ablation for the canonical asymmetric experts."""

    def __init__(self, args):
        strict_args = copy.copy(args)
        self.strict_ablation_view = getattr(
            args, "strict_ablation_view", "original"
        )
        if self.strict_ablation_view not in {"original", "validity_aware"}:
            raise ValueError(
                f"Unsupported strict ablation view: {self.strict_ablation_view}"
            )
        # The paper model is asymmetric: the original expert keeps the gate,
        # while the validity-aware expert uses fixed equal fusion.
        strict_args.force_disable_branch_gate = (
            self.strict_ablation_view == "validity_aware"
        )
        super().__init__(strict_args)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        hidden_states = self.add_position_embedding(input_ids)
        if self.strict_ablation_view == "validity_aware":
            valid_mask = input_ids.ne(0).unsqueeze(-1).to(hidden_states.dtype)
        else:
            # Ones preserve original padded embeddings before spectral transforms.
            valid_mask = torch.ones(
                (*input_ids.shape, 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        hidden_states = self.item_encoder(
            hidden_states,
            self.get_attention_mask(input_ids),
            valid_mask,
        )
        return [hidden_states] if all_sequence_output else hidden_states
