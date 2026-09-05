import copy

import torch
import torch.nn as nn

from model._abstract_model import SequentialRecModel
from model._modules import FeedForward, LayerNorm, MultiHeadAttention


class IsolatedAblationLayer(nn.Module):
    """DuoSpectra masked encoder with exactly one component disabled."""

    def __init__(self, args):
        super().__init__()
        self.mode = args.isolated_ablation_mode
        self.use_attention = self.mode != "without_temporal_branch"
        self.use_spectral = self.mode != "without_frequency_branch"
        self.use_fft = (
            self.use_spectral and self.mode != "without_global_spectral_branch"
        )
        self.use_wavelet = (
            self.use_spectral and self.mode != "without_local_wavelet_branch"
        )
        self.use_band_gate = self.mode != "without_adaptive_band_fusion"
        self.use_branch_gate = (
            self.mode != "without_time_frequency_gate"
            and not getattr(args, "force_disable_branch_gate", False)
        )

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.seq_len = args.max_seq_length
        self.freq_bins = self.seq_len // 2 + 1
        self.cutoff = max(1, min(args.freq_cutoff, self.freq_bins))
        self.alpha = args.spectral_alpha

        if self.use_attention:
            self.attention_layer = MultiHeadAttention(args)
        if self.use_fft:
            self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
            self.base_bias = nn.Parameter(torch.zeros(self.num_heads, self.freq_bins, 1))
            self.high_pass_scale = nn.Parameter(
                torch.full((1, self.num_heads, 1, self.head_dim), args.high_pass_init)
            )
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_heads * self.freq_bins * 2),
            )
            if self.use_band_gate:
                self.band_gate = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, self.num_heads * 3),
                )
        if self.use_wavelet:
            self.wavelet_weight = nn.Parameter(
                torch.randn(1, self.num_heads, self.seq_len // 2, self.head_dim)
                * args.wavelet_init
            )
        if self.use_attention and self.use_spectral and self.use_branch_gate:
            self.branch_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def _split_heads(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape
        return hidden_states.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, hidden_states):
        batch, _, seq_len, _ = hidden_states.shape
        return hidden_states.permute(0, 2, 1, 3).contiguous().view(
            batch, seq_len, self.hidden_size
        )

    def _wavelet_filter(self, spectral_heads):
        batch, heads, seq_len, dim = spectral_heads.shape
        even_len = seq_len if seq_len % 2 == 0 else seq_len - 1
        x_trunc = spectral_heads[:, :, :even_len, :]
        x_even = x_trunc[:, :, 0::2, :]
        x_odd = x_trunc[:, :, 1::2, :]
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)
        detail = detail * self.wavelet_weight[:, :, : detail.size(2), :]

        output = torch.zeros(
            (batch, heads, even_len, dim),
            device=spectral_heads.device,
            dtype=spectral_heads.dtype,
        )
        output[:, :, 0::2, :] = approx + detail
        output[:, :, 1::2, :] = approx - detail
        if even_len < seq_len:
            output = torch.cat([output, spectral_heads[:, :, even_len:, :]], dim=2)
        return output

    def _adaptive_band_filter(
        self, spectral_heads, residual_heads, input_tensor, valid_mask
    ):
        freq = torch.fft.rfft(spectral_heads, dim=2, norm="ortho")
        low_freq = freq.clone()
        low_freq[:, :, self.cutoff :, :] = 0
        low_pass = torch.fft.irfft(
            low_freq, dim=2, n=self.seq_len, norm="ortho"
        )
        high_pass = spectral_heads - low_pass

        context = (input_tensor * valid_mask).sum(dim=1) / float(self.seq_len)
        adaptive = self.adaptive_mlp(context).view(
            -1, self.num_heads, self.freq_bins, 2
        )
        adaptive_scale = adaptive[..., 0:1]
        adaptive_bias = adaptive[..., 1:2]
        global_filter = (
            freq * (self.base_filter * (1.0 + adaptive_scale))
            + self.base_bias
            + adaptive_bias
        )
        global_out = torch.fft.irfft(
            global_filter, dim=2, n=self.seq_len, norm="ortho"
        )

        low_branch = self.alpha * low_pass + (1.0 - self.alpha) * global_out
        high_branch = self.high_pass_scale.square() * high_pass
        branches = torch.stack([low_branch, high_branch, residual_heads], dim=2)
        if not self.use_band_gate:
            return branches.mean(dim=2)

        gate = self.band_gate(context).view(-1, self.num_heads, 3, 1, 1)
        return (torch.softmax(gate, dim=2) * branches).sum(dim=2)

    def forward(self, input_tensor, attention_mask, valid_mask):
        attention_output = (
            self.attention_layer(input_tensor, attention_mask)
            if self.use_attention
            else None
        )
        if not self.use_spectral:
            return attention_output
        residual_heads = self._split_heads(input_tensor)
        spectral_heads = residual_heads * valid_mask.unsqueeze(1)
        spectral_parts = []
        if self.use_fft:
            spectral_parts.append(
                (
                    self.alpha,
                    self._adaptive_band_filter(
                        spectral_heads, residual_heads, input_tensor, valid_mask
                    ),
                )
            )
        if self.use_wavelet:
            spectral_parts.append(
                (1.0 - self.alpha, self._wavelet_filter(spectral_heads))
            )

        if len(spectral_parts) == 1:
            combined_heads = spectral_parts[0][1]
        else:
            combined_heads = sum(weight * value for weight, value in spectral_parts)
        spectral_output = self._merge_heads(combined_heads)
        spectral_output = self.layer_norm(
            self.out_dropout(spectral_output) + input_tensor
        )

        if attention_output is None:
            return spectral_output
        if not self.use_branch_gate:
            return 0.5 * spectral_output + 0.5 * attention_output
        gate = self.branch_gate(
            torch.cat([spectral_output, attention_output], dim=-1)
        )
        return gate * spectral_output + (1.0 - gate) * attention_output


class IsolatedAblationBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = IsolatedAblationLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, valid_mask):
        return self.feed_forward(
            self.layer(hidden_states, attention_mask, valid_mask)
        )


class IsolatedAblationEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        block = IsolatedAblationBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, valid_mask):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, valid_mask)
        return hidden_states


class IsolatedAblationModel(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = IsolatedAblationEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        hidden_states = self.add_position_embedding(input_ids)
        valid_mask = input_ids.ne(0).unsqueeze(-1).to(dtype=hidden_states.dtype)
        hidden_states = self.item_encoder(
            hidden_states, self.get_attention_mask(input_ids), valid_mask
        )
        return [hidden_states] if all_sequence_output else hidden_states

    def calculate_loss(
        self, input_ids, answers, neg_answers, same_target, user_ids
    ):
        sequence_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(
            sequence_output, self.item_embeddings.weight.transpose(0, 1)
        )
        return nn.CrossEntropyLoss()(logits, answers)
