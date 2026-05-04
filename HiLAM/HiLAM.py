from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, RMSNorm

from x_transformers import Decoder      # attention
from h_net_dynamic_chunking import HNet # h-net from Sukjun Hwang et al. https://arxiv.org/abs/2507.07955

from discrete_continuous_embed_readout import EmbedAndReadout

from vector_quantize_pytorch import VectorQuantize

import einx
from einops import rearrange, repeat
from torch_einops_utils import lens_to_mask, pack_with_inverse, exclusive_cumsum

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor related

def batch_repeat_interleave(
    feats,  # float['b n ...'] | int['b n']
    lens,   # int['b n']
):
    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device = device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add('w, b n -> b n w', arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.clamp(min = 0).sum(dim = -1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter

    seq_arange = torch.arange(seq, device = device)
    seq_arange = repeat(seq_arange, 'n -> b (n w)', b = batch, w = window_size)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    feats, unpack_one = pack_with_inverse(feats, 'b n *')
    output_indices = repeat(output_indices, 'b m -> b m d', d = feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    output = einx.where(
        'b n, b n ..., -> b n ...',
        output_mask, output, 0
    )

    return output

# classes

class HierarchicalLatentActionModel(Module):
    def __init__(
        self,
        dim,
        *,
        h_net_head_depth,
        h_net_trunk_depth,
        h_net_tail_depth,
        actions_num_discrete = 0,
        actions_num_continuous = 0,
        num_high_level_discrete = None,
        h_net_target_avg_action_length = 4.,  # action length next level up
        h_net_ratio_loss_weight = 3e-2,
        inverse_dynamics_model: Module | None = None,
        decoder_kwargs: dict = dict(),
        h_net_kwargs: dict = dict(),
        vq_kwargs: dict = dict(),
    ):
        super().__init__()

        # to action embeds

        self.inverse_dynamics_model = inverse_dynamics_model

        assert actions_num_discrete > 0 or actions_num_continuous > 0

        self.action_embed, self.action_readout = EmbedAndReadout(dim = dim, num_discrete = actions_num_discrete, num_continuous = actions_num_continuous)

        # define the 3 transformers, head, trunk (working on the compressed skill vectors), tail

        maybe_vq = VectorQuantize(dim = dim, codebook_size = num_high_level_discrete, **vq_kwargs) if exists(num_high_level_discrete) else None

        self.discrete_high_level_actions = exists(num_high_level_discrete)
        self.num_high_level_discrete = num_high_level_discrete

        self.action_chunker = HNet(
            Decoder(dim = dim, depth = h_net_head_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            Decoder(dim = dim, depth = h_net_trunk_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            Decoder(dim = dim, depth = h_net_tail_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            dim = dim,
            vq = maybe_vq,
            target_avg_token_length = h_net_target_avg_action_length,
            ratio_loss_weight = h_net_ratio_loss_weight,
            **h_net_kwargs
        )

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        states,
        actions = None,
        return_actions_only = False,
        return_batch_repeat_interleaved = False,
        return_loss_breakdown = False
    ):

        # if actions not given and idm given at init, derive actions from state

        maybe_idm = self.inverse_dynamics_model

        assert exists(actions) or exists(maybe_idm), 'actions must be given if inverse dynamics model not supplied at init'

        if not exists(actions):

            with torch.no_grad():
                maybe_idm.eval()
                actions = maybe_idm(states)

        # encode actions with embed-readout lib

        action_embed = self.action_embed(actions)

        # attention, with temporal compression with h-net

        attended, ratio_loss, intermediates = self.action_chunker(action_embed, return_intermediates = True)

        # maybe early return

        if return_actions_only:

            if self.discrete_high_level_actions:
                higher_actions = intermediates.quantized_downsampled_indices
            else:
                higher_actions = intermediates.input_downsampled_tokens

            higher_action_lens = intermediates.chunk_lens

            if return_batch_repeat_interleaved:
                higher_actions = batch_repeat_interleave(higher_actions, higher_action_lens)

            return actions, higher_actions, higher_action_lens

        # reconstruction loss with behavior clone / autoregressive on actions (todo - states should be optionally included)

        attended = self.final_norm(attended)

        action_recon_loss = self.action_readout(attended[:, :-1], targets = actions[:, 1:], return_loss = True)

        # losses

        loss_breakdown = (action_recon_loss, ratio_loss)

        total_loss = action_recon_loss + ratio_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, loss_breakdown
