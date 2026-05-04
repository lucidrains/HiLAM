from __future__ import annotations

import torch
from torch.nn import Module, RMSNorm

from x_transformers import Decoder      # attention
from h_net_dynamic_chunking import HNet # h-net from Sukjun Hwang et al. https://arxiv.org/abs/2507.07955

from discrete_continuous_embed_readout import EmbedAndReadout

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        h_net_target_avg_action_length = 4.,  # action length next level up
        ratio_loss_weight = 3e-2,
        decoder_kwargs: dict = dict(),
        hnet_kwargs: dict = dict(),
        inverse_dynamics_model: Module | None = None,
    ):
        super().__init__()

        # to action embeds

        self.inverse_dynamics_model = inverse_dynamics_model

        assert actions_num_discrete > 0 or actions_num_continuous > 0

        self.action_embed, self.action_readout = EmbedAndReadout(dim = dim, num_discrete = actions_num_discrete, num_continuous = actions_num_continuous)

        # define the 3 transformers, head, trunk (working on the compressed skill vectors), tail

        self.action_chunker = HNet(
            Decoder(dim = dim, depth = h_net_head_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            Decoder(dim = dim, depth = h_net_trunk_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            Decoder(dim = dim, depth = h_net_tail_depth, pre_norm_has_final_norm = False, **decoder_kwargs),
            dim = dim,
        )

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        states,
        actions = None,
        return_actions_only = False,
        return_loss_breakdown = False
    ):

        # if actions not given and idm given at init, derive actions from state

        maybe_idm = self.inverse_dynamics_model

        assert exists(actions) or exists(maybe_idm), 'actions must be given if inverse dynamics model not supplied at init'

        if not exists(actions):
            actions = maybe_idm(states)

        # encode actions with embed-readout lib

        action_embed = self.action_embed(actions)

        # attention, with temporal compression with h-net

        attended, ratio_loss, intermediates = self.action_chunker(action_embed, return_intermediates = True)

        # maybe early return

        if return_actions_only:

            higher_actions = intermediates.input_downsampled_tokens
            higher_action_lens = intermediates.chunk_lens

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
