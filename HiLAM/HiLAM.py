from __future__ import annotations

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList, RMSNorm

# attention

from x_transformers import Decoder, CrossAttender
from x_transformers.x_transformers import Attention, FeedForward

# h-net from Sukjun Hwang et al. https://arxiv.org/abs/2507.07955

from h_net_dynamic_chunking import HNet

# einstein equations

import einx
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch_einops_utils import lens_to_mask, pack_with_inverse, exclusive_cumsum

from discrete_continuous_embed_readout import EmbedAndReadout
from vector_quantize_pytorch import VectorQuantize

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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

# video to patches

def VideoToPatchEmbed(dim, patch_size, channels = 3):
    patch_dim = channels * patch_size ** 2

    return nn.Sequential(
        Rearrange('b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
        nn.Linear(patch_dim, dim)
    )

# factorized space time attention

class FactorizedSpaceTimeAttention(Module):
    def __init__(
        self,
        dim,
        depth = 2,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        dropout = 0.
    ):
        super().__init__()

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                RMSNorm(dim),
                Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout),
                RMSNorm(dim),
                Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, causal = True),
                RMSNorm(dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(
        self,
        x   # Float['batch time space dim']
    ):
        batch, time, space, _ = x.shape

        for spatial_norm, spatial_attn, temporal_norm, temporal_attn, ff_norm, ff in self.layers:
            x = rearrange(x, 'b t s d -> (b t) s d')
            x = spatial_attn(spatial_norm(x)) + x
            x = rearrange(x, '(b t) s d -> (b s) t d', b = batch, t = time)
            x = temporal_attn(temporal_norm(x)) + x
            x = ff(ff_norm(x)) + x
            x = rearrange(x, '(b s) t d -> b t s d', b = batch, s = space)

        return self.norm(x)

# policy network

class PolicyNetwork(Module):
    def __init__(
        self,
        dim,
        *,
        transformer: Decoder | dict,
        state_to_embed: Module,
        actions_num_discrete = 0,
        actions_num_continuous = 0,
        higher_actions_num_discrete = 0,
        higher_actions_num_continuous = 0,
        accept_text_embed = False,
        dim_text_embed = None
    ):
        super().__init__()

        # states

        self.state_to_embed = state_to_embed

        # transformer

        if isinstance(transformer, dict):
            text_cond_kwargs = dict(cross_attend = True, attn_dim_context = default(dim_text_embed, dim)) if accept_text_embed else dict()

            transformer = Decoder(**{**transformer, **text_cond_kwargs})

        self.transformer = transformer

        # actions

        assert (
            (actions_num_discrete > 0 or actions_num_continuous > 0) and
            (higher_actions_num_discrete > 0 or higher_actions_num_continuous > 0)
        )

        self.low_action_embed, self.low_action_readout = EmbedAndReadout(dim = dim, num_discrete = actions_num_discrete, num_continuous = actions_num_continuous, regression_loss_type = 'l1')

        self.higher_action_embed, self.higher_action_readout = EmbedAndReadout(dim = dim, num_discrete = higher_actions_num_discrete, num_continuous = higher_actions_num_continuous, regression_loss_type = 'l1')

        # is high level embed - give the network a hint which mode it is in

        self.start_action_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.is_high_level_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        # whether to accept text embedding, cross attended to

        self.accept_text_embed = accept_text_embed

    def forward(
        self,
        states,
        actions = None,
        high_actions = None,
        return_pred_only = False,
        text_embed = None,
        text_mask = None
    ):
        batch, seq_len = states.shape[:2]

        is_low_level_policy = exists(high_actions) and exists(actions)
        is_high_level_policy = exists(high_actions) and not exists(actions)

        assert is_low_level_policy or is_high_level_policy

        # video to embed - even pixel space is fine, as recent papers show

        state_embed = self.state_to_embed(states)

        # take care of the two scenarios
        # think it can be done with the same policy, no reason not to

        embed = state_embed

        # let the network know if it is acting as low or high level policy

        if is_high_level_policy:
            embed = embed + self.is_high_level_embed

        # if training the low level policy, we now condition on the high level actions

        high_action_embed = self.higher_action_embed(high_actions)

        if is_low_level_policy:
            embed = einx.add('b t ... d, b t d', embed, high_action_embed)

        # take care of past actions

        start_action_embed = repeat(self.start_action_embed, 'd -> b 1 d', b = batch)

        if is_low_level_policy:
            low_action_embed = self.low_action_embed(actions)
            past_low_action_embed = cat((start_action_embed, low_action_embed), dim = 1)

            past_action_embed = past_low_action_embed[:, :seq_len]

        elif is_high_level_policy:
            past_high_action_embed = cat((start_action_embed, high_action_embed), dim = 1)

            past_action_embed = past_high_action_embed[:, :seq_len]

        embed = einx.add('b t s d, b t d', embed, past_action_embed)

        # maybe pack spatial

        embed, inverse_pack_time = pack_with_inverse(embed, 'b * d') # assume time is always closest to batch (b t s d) if spatial dimension exists

        # maybe cross attention to language commands

        assert not (exists(text_embed) and not self.accept_text_embed), '`accept_text_embed` should be set to True for PolicyNetwork if sending text conditioning'

        transformer_forward_kwargs = dict()

        if exists(text_embed):
            transformer_forward_kwargs = dict(
                context = text_embed,
                context_mask = text_mask
            )

        # attention

        attended = self.transformer(embed, **transformer_forward_kwargs)

        # maybe pool over spatial

        attended = inverse_pack_time(attended)

        attended = reduce(attended, 'b t ... d -> b t d', 'mean')

        # objectives are different depending on high or low level

        if return_pred_only:
            targets = None
        elif is_low_level_policy:
            targets = actions[:, 1:]
        elif is_high_level_policy:
            targets = high_actions[:, 1:]

        if not return_pred_only:
            attended = attended[:, :-1]

        # maybe loss or prediction

        if is_high_level_policy:
            readout = self.higher_action_readout
        elif is_low_level_policy:
            readout = self.low_action_readout

        return readout(attended, targets = targets)

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
        video_image_size: int | None = None,
        video_patch_size: int | None = None,
        video_channels = 3,
        state_action_attend = True,
        state_action_space_time_attend_kwargs: dict = dict(),
        decoder_kwargs: dict = dict(),
        h_net_kwargs: dict = dict(),
        vq_kwargs: dict = dict(),
        accept_text_embed = False,
        dim_text_embed = None
    ):
        super().__init__()

        # to action embeds

        self.inverse_dynamics_model = inverse_dynamics_model

        assert actions_num_discrete > 0 or actions_num_continuous > 0

        self.action_embed, self.action_readout = EmbedAndReadout(dim = dim, num_discrete = actions_num_discrete, num_continuous = actions_num_continuous, regression_loss_type = 'l1')

        # factorized space time attention on video patches + action tokens

        self.has_state_action_space_time_attend = state_action_attend

        self.accept_text_embed = accept_text_embed

        if self.accept_text_embed:
            self.text_cross_attender = CrossAttender(
                dim = dim,
                depth = 2,
                attn_dim_context = default(dim_text_embed, dim)
            )

        if self.has_state_action_space_time_attend:
            assert exists(video_image_size) and exists(video_patch_size), 'both video_image_size and video_patch_size must be given to use state action space time attention'
            assert divisible_by(video_image_size, video_patch_size), 'image size must be divisible by patch size'

            self.video_to_patch_embed = VideoToPatchEmbed(dim, video_patch_size, channels = video_channels)

            self.state_action_space_time_attend = FactorizedSpaceTimeAttention(
                dim = dim,
                **state_action_space_time_attend_kwargs
            )

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
        return_loss_breakdown = False,
        text_embed = None,
        text_mask = None
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

        # maybe cross attend to text before space time attend

        if exists(text_embed):
            assert self.accept_text_embed, '`accept_text_embed` must be set to True to accept text conditioning'
            action_embed, inverse_pack_time = pack_with_inverse(action_embed, 'b * d')
            action_embed = self.text_cross_attender(action_embed, context = text_embed, context_mask = text_mask)
            action_embed = inverse_pack_time(action_embed)

        # maybe factorized space time attention on video patches + action tokens

        if self.has_state_action_space_time_attend:
            space_tokens = self.video_to_patch_embed(states)

            # pack space and action tokens - space on the left, actions on the right

            packed, inverse_pack_space = pack_with_inverse([space_tokens, action_embed], 'b t * d')

            # factorized space time attend

            packed = self.state_action_space_time_attend(packed)

            # splice off action tokens (rightmost position)

            _, action_embed = inverse_pack_space(packed)

        # attention, with temporal compression with h-net

        hnet_return = self.action_chunker(action_embed, return_intermediates = True)
        attended, ratio_loss, intermediates = hnet_return.output, hnet_return.loss, hnet_return.intermediates

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
