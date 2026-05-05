import pytest
param = pytest.mark.parametrize

import torch
from torch.nn import Module

@param('discrete_high_level_actions', (False, True))
def test_hi_lam(discrete_high_level_actions):
    from HiLAM.HiLAM import HierarchicalLatentActionModel

    class MockIDM(Module):
        def __init__(self, dim_action = 20):
            super().__init__()
            self.dim_action = dim_action

        def forward(self, states):
            batch, time, device = *states.shape[:2], states.device
            return torch.randn((batch, time, self.dim_action))

    hi_lam = HierarchicalLatentActionModel(
        dim = 512,
        h_net_head_depth = 2,
        h_net_trunk_depth = 2,
        h_net_tail_depth = 2,
        actions_num_continuous = 20,
        num_high_level_discrete = 1024 if discrete_high_level_actions else None,
        inverse_dynamics_model = MockIDM(20)
    )

    states = torch.randn(2, 10, 3, 64, 64) # video 64x64

    loss = hi_lam(states)
    loss.backward()

    # after much training, you have access to the actions as well as the learnt higher level actions and their lengths

    lower_actions, higher_actions, higher_action_lens = hi_lam(states, return_actions_only = True)

    if discrete_high_level_actions:
        assert higher_actions.dtype == torch.long
    else:
        assert higher_actions.dtype == torch.float

    # take care of batch repeat interleave, as their successful strategy was to simply condition their policy with each hierarchical action at each timestep

    _, interleaved_higher_actions, _ = hi_lam(states, return_actions_only = True, return_batch_repeat_interleaved = True)

    total_lens = higher_action_lens.clamp(min = 0).sum(dim = -1)

    assert (interleaved_higher_actions.shape[1] == total_lens).all()

@param('accept_text', (False, True))
def test_policy_e2e(accept_text):

    from HiLAM.HiLAM import (
        HierarchicalLatentActionModel,
        PolicyNetwork,
        VideoToPatchTokens
    )
    from x_transformers import Decoder

    dim = 256
    num_discrete_actions = 10
    num_high_level_discrete = 16
    patch_size = 8
    img_size = 32

    # patchify video frames

    state_to_embed = VideoToPatchTokens(
        dim = dim,
        patch_size = patch_size,
        channels = 3
    )

    # hi-lam for discovering hierarchical latent actions

    hi_lam = HierarchicalLatentActionModel(
        dim = dim,
        h_net_head_depth = 1,
        h_net_trunk_depth = 1,
        h_net_tail_depth = 1,
        actions_num_discrete = num_discrete_actions,
        num_high_level_discrete = num_high_level_discrete,
    )

    # policy that acts at both levels

    policy = PolicyNetwork(
        dim = dim,
        transformer = dict(dim = dim, depth = 2, heads = 4),
        state_to_embed = state_to_embed,
        actions_num_discrete = num_discrete_actions,
        higher_actions_num_discrete = num_high_level_discrete,
        accept_text_embed = accept_text
    )

    # mock data

    states = torch.randn(2, 16, 3, img_size, img_size)
    actions = torch.randint(0, num_discrete_actions, (2, 16))

    # 1. train hi-lam, obtain interleaved higher level actions

    _, interleaved_higher_actions, _ = hi_lam(
        states,
        actions = actions,
        return_actions_only = True,
        return_batch_repeat_interleaved = True,
    )

    # text

    text_kwargs = dict()

    if accept_text:
        text_embed = torch.randn(2, 8, dim)
        text_mask = torch.randint(0, 2, (2, 8)).bool()

        text_kwargs = dict(text_embed = text_embed, text_mask = text_mask)

    # 2. train high level policy - predict higher actions from states

    high_loss = policy(states, high_actions = interleaved_higher_actions, **text_kwargs)
    high_loss.backward()

    # 3. train low level policy - predict actions conditioned on higher actions

    low_loss = policy(states, actions = actions, high_actions = interleaved_higher_actions, **text_kwargs)
    low_loss.backward()

    # 4. inference - first predict high level, then low level

    policy.eval()

    with torch.no_grad():

        # predict high level actions from states alone

        high_logits = policy(states, high_actions = interleaved_higher_actions, return_pred_only = True, **text_kwargs)

        # predict low level actions conditioned on predicted high level

        low_logits = policy(states, actions = actions, high_actions = interleaved_higher_actions, return_pred_only = True, **text_kwargs)

    assert high_logits.shape[:2] == (2, 16)
    assert low_logits.shape[:2] == (2, 16)
