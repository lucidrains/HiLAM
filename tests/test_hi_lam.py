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
