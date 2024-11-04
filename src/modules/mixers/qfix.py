from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .qfix_si_weight import QFix_SI_Weight
from .qmix import QMixer
from .vdn import VDNMixer


def make_inner_mixer(inner_args: SimpleNamespace, args: SimpleNamespace) -> nn.Module:
    """Create the inner mixer module, which satisfies IGM but is not IGM-complete."""
    if inner_args.mixer == "vdn":
        return VDNMixer()

    if inner_args.mixer == "qmix":
        inner_args.n_agents = args.n_agents
        inner_args.n_actions = args.n_actions
        inner_args.state_shape = args.state_shape
        return QMixer(inner_args)

    raise ValueError(f'invalid inner mixer type "{inner_args.mixer}"')


class QFix(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * args.n_actions

        self.w_module = QFix_SI_Weight(args, single_output=True)
        self.b_module = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(args.hypernet_embed, 1),
        )

        inner_args = SimpleNamespace(**args.inner_mixer)
        self.inner_mixer = make_inner_mixer(inner_args, args)

    def forward(
        self,
        individual_qvalues: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        individual_vvalues: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies QFIX mixing.

        :param individual_qvalues: FloatTensor, shape = (B, T, N)
        :param states: FloatTensor, shape = (B, T, S)
        :param actions: FloatTensor, shape = (B, T, N, A), onehot encoding
        :param individual_vvalues: FloatTensor, shape = (B, T, N)
        :return: FloatTensor, shape = (B, T, 1)
        """

        # store batch size
        batch_size = individual_qvalues.size(0)

        inner_qvalues = self.inner_mixer(individual_qvalues, states)
        inner_vvalues = self.inner_mixer(individual_vvalues, states)
        inner_advantages = inner_qvalues - inner_vvalues
        inner_advantages = inner_advantages.view(-1, 1)

        # flatten batch and time dimensions
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        w = self.w_module(states, actions)
        b = self.b_module(states)

        outputs = w * inner_advantages + b

        # restore batch dimension
        return outputs.view(batch_size, -1, 1)
