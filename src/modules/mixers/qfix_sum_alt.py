from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .qfix_si_weight import QFix_SI_Weight


class QFixSumAlt(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * args.n_actions

        self.w_module = QFix_SI_Weight(args, single_output=False)
        self.b_module = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(args.hypernet_embed, 1),
        )

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

        individual_advantages = individual_qvalues - individual_vvalues
        individual_advantages = individual_advantages.view(-1, self.n_agents)

        # flatten batch and time dimensions
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        w = self.w_module(states, actions)
        b = self.b_module(states)

        outputs = (w * individual_advantages).sum(dim=-1, keepdim=True) + b

        # restore batch size
        return outputs.view(batch_size, -1, 1)
