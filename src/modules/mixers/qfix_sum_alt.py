from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .qfix_weight import QFix_FF_Weight, QFix_SI_Weight, gt_constraint


class QFixSumAlt(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = np.prod(args.state_shape).item()
        self.joint_action_dim = args.n_agents * args.n_actions

        self.w_module = (
            QFix_SI_Weight(args, single_output=False)
            if args.w_attention
            else QFix_FF_Weight(args, single_output=False)
        )
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
        actions = actions.reshape(-1, self.joint_action_dim)

        w = self.w_module(states, actions)
        w = gt_constraint(w, self.args.qfix_w_gt)
        b = self.b_module(states)

        outputs: torch.Tensor
        if self.args.qfix_type == "qfix":
            outputs = (w * individual_advantages).sum(dim=-1, keepdim=True) + b
        elif self.args.qfix_type == "q+fix":
            if self.args.qfix_detach_advantages:
                individual_advantages = individual_advantages.detach()
            outputs = (
                individual_qvalues.sum(dim=-1, keepdim=True)
                + (w * individual_advantages).sum(dim=-1, keepdim=True)
                + b
            )
        else:
            raise NotImplementedError

        # restore batch size
        return outputs.view(batch_size, -1, 1)
