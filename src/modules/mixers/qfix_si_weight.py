import itertools as itt
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positive(x: torch.Tensor) -> torch.Tensor:
    return x.abs() + 1e-10


def make_feedforward_module(sizes: list[int]) -> nn.Module:
    """create a feedforward neural network module interleaved with ReLU activations"""

    if len(sizes) < 2:
        raise ValueError(f"module requires at least 2 sizes, given {len(sizes)}")

    if len(sizes) == 2:
        # just return Linear
        in_features, out_features = sizes
        return nn.Linear(in_features, out_features)

    # construct interleaved sequential module
    layers: list[nn.Module] = []
    for i, (in_features, out_features) in enumerate(itt.pairwise(sizes)):
        if i > 0:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, out_features))

    return nn.Sequential(*layers)


class QFix_SI_Weight(nn.Module):
    def __init__(self, args: SimpleNamespace, *, single_output: bool):
        super().__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.output_dim = 1 if single_output else self.n_agents

        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed
        adv_hypernet_layers = getattr(args, "adv_hypernet_layers", 1)
        adv_hypernet_sizes = [adv_hypernet_embed] * (adv_hypernet_layers - 1)
        key_extractor_sizes = [self.state_dim] + adv_hypernet_sizes + [1]
        agents_extractor_sizes = (
            [self.state_dim] + adv_hypernet_sizes + [self.output_dim]
        )
        action_extractor_sizes = (
            [self.state_action_dim] + adv_hypernet_sizes + [self.output_dim]
        )

        for _ in range(self.num_kernel):
            key_extractor = make_feedforward_module(key_extractor_sizes)
            self.key_extractors.append(key_extractor)

            agents_extractor = make_feedforward_module(agents_extractor_sizes)
            self.agents_extractors.append(agents_extractor)

            action_extractor = make_feedforward_module(action_extractor_sizes)
            self.action_extractors.append(action_extractor)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = torch.cat([states, actions], dim=1)

        head_attend_weights = []
        for key_extractor, agents_extractor, action_extractor in zip(
            self.key_extractors,
            self.agents_extractors,
            self.action_extractors,
        ):
            x_key = positive(key_extractor(states)).repeat(1, self.output_dim)
            x_agents = F.sigmoid(agents_extractor(states))
            x_action = F.sigmoid(action_extractor(data))
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.output_dim)
        head_attend = head_attend.sum(dim=1)

        return head_attend
