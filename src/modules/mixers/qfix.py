import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .vdn import VDNMixer
from .qmix import QMixer

class QfixMixer(nn.Module):
    def __init__(self, scheme, args):
        super(QfixMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.sub_mixer = args.sub_mixer
        self.fix_alt = args.fix_alt
        self.obs_obs = args.obs_obs
        self.obs_state = args.obs_state
        if self.sub_mixer == "vdn":
            self.mixer = VDNMixer(args)
        elif self.sub_mixer == "qmix":
            self.mixer = QMixer(args)
            if self.fix_alt:
                raise "Cannot alt-fix Qfix-mono"
        else:
            raise f"Qfix-{self.mixer_name} not implemented"

        if self.fix_alt:
            fixer_input_shape = self._get_input_shape(scheme, obs_obs=self.obs_obs, obs_last_action=True, obs_state=self.obs_state)
            self.fixer = QfixNRNN(fixer_input_shape, args)
        else:
            fixer_input_shape = self._get_input_shape(scheme, obs_obs=self.obs_obs, obs_last_action=True, obs_state=self.obs_state)
            self.fixer = QfixRNN(fixer_input_shape * self.n_agents, args)
        
        biaser_input_shape = self._get_input_shape(scheme, obs_obs=self.obs_obs, obs_state=self.obs_state)
        self.biaser = QfixRNN(biaser_input_shape * self.n_agents, args)

    def forward(self, agent_qs, actions, batch):
        if self.fix_alt:
            return self._fix_alt_forward(agent_qs, actions, batch)
        else:
            return self._fix_forward(agent_qs, actions, batch)

    def _fix_forward(self, agent_qs, actions, batch):
        batch_states = batch["state"]
        batch_size, max_t_filled, n_agents, n_actions = agent_qs.size()
        agent_vs, agent_advs = self._q_to_v_adv(agent_qs)
        agent_advs_action = th.gather(agent_advs, 3, index=actions)
        fixer_output = th.zeros(batch_size, max_t_filled, 1).to(self.args.device)
        fixer_hidden_states = self.fixer.init_hidden().expand(batch_size, -1)
        biaser_output = th.zeros(batch_size, max_t_filled, 1).to(self.args.device)
        biaser_hidden_states = self.biaser.init_hidden().expand(batch_size, -1)
        for t in range(max_t_filled):
            fixer_inputs = self._build_inputs(batch, t, obs_obs=self.obs_obs, obs_last_action=True, obs_state=self.obs_state)
            fixer_rnn_output, fixer_hidden_states = self.fixer(fixer_inputs, fixer_hidden_states)
            fixer_output[:, t, :] = th.abs(fixer_rnn_output)
            biaser_inputs = self._build_inputs(batch, t, obs_obs=self.obs_obs, obs_state=self.obs_state)
            biaser_rnn_output, biaser_hidden_states = self.biaser(biaser_inputs, biaser_hidden_states)
            biaser_output[:, t, :] = biaser_rnn_output

        mix_advs = self.mixer(agent_advs_action.squeeze(3), batch_states)
        fix_advs = F.elu(mix_advs.view(-1, 1) * fixer_output.view(-1, 1)).view(batch_size, max_t_filled, 1)

        return fix_advs + biaser_output

    def _fix_alt_forward(self, agent_qs, actions, batch):
        batch_states = batch["state"]
        batch_size, max_t_filled, n_agents, n_actions = agent_qs.size()
        agent_vs, agent_advs = self._q_to_v_adv(agent_qs)
        agent_advs_action = th.gather(agent_advs, 3, index=actions)
        fixer_output = th.zeros(batch_size, max_t_filled, n_agents, 1).to(self.args.device)
        fixer_hidden_states = self.fixer.init_hidden().expand(batch_size, n_agents, -1)
        biaser_output = th.zeros(batch_size, max_t_filled, 1).to(self.args.device)
        biaser_hidden_states = self.biaser.init_hidden().expand(batch_size, -1)
        for t in range(max_t_filled):
            fixer_inputs = self._build_inputs(batch, t, obs_obs=self.obs_obs, obs_last_action=True, obs_state=self.obs_state)
            fixer_rnn_output, fixer_hidden_states = self.fixer(fixer_inputs, fixer_hidden_states)
            fixer_output[:, t, :, :] = th.abs(fixer_rnn_output)
            biaser_inputs = self._build_inputs(batch, t, obs_obs=self.obs_obs, obs_state=self.obs_state)
            biaser_rnn_output, biaser_hidden_states = self.biaser(biaser_inputs, biaser_hidden_states)
            biaser_output[:, t, :] = biaser_rnn_output

        fix_advs = self.mixer((agent_advs_action * fixer_output).squeeze(3), batch_states)
        return fix_advs + biaser_output

    def _q_to_v_adv(self, agent_qs): # advantage calculation
        agent_vs = agent_qs.max(dim=3, keepdim=True)[0]
        agent_advs = agent_qs - agent_vs.expand(agent_qs.shape)
        return agent_vs, agent_advs

    def _get_input_shape(self, scheme, obs_obs=True, obs_last_action=False, obs_agent_id=False, obs_state=False):
        if obs_obs:
            input_shape = scheme["obs"]["vshape"]
        if obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if obs_agent_id:
            input_shape += self.n_agents
        if obs_state:
            input_shape += scheme["state"]["vshape"]
        if input_shape == 0:
            raise ValueError("Cannot forward empty input")

        return input_shape

    def _build_inputs(self, batch, t, obs_obs=True, obs_last_action=False, obs_agent_id=False, obs_state=False):
        bs = batch.batch_size
        inputs = []
        if obs_obs:
            inputs.append(batch["obs"][:, t])
        if obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if obs_state:
            inputs.append(batch["state"][:, t].unsqueeze(1).repeat(1, self.n_agents, 1))
        if len(inputs) == 0:
            raise ValueError("Cannot forward empty input")

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

class QfixRNN(nn.Module):
    def __init__(self, input_shape, args):
        super(QfixRNN, self).__init__()
        self.args = args
        self.hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        x = F.relu(self.fc1(inputs.view(b, a * e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(b, self.hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h

class QfixNRNN(nn.Module):
    def __init__(self, input_shape, args):
        super(QfixNRNN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)