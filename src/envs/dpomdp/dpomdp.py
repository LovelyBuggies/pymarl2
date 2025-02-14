import os
from itertools import product
import numpy as np
from gym import spaces, Env
from rl_parsers.dpomdp import parse  # pip install git+https://github.com/abaisero/rl-parsers
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th

class DPOMDP(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # inner env
        self.env = DPOMDPEnv(problem=kwargs["map_name"], horizon=kwargs["episode_limit"], seed=kwargs["seed"])

        # static variables
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions
        self.episode_limit = kwargs["episode_limit"]

        # dynamic variables
        self.ep_steps = None # does not determine termination, just for calculating mean reward
        self.reward = None
        self.total_reward = None

        # initialization for pymarl2 running requirements
        self.reset()

    def reset(self):
        self.env.reset()
        self.ep_steps = 0
        self.reward = 0
        self.total_reward = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        _, reward, terminated = self.env.step(actions) # _ is a non-onehot observation, useless here=
        self.ep_steps += 1
        self.reward = reward
        self.total_reward += reward
        info = {}
        return reward, terminated, info

    def get_obs(self):
        # !!! onehot jt observation with onehot_obs is poorly implemented in the original code
        return self.env.onehot_joint_obs(self.env.observations)

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return len(self.get_state())

    def get_avail_actions(self):
        return [np.ones(self.n_actions) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        # all actions are available
        return np.ones(self.n_actions)

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "reward": self.reward,
            "total_reward": self.total_reward,
        }
        return stats


# from https://github.com/lyu-xg/MAPI/blob/main/src/environments/decpomdp.py
# input file path changed

DPOMDP_ENVS = {
    "grid_small": "GridSmall.dpomdp",
    "grid": "grid3x3corners.dpomdp",
    "recycling": "recycling.dpomdp",
    "mars": "Mars.dpomdp",
    "boxpushing": "boxpushing.dpomdp",
    "firefighting": "fireFighting.dpomdp",
    "firefighting_4house": "fireFighting_2_4_3.dpomdp",
    "wireless": "wirelessWithOverhead.dpomdp",
    "long_fire_fight": "longFireFight.dpomdp",
    "dtiger": "dectiger_original.dpomdp",
    "broadcast_channel": "broadcast_channel.dpomdp",
    "cooperative_box_pushing": "cooperative_box_pushing.dpomdp",
}


class DPOMDPEnv(object):
    def __init__(self, problem, horizon=2, seed=1998):
        np.random.seed(seed)
        self.horizon = horizon
        if DPOMDP_ENVS.get(problem):
            filename = os.path.join(os.path.dirname(__file__), DPOMDP_ENVS.get(problem))
        else:
            raise FileNotFoundError(problem + "environment not found")
        if not filename:
            raise FileNotFoundError(problem + "file not found")
        with open(filename, encoding="utf-8") as f:
            self.d = parse(f.read())

        self.n_agents = len(self.d.agents)

        self.n_states = len(self.d.states)
        self.states = tuple(range(self.n_states))

        self.n_obs = len(self.d.observations[0])
        self.observations = tuple(range(self.n_obs))
        assert all(self.d.observations[0] == O for O in self.d.observations)

        self.n_actions = len(self.d.actions[0])
        self.actions = tuple(range(self.n_actions))
        assert all(self.d.actions[0] == A for A in self.d.actions)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)
        self.state_space = spaces.Discrete(self.n_states)

        self.joint_observations = tuple(
            product(self.observations, repeat=self.n_agents)
        )
        self.joint_actions = tuple(product(self.actions, repeat=self.n_agents))

        self.state = None
        self.i_step = 0

        self.one_hot_obs = False

    @property
    def no_obs(self):
        if self.one_hot_obs:
            return [np.zeros(self.n_obs) for _ in self.d.agents]
        else:
            return [[-1] for _ in self.d.agents]

    def reset(self):
        self.i_step = 1
        self.reset_state()
        return self.no_obs

    def reset_state(self):
        self.state = np.random.choice(self.states, p=self.d.start)

    def step(self, action):
        self.i_step += 1
        action = tuple(action)
        probs = [self.d.T[(*action, self.state, sp)] for sp in self.states]
        new_state = np.random.choice(self.states, p=probs)
        obs = self._emit(action, new_state)
        reward = float(self.d.R[(*action, self.state, new_state, *obs)])
        is_done = bool(self.d.reset[(*action, new_state)])

        if is_done:
            self.reset_state()
        else:
            self.state = new_state  # Assign the new_state to the 'state' attribute

        return (
            self.onehot_joint_obs(obs) if self.one_hot_obs else obs,
            reward,
            self.i_step > self.horizon,
            # we do not use `is_done or self.i_step > self.horizon` because the horizon continues after reset
        )

    def get_state(self):
        return self.onehot_state(self.state)

    def _emit(self, actions, new_state):
        probs = [self.d.O[(*actions, new_state, *o)] for o in self.joint_observations]
        i = np.random.choice(len(self.joint_observations), p=probs)
        return self.joint_observations[i]

    def onehot_state(self, s):
        res = np.zeros(self.n_states)
        res[s] = 1
        return res

    def onehot_joint_obs(self, O):
        return [self.onehot_obs(o) for o in O]

    def onehot_obs(self, o):
        res = np.zeros(self.n_obs)
        res[o] = 1
        return res