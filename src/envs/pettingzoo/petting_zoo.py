# from pettingzoo.atari import space_invaders_v2
from pettingzoo.mpe import simple_spread_v3
from envs.multiagentenv import MultiAgentEnv
import numpy as np

PETTINGZOO_ENVS = {
    # "SpaceInvaders": space_invaders_v2.parallel_env, # todo: not test yet
    "SimpleSpread": simple_spread_v3.parallel_env,
}

class PettingZoo(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        self.env_name = kwargs["map_name"]
        self.episode_limit = kwargs["episode_limit"]
        self.seed = kwargs["seed"]
        # inner environment
        self.env = PETTINGZOO_ENVS[self.env_name](render_mode="human")
        self.env.reset()

        # static variables
        self.n_agents = len(self.env.agents)
        self.n_actions = self.env.action_space(self.env.agents[0]).n

        self.reset()

    def reset(self):
        self.env.reset(self.seed)

        # dynamic variables
        self.ep_steps = 0
        self.reward = 0
        self.total_reward = 0
        # we also have these two because parallel_env has no .last() method
        self.observations = self.no_obs()
        self.terminated = False
        self.truncated = False
        return self.get_obs(), self.get_state()

    def step(self, actions):
        actions = {agent: action for agent, action in zip(self.env.agents, actions)}
        observations, reward, terminated, truncations, info = self.env.step(actions)

        self.ep_steps += 1
        self.reward = float(sum(reward.values()) / self.n_agents)
        self.total_reward += float(sum(reward.values()) / self.n_agents)
        self.observations = observations
        self.terminated = all(terminated.values())
        self.truncated = any(truncations.values()) or self.ep_steps >= self.episode_limit

        return self.reward, self.terminated or self.truncated, {}

    def no_obs(self):
        return {agent: np.zeros(self.get_obs_size(), dtype=np.float32) for agent in self.env.agents} # maybe problematic, because its still in the observation space

    def get_obs(self):
        if len(list(self.observations.values())) == 0:
            self.observations = self.no_obs()

        return list(self.observations.values())

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.env.observation_space(self.env.agents[0]).shape[0]

    def get_state(self):
        return self.env.state()

    def get_state_size(self):
        return len(self.env.state())

    def get_avail_actions(self):
        return [np.ones(self.n_actions) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        # all actions are available
        return np.ones(self.n_actions)

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "reward": float(self.reward),
            "total_reward": float(self.total_reward),
            "episode_length": self.ep_steps,
        }
        return stats