import heapq
import random

import matplotlib.pyplot as plt
from collections import deque

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from environments.env_SACG_LMAPF import SimEnvLMAPF
from algs.alg_gen_cor_v1 import copy_nodes
from algs.alg_clean_corridor import *
from create_animation import do_the_animation
from algs.params import *
from algs.alg_temporal_a_star import calc_temporal_a_star, calc_fastest_escape, create_constraints


class AlgAgentLaCAM:

    def __init__(self, num: int, start_node: Node, next_goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.path: List[Node] = [start_node]
        self.goals_per_iter_list: List[Node] = [next_goal_node]
        self.arrived: bool = False

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def a_last_path_node_name(self):
        return self.path[-1].xy_name

    @property
    def a_curr_node_name(self):
        return self.curr_node.xy_name

    @property
    def a_prev_node_name(self):
        return self.prev_node.xy_name

    @property
    def a_next_node_name(self):
        return self.next_node.xy_name

    @property
    def a_next_goal_node_name(self):
        return self.next_goal_node.xy_name

    def __eq__(self, other):
        return self.num == other.num


class ALgLaCAM:
    name = 'LaCAM'

    def __init__(self, env: SimEnvLMAPF, to_check_paths: bool = False, to_assert: bool = False, **kwargs):

        self.env = env
        assert self.env.is_sacg
        self.to_check_paths: bool = to_check_paths
        self.to_assert: bool = to_assert
        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np: np.ndarray = self.env.img_np
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func
        self.h_dict = self.env.h_dict

        self.agents: List[AlgAgentLaCAM] = []
        self.agents_dict: Dict[str, AlgAgentLaCAM] = {}
        self.start_nodes: List[Node] = []
        self.next_iteration: int = 0
        self.logs: dict = {}

    def initiate_problem(self, obs: dict) -> bool:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._create_agents(obs)

        self.logs = {
            'runtime': 0,
            'expanded_nodes': 0,
        }
        start_time = time.time()
        succeeded = self._solve()

        runtime = time.time() - start_time
        self.logs['runtime'] = runtime

        return succeeded

    def get_actions(self, obs: dict) -> Dict[str, str]:
        # updates
        self.next_iteration = obs['iteration']
        self._update_agents(obs)
        actions = {
            agent.name: agent.path[self.next_iteration].xy_name for agent in self.agents
        }
        return actions

    def _create_agents(self, obs: dict) -> None:
        self.agents: List[AlgAgentLaCAM] = []
        self.agents_dict: Dict[str, AlgAgentLaCAM] = {}
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            if num == 0:
                next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            else:
                next_goal_node = self.nodes_dict[obs_agent.curr_node_name]
            new_agent = AlgAgentLaCAM(num=num, start_node=start_node, next_goal_node=next_goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _update_agents(self, obs: dict) -> None:
        for agent in self.agents:
            obs_agent = obs[agent.name]
            agent.prev_node = agent.curr_node
            agent.curr_node = self.nodes_dict[obs_agent.curr_node_name]
            if agent.num == 0:
                agent.next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            else:
                agent.next_goal_node = self.nodes_dict[obs_agent.curr_node_name]
            agent.arrived = obs_agent.arrived
            agent.goals_per_iter_list.append(agent.next_goal_node)

    def _solve(self) -> bool:
        return False


@use_profiler(save_dir='../stats/alg_LaCAM.pstat')
def main():
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_SACG()
    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    # for rendering
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = ALgLaCAM(env=env, to_check_paths=to_check_paths)
    solved = alg.initiate_problem(obs=obs)

    if solved:

        for i_step in range(max_time):
            actions = alg.get_actions(obs)  # alg part
            obs, metrics, terminated, info = env.step(actions)

            # update metrics + render
            if to_render:
                i_agent = alg.agents_dict['agent_0']
                plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
                             'n_agents': env.n_agents, 'agents': alg.agents, 'i_agent': i_agent,
                             'corridor': i_agent.path[i_step:]}
                plot_step_in_env(ax[0], plot_info)
                plt.pause(plot_rate)

            if terminated:
                break
    else:
        print('\nUNSOLVED!')

    # if to_render:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_info = {'i': max_time, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                 'n_agents': env.n_agents, 'agents': env.agents}
    plot_step_in_env(ax[0], plot_info)
    plt.show()
    if env.is_sacg:
        max_time = len(alg.agents_dict['agent_0'].path)
    do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
                           'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')


if __name__ == '__main__':
    main()
