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
from algs.alg_temporal_a_star import calc_temporal_a_star


class AlgCBSAgent:

    def __init__(self, num: int, start_node: Node, next_goal_node: Node):
        self.num = num
        self.name = f'agent_{self.num}'
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.path: List[Node] = [start_node]
        self.goals_per_iter_list: List[Node] = [next_goal_node]
        self.arrived: bool = False

    def __eq__(self, other):
        return self.num == other.num

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class CBSNode:
    def __init__(self, agents: List[AlgCBSAgent], nodes, nodes_dict, h_dict, parent=None):
        self.agents = agents
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.h_dict = h_dict
        self.vertex_conf_list: List[Tuple[int, int, int]] = []
        self.edge_conf_list: List[Tuple[int, int, int, int, int]] = []
        self.solution = {}
        self.cost = 1e10
        self.parent = parent
        if self.parent is not None:
            self.vertex_conf_list = self.parent.vertex_conf_list[:]
            self.edge_conf_list = self.parent.edge_conf_list[:]
            self.solution = self.parent.solution.copy()

    def __lt__(self, other):
        return self.cost < other.cost

    def create_init_solution(self):
        for agent in self.agents:
            path, info = calc_temporal_a_star(
                curr_node=agent.start_node, goal_node=agent.next_goal_node, nodes_dict=self.nodes_dict,
                h_dict=self.h_dict, max_len=1000
            )
            self.solution[agent.name] = path
        self.cost = len(self.solution['agent_0'].path)


def validate_cbs_node(cbs_node: CBSNode) -> Tuple[bool, tuple, str, List[AlgCBSAgent]]:
    # returns: no_conf_bool, first_conf, conf_type, conf_agents
    pass


class ALgCBS:

    name = 'CBS-SACG'

    def __init__(self, env: SimEnvLMAPF, to_check_paths: bool = False, to_assert: bool = False, **kwargs):

        self.env = env
        self.to_check_paths: bool = to_check_paths
        self.to_assert: bool = to_assert
        # self.name = 'CGA'
        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np: np.ndarray = self.env.img_np
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func
        self.h_dict = self.env.h_dict
        self.is_sacg = self.env.is_sacg

        self.agents: List[AlgCBSAgent] = []
        self.agents_dict: Dict[str, AlgCBSAgent] = {}
        self.start_nodes: List[Node] = []

        self.max_time: int | None = self.env.max_time
        self.next_iteration: int = 0

        self.global_order: List[AlgCBSAgent] = []
        self.logs: dict | None = None

    def initiate_problem(self, obs: dict) -> bool:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._create_agents(obs)
        self.global_order = self.agents[:]
        self.logs = {
            'runtime': 0,
            'expanded_nodes': 0,
        }
        solved = self._solve()
        return solved

    def get_actions(self, obs: dict) -> Dict[str, str]:
        # updates
        self.next_iteration = obs['iteration']
        self._update_agents(obs)
        actions = {}
        for agent in self.agents:
            if len(agent.path) < self.next_iteration:
                actions[agent.name] = agent.path[self.next_iteration].xy_name
            else:
                actions[agent.name] = agent.path[-1].xy_name
        return actions

    def _create_agents(self, obs: dict) -> None:
        self.agents: List[AlgCBSAgent] = []
        self.agents_dict: Dict[str, AlgCBSAgent] = {}
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            if self.is_sacg and num != 0:
                next_goal_node = self.nodes_dict[obs_agent.curr_node_name]
            new_agent = AlgCBSAgent(num=num, start_node=start_node, next_goal_node=next_goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _update_agents(self, obs: dict) -> None:
        for agent in self.agents:
            obs_agent = obs[agent.name]
            agent.prev_node = agent.curr_node
            agent.curr_node = self.nodes_dict[obs_agent.curr_node_name]
            agent.next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            if self.is_sacg and agent.num != 0:
                agent.next_goal_node = self.nodes_dict[obs_agent.curr_node_name]
            agent.arrived = obs_agent.arrived
            agent.goals_per_iter_list.append(agent.next_goal_node)

    def _solve(self):
        # solve with CBS
        root = CBSNode(agents=self.agents, nodes=self.nodes, nodes_dict=self.nodes_dict, h_dict=self.h_dict)
        root.create_init_solution()
        open_list = [root]
        while len(open_list) > 0:
            next_cbs_node = heapq.heappop(open_list)
            no_conf_bool, first_conf, conf_type, conf_agents = validate_cbs_node(next_cbs_node)
            if no_conf_bool:
                for agent in self.agents:
                    agent.path = next_cbs_node.solution[agent.name]
                    return True
            for conf_agent in conf_agents:
                new_cbs_node = CBSNode(
                    agents=self.agents, nodes=self.nodes, nodes_dict=self.nodes_dict,
                    h_dict=self.h_dict, parent=next_cbs_node
                )
                new_cbs_node.add_constraint(conf_agent, first_conf, conf_type)
                succeeded = new_cbs_node.udpate_plan(conf_agent)
                if succeeded:
                    heapq.heappush(open_list, new_cbs_node)
        return False

        # for agent in self.agents:
        #     for _ in range(self.max_time):
        #         agent.path.append(agent.curr_node)



@use_profiler(save_dir='../stats/alg_CBS_SACG.pstat')
def main():
    # SACG
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_SACG()

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg, to_check_collisions=True)
    # env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg, to_check_collisions=False)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    # for rendering
    total_unique_moves_list = []
    total_finished_goals_list = []
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = ALgCBS(env=env, to_check_paths=to_check_paths)
    solved = alg.initiate_problem(obs=obs)
    if not solved:
        return

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        if to_render:
            i_agent = alg.global_order[0]
            plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
                         'n_agents': env.n_agents, 'agents': alg.agents, 'total_unique_moves_list': total_unique_moves_list,
                         'total_finished_goals_list': total_finished_goals_list, 'i_agent': i_agent, 'corridor': i_agent.path[i_step:]}
            plot_step_in_env(ax[0], plot_info)
            # plot_total_finished_goals(ax[1], plot_info)
            plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    # if to_render:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_info = {'i': max_time, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                 'n_agents': env.n_agents, 'agents': env.agents, 'total_unique_moves_list': total_unique_moves_list,
                 'total_finished_goals_list': total_finished_goals_list, }
    plot_step_in_env(ax[0], plot_info)
    # plot_total_finished_goals(ax[1], plot_info)
    plot_unique_movements(ax[1], plot_info)
    plt.show()
    if env.is_sacg:
        max_time = len(env.agents_dict['agent_0'].path)
    do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
                           'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')


if __name__ == '__main__':
    main()
