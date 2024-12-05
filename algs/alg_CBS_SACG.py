import heapq
import random
from itertools import combinations

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
        return f'{self.name} ({self.start_node.xy_name}->{self.next_goal_node.xy_name})'

    def __repr__(self):
        return f'{self.name} ({self.start_node.xy_name}->{self.next_goal_node.xy_name})'


class CBSNode:
    def __init__(self, cbs_i: int, agents: List[AlgCBSAgent], main_agent: AlgCBSAgent, nodes, nodes_dict, h_dict, map_dim, parent=None):
        self.cbs_i = cbs_i
        self.agents = agents
        self.main_agent = main_agent
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.h_dict = h_dict
        self.map_dim = map_dim
        self.vertex_conf_dict: Dict[str, List[Tuple[Node, int]]]= {a.name: [] for a in self.agents}
        self.edge_conf_dict: Dict[str, List[Tuple[Node, Node, int]]] = {a.name: [] for a in self.agents}
        self.solution: Dict[str, List[Node]] = {a.name: [] for a in self.agents}
        self.cost = 1e10
        self.parent = parent
        if self.parent is not None:
            for agent in self.agents:
                self.vertex_conf_dict[agent.name] = self.parent.vertex_conf_dict[agent.name][:]
                self.edge_conf_dict[agent.name] = self.parent.edge_conf_dict[agent.name][:]
                self.solution[agent.name] = self.parent.solution[agent.name][:]

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f'CBSNode([{self.cbs_i}]-{self.cost})'

    def __repr__(self):
        return f'CBSNode([{self.cbs_i}]-{self.cost})'

    def create_init_solution(self):
        for agent in self.agents:
            path, info = calc_temporal_a_star(
                curr_node=agent.start_node, goal_node=agent.next_goal_node, nodes_dict=self.nodes_dict,
                h_dict=self.h_dict, max_len=1000
            )
            self.solution[agent.name] = path
        self.cost = len(self.solution['agent_0'])

    def add_constraint(
            self, conf_agent: AlgCBSAgent, first_conf: Tuple[Node, int] | Tuple[Node, Node, int], conf_type: str
    ) -> None:
        if conf_type == 'vertex':
            n, i = first_conf
            self.vertex_conf_dict[conf_agent.name].append((n, i))
        elif conf_type == 'edge':
            from_n, to_n, i = first_conf
            self.edge_conf_dict[conf_agent.name].append((from_n, to_n, i))
            self.edge_conf_dict[conf_agent.name].append((to_n, from_n, i))
        else:
            raise RuntimeError('no way')

    def update_plan(self, conf_agent: AlgCBSAgent) -> bool:
        """
        Update solution and cost
        """
        conf_vertex_list = self.vertex_conf_dict[conf_agent.name]
        conf_edge_list = self.edge_conf_dict[conf_agent.name]
        max_conf_time = len(self.solution['agent_0'])
        vc_np, ec_np, pc_np = get_np_constraints(self.map_dim, max_conf_time, conf_vertex_list, conf_edge_list)
        any_goal_bool = conf_agent.name != 'agent_0'  # false for agent_0 and true for others
        path, info = calc_temporal_a_star(
            curr_node=conf_agent.start_node, goal_node=conf_agent.next_goal_node, nodes_dict=self.nodes_dict,
            h_dict=self.h_dict, max_len=1000, vc_np=vc_np, ec_np=ec_np, pc_np=pc_np, any_goal_bool=any_goal_bool
        )
        if path is not None:
            max_path_len = max(len(path), len(self.solution[self.main_agent.name]))
            self.solution[conf_agent.name] = extend_path(max_path_len, path)
            # self.solution[conf_agent.name] = path
            self.cost = len(self.solution['agent_0'])
            return True
        return False


class ALgCBS:

    name = 'CBS_SACG'

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
        self.main_agent: AlgCBSAgent | None = None
        self.start_nodes: List[Node] = []
        self.open_list: List[CBSNode] = []

        self.max_time: int | None = self.env.max_time
        self.next_iteration: int = 0

        self.global_order: List[AlgCBSAgent] = []
        self.logs: dict | None = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def initiate_problem(self, obs: dict) -> bool:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._create_agents(obs)
        self.global_order = self.agents[:]
        self.open_list = []
        self.logs = {
            'runtime': 0,
            'expanded_nodes': 0,
        }
        solved = self._solve()
        if solved:
            self.logs['soc'] = len(self.main_agent.path)
        return solved

    def get_actions(self, obs: dict) -> Dict[str, str]:
        # updates
        self.next_iteration = obs['iteration']
        self._update_agents(obs)
        actions = {}
        for agent in self.agents:
            if self.next_iteration < len(agent.path):
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
        self.main_agent = self.agents[0]
        assert self.main_agent.name == 'agent_0'

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

    def _upload_paths(self, cbs_node: CBSNode):
        max_path_len = len(cbs_node.solution[self.main_agent.name])
        for agent in self.agents:
            agent.path = cbs_node.solution[agent.name]
            agent.path = extend_path(max_path_len, agent.path)

    def _solve(self):
        # solve with CBS
        cbs_i = 0
        root = CBSNode(
            cbs_i=cbs_i, agents=self.agents, main_agent=self.main_agent, nodes=self.nodes, nodes_dict=self.nodes_dict, h_dict=self.h_dict, map_dim=self.map_dim
        )
        root.create_init_solution()
        self.open_list = [root]
        while len(self.open_list) > 0:
            next_cbs_node = heapq.heappop(self.open_list)
            no_conf_bool, first_conf, conf_type, conf_agents = validate_cbs_node(next_cbs_node)
            if no_conf_bool:
                self._upload_paths(next_cbs_node)
                return True
            for conf_agent in conf_agents:
                cbs_i += 1
                new_cbs_node = CBSNode(
                    cbs_i=cbs_i, agents=self.agents, main_agent=self.main_agent, nodes=self.nodes, nodes_dict=self.nodes_dict,
                    h_dict=self.h_dict, map_dim=self.map_dim, parent=next_cbs_node
                )
                new_cbs_node.add_constraint(conf_agent, first_conf, conf_type)
                succeeded = new_cbs_node.update_plan(conf_agent)
                if succeeded:
                    heapq.heappush(self.open_list, new_cbs_node)
        return False

def validate_cbs_node(
        cbs_node: CBSNode
) -> Tuple[bool, Tuple[Node, int] | Tuple[Node, Node, int] | tuple, str, List[AlgCBSAgent]]:
    # returns: no_conf_bool, first_conf, conf_type, conf_agents
    for a1, a2 in combinations(cbs_node.agents, 2):
        if cbs_node.cbs_i == 18 and a1.num == 0 and a2.num == 29:
            print('', end='')
        a1_path = cbs_node.solution[a1.name]
        a2_path = cbs_node.solution[a2.name]
        max_iter = max(len(a1_path), len(a2_path))
        for i in range(max_iter):
            a1_iter = min(i, len(cbs_node.solution[a1.name]) - 1)
            a2_iter = min(i, len(cbs_node.solution[a2.name]) - 1)
            a1_prev_iter = max(0, a1_iter - 1)
            a2_prev_iter = max(0, a2_iter - 1)
            curr_node_1 = cbs_node.solution[a1.name][a1_iter]
            curr_node_2 = cbs_node.solution[a2.name][a2_iter]
            prev_node_1 = cbs_node.solution[a1.name][a1_prev_iter]
            prev_node_2 = cbs_node.solution[a2.name][a2_prev_iter]
            # vertex conf
            if curr_node_1 == curr_node_2:
                conf_vertex = (curr_node_1, i)
                return False, conf_vertex, 'vertex', [a1, a2]  # vertex conflict
            # edge conf
            edge1 = (prev_node_1.x, prev_node_1.y, curr_node_1.x, curr_node_1.y)
            edge2 = (curr_node_2.x, curr_node_2.y, prev_node_2.x, prev_node_2.y)
            if edge1 == edge2:
                conf_edge = (cbs_node.solution[a1.name][i], cbs_node.solution[a1.name][i + 1], i)
                # edge_conf_dict: Dict[str, List[Tuple[str, str, int]]] = {}
                return False, conf_edge, 'edge', [a1, a2]  # edge conflict
    return True, (), '', []  # no conflict


def get_np_constraints(
        map_dim: tuple[int, int], max_conf_time: int,
        conf_vertex_list: List[Tuple[Node, int]], conf_edge_list: List[Tuple[Node, Node, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vc_np = np.zeros((map_dim[0], map_dim[1], max_conf_time))
    ec_np = np.zeros((map_dim[0], map_dim[1], map_dim[0], map_dim[1], max_conf_time))
    pc_np = np.ones((map_dim[0], map_dim[1])) * -1
    for conf_v in conf_vertex_list:
        vc_np[conf_v[0].x, conf_v[0].y, conf_v[1]] = 1
    for conf_e in conf_edge_list:
        ec_np[conf_e[0].x, conf_e[0].y, conf_e[1].x, conf_e[1].y, conf_e[2]] = 1
    return vc_np, ec_np, pc_np


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

# for agent in self.agents:
#     for _ in range(self.max_time):
#         agent.path.append(agent.curr_node)
