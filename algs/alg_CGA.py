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


class AlgCGAAgent:

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

    def __str__(self):
        return f'{self.name} ({self.start_node.xy_name}->{self.next_goal_node.xy_name})'

    def __repr__(self):
        return f'{self.name} ({self.start_node.xy_name}->{self.next_goal_node.xy_name})'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def last_path_node_name(self):
        return self.path[-1].xy_name

    def __eq__(self, other):
        return self.num == other.num


class ALgCGA:

    name = 'CGA'

    def __init__(self, env: SimEnvLMAPF, to_check_paths: bool = False, to_assert: bool = False, **kwargs):

        self.env = env
        self.to_check_paths: bool = to_check_paths
        self.to_assert: bool = to_assert
        # self.name = 'CGA'
        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np: np.ndarray = self.env.img_np
        self.freedom_nodes_np: np.ndarray = get_freedom_nodes_np(self.nodes, self.nodes_dict, self.img_np, self.img_dir)
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func
        self.h_dict = self.env.h_dict
        self.is_sacg = self.env.is_sacg

        self.agents: List[AlgCGAAgent] = []
        self.agents_dict: Dict[str, AlgCGAAgent] = {}
        self.start_nodes: List[Node] = []

        self.max_time: int | None = self.env.max_time
        self.corridor_size: int = self.env.corridor_size
        self.next_iteration: int = 0

        self.global_order: List[AlgCGAAgent] = []
        self.logs: dict | None = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def initiate_problem(self, obs: dict) -> bool:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._check_solvability()
        self._create_agents(obs)
        self.global_order = self.agents[:]
        self.logs = {
            'runtime': 0,
            'expanded_nodes': 0,
        }
        return True

    def get_actions(self, obs: dict) -> Dict[str, str]:
        # updates
        self.next_iteration = obs['iteration']
        self._update_agents(obs)
        self._update_global_order()
        # ---
        self._calc_next_steps()
        actions = {
            agent.name: agent.path[self.next_iteration].xy_name for agent in self.agents
        }
        self.logs['soc'] = len(self.agents_dict['agent_0'].path)
        return actions

    def _check_solvability(self) -> None:
        assert len(self.nodes) - len(self.start_nodes) >= self.corridor_size, 'UNSOLVABLE'

    def _create_agents(self, obs: dict) -> None:
        self.agents: List[AlgCGAAgent] = []
        self.agents_dict: Dict[str, AlgCGAAgent] = {}
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            if self.is_sacg and num != 0:
                next_goal_node = self.nodes_dict[obs_agent.curr_node_name]
            new_agent = AlgCGAAgent(num=num, start_node=start_node, next_goal_node=next_goal_node)
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

    def _update_global_order(self):
        unfinished_agents, finished_agents = [], []
        for agent in self.global_order:
            if agent.arrived:
                finished_agents.append(agent)
            else:
                unfinished_agents.append(agent)
        self.global_order = unfinished_agents
        self.global_order.extend(finished_agents)

    def _calc_next_steps(self) -> None:
        """
        for each agent:
            - if there have future steps in the path -> continue
            # if you here, that means you don't have any future moves
            - create your path while moving others out of your way (already created paths are walls for you)
            - if you succeeded, add yourself and those agents, that you affected, to the list of already created paths
            - if you didn't succeed to create a path -> be flexible for others
        for each agent:
            - if you still have no path (you didn't create it for yourself and others didn't do it for you) then stay at place for the next move
        :return:
        """
        start_time = time.time()
        # print(f'\n[{self.next_iteration}][{self.global_order.index(self.agents_dict['agent_0'])}] _calc_next_steps')
        # get all agents that already have plans for the future
        if self.to_assert:
            assert self.next_iteration != 0
        all_captured_agents: List[AlgCGAAgent] = []
        planned_agents: List[AlgCGAAgent] = []
        pa_heap: List[int] = []
        heapq.heapify(pa_heap)
        fresh_agents: List[AlgCGAAgent] = []
        for agent in self.global_order:
            if self.to_assert:
                assert agent.curr_node == agent.path[self.next_iteration - 1]
            if self.next_iteration % self.corridor_size == 0:
                agent.path = agent.path[:self.next_iteration]
            if len(agent.path[self.next_iteration:]) > 0:
                planned_agents.append(agent)
                heapq.heappush(pa_heap, agent.num)
                if self.to_assert:
                    assert agent.path[self.next_iteration].xy_name in agent.path[self.next_iteration - 1].neighbours
            else:
                fresh_agents.append(agent)

        p_counter = 0
        for agent in fresh_agents:
            # if there have future steps in the path -> continue
            if agent.num in pa_heap:
                continue
            if self.is_sacg and agent.num != 0:
                continue
            p_counter += 1
            # agents that are flexible for the current agent
            flex_agents: List[AlgCGAAgent] = [a for a in self.agents if a.num not in pa_heap and a != agent]
            if self.to_assert:
                assert agent not in flex_agents
                assert agent.num not in pa_heap
                for p_agent in planned_agents:
                    assert p_agent.curr_node == p_agent.path[self.next_iteration - 1]
                    assert len(p_agent.path[self.next_iteration:]) > 0
                    assert p_agent not in flex_agents
                for f_agent in flex_agents:
                    assert f_agent.curr_node == f_agent.path[self.next_iteration - 1]
                    assert len(f_agent.path[self.next_iteration:]) == 0

            # if you here, that means you don't have any future moves
            # create your path while moving others out of your way (already created paths are walls for you)
            # plan_up_to = self.corridor_size - self.next_iteration % self.corridor_size
            planned, captured_agents = self._plan_for_agent(agent, planned_agents, flex_agents, p_counter)

            # if you succeeded, add yourself and those agents, that you affected, to the list of already created paths
            if planned:
                agent.captured_agents = captured_agents
                if self.to_assert:
                    assert agent.curr_node == agent.path[self.next_iteration - 1]
                    assert len(agent.path[self.next_iteration:]) > 0
                    assert agent.path[self.next_iteration].xy_name in agent.path[self.next_iteration - 1].neighbours
                    for cap_agent in captured_agents:
                        assert cap_agent.curr_node == cap_agent.path[self.next_iteration - 1]
                        assert len(cap_agent.path[self.next_iteration:]) > 0
                        assert cap_agent.path[self.next_iteration].xy_name in cap_agent.path[
                            self.next_iteration - 1].neighbours
                    assert agent not in captured_agents

                planned_agents.append(agent)
                planned_agents.extend(captured_agents)
                all_captured_agents.extend(captured_agents)
                heapq.heappush(pa_heap, agent.num)
                for cap_agent in captured_agents:
                    heapq.heappush(pa_heap, cap_agent.num)
                continue

            # if you didn't succeed to create a path -> be flexible for others
            if self.to_assert:
                assert not planned
                assert len(captured_agents) == 0
                assert agent.curr_node == agent.path[self.next_iteration - 1]
                assert len(agent.path[self.next_iteration:]) == 0
                for f_agent in flex_agents:
                    assert f_agent.curr_node == f_agent.path[self.next_iteration - 1]
                    assert len(f_agent.path[self.next_iteration:]) == 0

        # path_horizon = self.next_iteration + (self.corridor_size - self.next_iteration % self.corridor_size)
        remained_agents: List[AlgCGAAgent] = list(filter(lambda a: a.num not in pa_heap, fresh_agents))
        for agent in remained_agents:
            # if you still have no path (you didn't create it for yourself and others didn't do it for you),
            # then stay at place for the next move
            if len(agent.path[self.next_iteration:]) == 0:
                agent.path.append(agent.path[-1])
                if self.to_assert:
                    assert agent.path[-2] == agent.curr_node
                    assert agent.path[-1] == agent.curr_node
                    assert agent not in all_captured_agents
                continue

            # if len(agent.path) < path_horizon:
            #     while len(agent.path) < path_horizon:
            #         agent.path.append(agent.path[-1])
            #     planned_agents.append(agent)
            #     if self.to_assert:
            #         assert agent.path[-2] == agent.curr_node
            #         assert agent.path[-1] == agent.curr_node
            #         assert agent not in all_captured_agents

            if self.to_assert:
                assert agent.curr_node == agent.path[self.next_iteration - 1]
                assert len(agent.path[self.next_iteration:]) > 0
                assert agent.path[self.next_iteration].xy_name in agent.path[self.next_iteration - 1].neighbours

        # check_vc_ec_neic_iter(self.agents, self.next_iteration)
        runtime = time.time() - start_time
        self.logs['runtime'] += runtime
        # --------------------------- #

    def _plan_for_agent(self, agent: AlgCGAAgent, planned_agents: List[AlgCGAAgent],
                        flex_agents: List[AlgCGAAgent], p_counter: int) -> Tuple[bool, List[AlgCGAAgent]]:
        """
        v- create a relevant map where the planned agents are considered as walls
        v- create a corridor to agent's goal in the given map to the max length straight through descending h-values
        v- if a corridor just a single node (that means it only contains the current location), then return False
        v # if you are here that means the corridor is of the length of 2 or higher
        v- check if there are any agents inside the corridor
        v- if there are no agents inside the corridor, then update agent's path and return True with []
        v # if you are here that means there are other agents inside the corridor (lets call them c_agents)
        v- let's define a list of tubes that will include the free nodes that those c_agents will potentially capture
        for c_agent in c_agents:
            v- try to create a tube + free_node for c_agent (a new free_node must be different from other free_nodes)
            v- if there is no tube, then return False
            v- tubes <- tube, free_node
        v # if you are here that means all c_agents found their tubes, and we are ready to move things
        v # up until now no one moved
        v- let's define captured_agents to be the list of all agents that we will move in addition to the main agent
        for tube in tubes:
            v- let's call all agents in the tube as t_agents
            v- move all t_agents forward such that the free node will be occupied, the last node will be free
               and the rest of the nodes inside a tube will remain state the same state
            v- add t_agents to the captured_agents
        v- finally, let's move the main agent through the corridor
        return True and captured_agents

        :param agent:
        :param planned_agents:
        :param flex_agents:
        :return: planned, captured_agents
        """
        # print(f'\r[{p_counter}][{agent.name}] _plan_for_agent', end='')
        node_name_to_f_agent_dict = {f_agent.curr_node.xy_name: f_agent for f_agent in flex_agents}
        node_name_to_f_agent_heap = list(node_name_to_f_agent_dict.keys())
        heapq.heapify(node_name_to_f_agent_heap)
        # create a relevant map where the planned agents are considered as walls
        new_map: np.ndarray = create_new_map(self.img_np, planned_agents, self.next_iteration)
        assert new_map[agent.curr_node.x, agent.curr_node.y] == 1

        # ------------------------- #
        corridor, c_agents, tubes = self._get_corridor_and_tubes(
            agent.curr_node, agent.next_goal_node, new_map, node_name_to_f_agent_dict, node_name_to_f_agent_heap)

        if len(corridor) == 1:
            assert corridor[-1] == agent.curr_node
            return False, []

        if len(c_agents) == 0:
            agent.path.extend(corridor[1:])
            return True, []

        # ------------------------- #
        # ------------------------- #

        # if you are here that means all c_agents found their tubes, and we are ready to move things
        if self.to_assert:
            for tube in tubes:
                assert len(tube) >= 2

        # up until now no one moved
        if self.to_assert:
            assert len(agent.path[self.next_iteration:]) == 0
            for f_agent in flex_agents:
                assert len(f_agent.path[self.next_iteration:]) == 0

        # let's define captured_agents to be the list of all agents that we will move in addition to the main agent
        captured_agents: List[AlgCGAAgent] = []
        captured_agents_names: List[str] = []
        heapq.heapify(captured_agents_names)

        if self.to_assert:
            for c_agent in c_agents:
                assert c_agent in flex_agents

        # NOW THE AGENTS WILL PLAN FUTURE STEPS
        for tube in tubes:
            # let's call all agents in the tube as t_agents
            t_agents: List[AlgCGAAgent] = find_t_agents(tube, flex_agents)
            # move all t_agents forward such that the free node will be occupied, the last node will be free,
            # and the rest of the nodes inside a tube will remain state the same state
            tube.move(t_agents, self.next_iteration, captured_agents)

            # CHECK FULL PATHS FORWARD
            if self.to_check_paths:
                check_paths(t_agents, self.next_iteration)
                agents_to_check = t_agents[:]
                agents_to_check.append(agent)
                check_paths(t_agents, self.next_iteration)

            for t_agent in t_agents:
                if t_agent.name not in captured_agents_names:
                    heapq.heappush(captured_agents_names, t_agent.name)
                    captured_agents.append(t_agent)

            if self.to_check_paths:
                check_paths(captured_agents, self.next_iteration)

        # finally, let's move the main agent through the corridor
        move_main_agent(agent, corridor, captured_agents, self.next_iteration)

        # CHECK FULL PATHS FORWARD planned, captured, agent
        if self.to_check_paths:
            # only planned
            check_paths(planned_agents, self.next_iteration)
            # only captured
            check_paths(captured_agents, self.next_iteration)
            # captured + agent
            agents_to_check = captured_agents[:]
            agents_to_check.append(agent)
            check_paths(agents_to_check, self.next_iteration)
            # captured + planned
            agents_to_check = planned_agents[:]
            agents_to_check.extend(captured_agents)
            check_paths(agents_to_check, self.next_iteration)
            # panned + agent
            agents_to_check = planned_agents[:]
            agents_to_check.append(agent)
            check_paths(agents_to_check, self.next_iteration)
            # all
            # agents_to_check = planned_agents[:]
            # agents_to_check.extend(captured_agents)
            # agents_to_check.append(agent)
            # check_paths(agents_to_check, self.next_iteration)
        return True, captured_agents

    def _get_corridor_and_tubes(self, curr_node: Node, goal_node: Node, new_map: np.ndarray,
                                node_name_to_f_agent_dict: Dict[str, AlgCGAAgent],
                                node_name_to_f_agent_heap: List[str]) -> Tuple[
        List[Node], List[AlgCGAAgent], List[Tube]]:
        """
        :return: corridor, c_agents, tubes
        """
        # ------------------------- #
        i_goal_node = goal_node
        for i_try in range(2):
            new_map[curr_node.x, curr_node.y] = 1
            # create a corridor to agent's goal in the given map to the max length straight through descending h-values
            corridor = calc_smart_corridor(curr_node, i_goal_node,  self.nodes_dict, self.h_dict, new_map,
                                           self.freedom_nodes_np, self.corridor_size, node_name_to_f_agent_heap)
            # corridor = calc_simple_corridor(agent, self.nodes_dict, self.h_dict, self.corridor_size, new_map)
            # corridor = calc_a_star_corridor(agent, self.nodes_dict, self.h_dict, self.corridor_size, new_map)
            # self.logs['expanded_nodes'] += len(corridor)
            self.logs['expanded_nodes'] += 1
            # if a corridor just a single node (that means it only contains the current location), then return False
            assert len(corridor) != 0
            if len(corridor) == 1:
                assert corridor[-1] == curr_node
                return corridor, [], []

            # if you are here that means the corridor is of the length of 2 or higher
            if self.to_assert:
                assert len(corridor) >= 2
                assert corridor[0] == curr_node  # the first node is agent's current location
                for n in corridor:
                    assert new_map[n.x, n.y]

            # check if there are any agents inside the corridor
            c_agents: List[AlgCGAAgent] = get_agents_in_corridor(corridor, node_name_to_f_agent_dict,
                                                                 node_name_to_f_agent_heap)

            # if there are no agents inside the corridor, then update agent's path and return True with []
            if len(c_agents) == 0:
                return corridor, [], []

            # if you are here that means there are other agents inside the corridor (lets call them c_agents)
            assert len(c_agents) > 0

            # the c_agents are not allowed to pass through current location of the agent
            corridor_for_c_agents = corridor[1:]
            new_map[curr_node.x, curr_node.y] = 0
            tubes: List[Tube] = []

            tubes_are_good: bool = True
            for c_agent in c_agents:
                # try to create a tube + free_node for c_agent (a new free_node must be different from other free_nodes)
                # c_agent_h_dict = self.h_dict[c_agent.next_goal_node.xy_name]
                solvable, tube, get_tube_info = get_tube(
                    c_agent, self.h_dict, new_map, tubes, corridor_for_c_agents, self.nodes_dict, node_name_to_f_agent_heap,
                    self.to_assert
                )
                self.logs['expanded_nodes'] += len(get_tube_info['open_list']) + len(get_tube_info['closed_list'])

                # if there is no tube, then return False
                if not solvable:
                    tubes_are_good = False
                    break

                # tubes <- tube, free_node
                tubes.append(tube)

            if not tubes_are_good:
                i_goal_node = find_nearest_freedom_node(curr_node, self.nodes_dict, self.freedom_nodes_np)
                continue

            return corridor, c_agents, tubes

        return [curr_node], [], []

        # ------------------------- #

    def _old_get_corridor_and_tubes(self, curr_node: Node, goal_node: Node, new_map: np.ndarray,
                                    node_name_to_f_agent_dict: Dict[str, AlgCGAAgent],
                                    node_name_to_f_agent_heap: List[str]) -> Tuple[
        List[Node], List[AlgCGAAgent], List[Tube]]:

        # ------------------------- #
        # # create a corridor to agent's goal in the given map to the max length straight through descending h-values
        # corridor = calc_smart_corridor(curr_node, goal_node, self.nodes_dict, self.h_dict, new_map,
        #                                self.freedom_nodes_np, self.corridor_size, node_name_to_f_agent_dict, node_name_to_f_agent_heap)
        # # corridor = calc_simple_corridor(agent, self.nodes_dict, self.h_dict, self.corridor_size, new_map)
        # # corridor = calc_a_star_corridor(agent, self.nodes_dict, self.h_dict, self.corridor_size, new_map)
        #
        # # if a corridor just a single node (that means it only contains the current location), then return False
        # assert len(corridor) != 0
        # if len(corridor) == 1:
        #     assert corridor[-1] == curr_node
        #     return False, []
        #
        # # if you are here that means the corridor is of the length of 2 or higher
        # assert len(corridor) >= 2
        # assert corridor[0] == curr_node  # the first node is agent's current location
        # for n in corridor:
        #     assert new_map[n.x, n.y]
        #
        # # check if there are any agents inside the corridor
        # c_agents: List[AlgCBSAgent] = get_agents_in_corridor(corridor, node_name_to_f_agent_dict,
        #                                                        node_name_to_f_agent_heap)
        #
        # # if there are no agents inside the corridor, then update agent's path and return True with []
        # if len(c_agents) == 0:
        #     agent.path.extend(corridor[1:])
        #
        #     # CHECK FULL PATHS FORWARD
        #     if self.to_check_paths:
        #         agents_to_check = planned_agents[:]
        #         agents_to_check.append(agent)
        #         check_paths(agents_to_check, self.next_iteration)
        #
        #     return True, []
        #
        # # if you are here that means there are other agents inside the corridor (lets call them c_agents)
        # assert len(c_agents) > 0
        #
        # # the c_agents are not allowed to pass through current location of the agent
        # corridor_for_c_agents = corridor[1:]
        # new_map[agent.curr_node.x, agent.curr_node.y] = 0
        # tubes: List[Tube] = []
        #
        # for c_agent in c_agents:
        #     # try to create a tube + free_node for c_agent (a new free_node must be different from other free_nodes)
        #     solvable, tube = get_tube(
        #         c_agent, new_map, tubes, corridor_for_c_agents, self.nodes_dict, node_name_to_f_agent_heap, self.to_assert
        #     )
        #
        #     # if there is no tube, then return False
        #     if not solvable:
        #         return False, []
        #
        #     # tubes <- tube, free_node
        #     tubes.append(tube)

        # ------------------------- #
        pass


@use_profiler(save_dir='../stats/alg_CGA.pstat')
def main():
    # SACG
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_SACG()
    # LMAPF
    # N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_LMAPF()

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
    alg = ALgCGA(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

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
            plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
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
    plot_total_finished_goals(ax[1], plot_info)
    # plot_unique_movements(ax[1], plot_info)
    plt.show()
    if env.is_sacg:
        max_time = len(env.agents_dict['agent_0'].path)
    do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
                           'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')


if __name__ == '__main__':
    main()

# # move others out of the corridor
# succeeded, cc_paths_dict = clean_corridor(
#     next_agent, corridor, c_agents, occupied_nodes, l_agents,
#     node_name_to_agent_dict, self.nodes, self.nodes_dict, self.img_np,
#     next_iteration=next_iteration
# )
# if not succeeded:
#     return False, []
#
# # assign new paths to the next_agent and other captured agents
#
# path_through_corridor = get_path_through_corridor(next_agent, corridor, cc_paths_dict)
# next_agent.path = next_agent.path[:next_iteration]
# next_agent.path.extend(path_through_corridor)
# for agent_name, alt_path in cc_paths_dict.items():
#     o_agent = self.agents_dict[agent_name]
#     o_agent.path = o_agent.path[:next_iteration]
#     o_agent.path.extend(alt_path)
#     captured_agents.append(o_agent)
#
# return True, captured_agents

# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# agents_to_plot = [a for a in flex_agents]
# agents_to_plot.append(agent)
# plot_info = {'img_np': new_map, 'agents': agents_to_plot}
# plot_step_in_env(ax[0], plot_info)
# plt.show()
