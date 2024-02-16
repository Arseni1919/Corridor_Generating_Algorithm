import matplotlib.pyplot as plt
from collections import deque

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_LMAPF import SimEnvLMAPF
from alg_gen_cor_v1 import copy_nodes
from alg_clean_corridor import *


def build_perm_constr_list(curr_iteration: int, finished_agents: list, nodes: List[Node]) -> Tuple[Dict[str, List[Node]], List[Node]]:
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    occupied_nodes: List[Node] = []
    for agent in finished_agents:
        a_path = agent.path[curr_iteration:]
        for n in a_path:
            perm_constr_dict[n.xy_name] = [0]
            occupied_nodes.append(n)
    return perm_constr_dict, occupied_nodes


def build_occupied_nodes(curr_iteration: int, finished_agents: list) -> List[Node]:
    occupied_nodes: List[Node] = []
    for agent in finished_agents:
        a_path = agent.path[curr_iteration:]
        for n in a_path:
            occupied_nodes.append(n)
    return occupied_nodes


def calc_corridor(next_agent, nodes, nodes_dict, h_func, corridor_size, perm_constr_dict: Dict[str, List[Node]]) -> \
        List[Node] | None:
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    result, info = a_star_xyt(
        start=next_agent.curr_node, goal=next_agent.next_goal_node, nodes=nodes, h_func=h_func,
        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict, perm_constr_dict=perm_constr_dict,
        plotter=None, middle_plot=False, nodes_dict=nodes_dict,
        xyt_problem=True, k_time=corridor_size,
    )
    return result


class AlgAgentLMAPF:

    def __init__(self, num: int, start_node: Node, next_goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.path: List[Node] = [start_node]
        self.arrived: bool = False

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def last_path_node_name(self):
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


class ALgLMAPFGenCor:
    def __init__(self, env: SimEnvLMAPF, to_check_paths: bool = False, to_assert: bool = False, **kwargs):

        self.env = env
        self.to_check_paths: bool = to_check_paths
        self.to_assert: bool = to_assert
        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np = self.env.img_np
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func

        self.agents: List[AlgAgentLMAPF] = []
        self.agents_dict: Dict[str, AlgAgentLMAPF] = {}
        self.start_nodes: List[Node] = []

        self.max_time: int | None = self.env.max_time
        self.corridor_size: int = self.env.corridor_size
        self.next_iteration: int = 0

        self.global_order: List[AlgAgentLMAPF] = []

    def initiate_problem(self, obs: dict) -> None:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._check_solvability()
        self._create_agents(obs)
        self.global_order = self.agents[:]

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
        return actions

    def _check_solvability(self) -> None:
        assert len(self.nodes) - len(self.start_nodes) >= self.corridor_size, 'UNSOLVABLE'

    def _create_agents(self, obs: dict) -> None:
        self.agents: List[AlgAgentLMAPF] = []
        self.agents_dict: Dict[str, AlgAgentLMAPF] = {}
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            new_agent = AlgAgentLMAPF(num=num, start_node=start_node, next_goal_node=next_goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _update_agents(self, obs: dict) -> None:
        for agent in self.agents:
            obs_agent = obs[agent.name]
            agent.prev_node = agent.curr_node
            agent.curr_node = self.nodes_dict[obs_agent.curr_node_name]
            agent.next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            agent.arrived = obs_agent.arrived

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
            - check
            # if you here, that means you don't have any future moves
            - create your path while moving others out of your way (already created paths are walls for you)
            - if you succeeded, add yourself and those agents, that you affected, to the list of already created paths
            - check
            - if you didn't succeed to create a path -> be flexible for others
            - check
        for each agent:
            - if you still have no path (you didn't create it for yourself and others didn't do it for you) then stay at place for the next move
            - check
        :return:
        """
        print(f'\n[{self.next_iteration}] _calc_next_steps')
        # get all agents that already have plans for the future
        if self.to_assert:
            assert self.next_iteration != 0
        all_captured_agents: List[AlgAgentLMAPF] = []  # only for debug
        planned_agents: List[AlgAgentLMAPF] = []
        for agent in self.global_order:
            if self.to_assert:
                assert agent.curr_node == agent.path[self.next_iteration - 1]
            if len(agent.path[self.next_iteration:]) > 0:
                planned_agents.append(agent)
                if self.to_assert:
                    assert agent.path[self.next_iteration].xy_name in agent.path[self.next_iteration - 1].neighbours

        p_counter = 0
        for agent in self.global_order:
            # if there have future steps in the path -> continue
            if agent in planned_agents:
                continue
            p_counter += 1
            # agents that are flexible for the current agent
            flex_agents: List[AlgAgentLMAPF] = [a for a in self.agents if a not in planned_agents and a != agent]
            if self.to_assert:
                assert agent not in flex_agents
                assert agent not in planned_agents
                for p_agent in planned_agents:
                    assert p_agent.curr_node == p_agent.path[self.next_iteration - 1]
                    assert len(p_agent.path[self.next_iteration:]) > 0
                    assert p_agent not in flex_agents
                for f_agent in flex_agents:
                    assert f_agent.curr_node == f_agent.path[self.next_iteration - 1]
                    assert len(f_agent.path[self.next_iteration:]) == 0

            # if you here, that means you don't have any future moves
            # create your path while moving others out of your way (already created paths are walls for you)
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
                        assert cap_agent.path[self.next_iteration].xy_name in cap_agent.path[self.next_iteration - 1].neighbours
                    assert agent not in captured_agents

                planned_agents.append(agent)
                planned_agents.extend(captured_agents)
                all_captured_agents.extend(captured_agents)  # only for debug
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

        path_horizon = self.next_iteration + (self.corridor_size - self.next_iteration % self.corridor_size)
        # print()
        for agent in self.global_order:
            # if you still have no path (you didn't create it for yourself and others didn't do it for you),
            # then stay at place for the next move
            # if len(agent.path[self.next_iteration:]) == 0:
            #     agent.path.append(agent.path[-1])
            #     planned_agents.append(agent)
            #     if self.to_assert:
            #         assert agent not in planned_agents

            if len(agent.path) < path_horizon:
                while len(agent.path) < path_horizon:
                    agent.path.append(agent.path[-1])
                planned_agents.append(agent)
                if self.to_assert:
                    assert agent.path[-2] == agent.curr_node
                    assert agent.path[-1] == agent.curr_node
                    assert agent not in all_captured_agents
            if self.to_assert:
                assert agent.curr_node == agent.path[self.next_iteration - 1]
                assert len(agent.path[self.next_iteration:]) > 0
                assert agent.path[self.next_iteration].xy_name in agent.path[self.next_iteration - 1].neighbours

        # check_vc_ec_neic_iter(self.agents, self.next_iteration)
        # --------------------------- #

    def _plan_for_agent(self, agent: AlgAgentLMAPF, planned_agents: List[AlgAgentLMAPF], flex_agents: List[AlgAgentLMAPF], p_counter: int) -> Tuple[bool, List[AlgAgentLMAPF]]:
        """
        v- create a relevant map where the planned agents are considered as walls
        - check
        v- create a corridor to agent's goal in the given map to the max length straight through descending h-values
        - check
        v- if a corridor just a single node (that means it only contains the current location), then return False
        - check
        v # if you are here that means the corridor is of the length of 2 or higher
        - check
        v- check if there are any agents inside the corridor
        - check
        v- if there are no agents inside the corridor, then update agent's path and return True with []
        - check
        v # if you are here that means there are other agents inside the corridor (lets call them c_agents)
        - check
        v- let's define a list of tubes that will include the free nodes that those c_agents will potentially capture
        for c_agent in c_agents:
            v- try to create a tube + free_node for c_agent (a new free_node must be different from other free_nodes)
            - check
            v- if there is no tube, then return False
            - check
            v- tubes <- tube, free_node
            - check
        - check
        v # if you are here that means all c_agents found their tubes, and we are ready to move things
        - check
        v # up until now no one moved
        - check
        v- let's define captured_agents to be the list of all agents that we will move in addition to the main agent
        - check
        for tube in tubes:
            - check
            v- let's call all agents in the tube as t_agents
            - check
            v- move all t_agents forward such that the free node will be occupied, the last node will be free
               and the rest of the nodes inside a tube will remain state the same state
            - check
            v- add t_agents to the captured_agents
            - check
        v- finally, let's move the main agent through the corridor
        - check
        return True and captured_agents

        :param agent:
        :param planned_agents:
        :param flex_agents:
        :return: planned, captured_agents
        """
        print(f'\r[{p_counter}][{agent.name}] _plan_for_agent', end='')
        assert agent not in planned_agents
        assert self.img_np[agent.curr_node.x, agent.curr_node.y] == 1
        # create a relevant map where the planned agents are considered as walls
        new_map: np.ndarray = create_new_map(self.img_np, planned_agents, self.next_iteration)
        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # agents_to_plot = [a for a in flex_agents]
        # agents_to_plot.append(agent)
        # plot_info = {'img_np': new_map, 'agents': agents_to_plot}
        # plot_step_in_env(ax[0], plot_info)
        # plt.show()
        assert new_map[agent.curr_node.x, agent.curr_node.y] == 1

        # create a corridor to agent's goal in the given map to the max length straight through descending h-values
        corridor = calc_simple_corridor(agent, self.nodes_dict, self.h_func, self.corridor_size, new_map)
        # if a corridor just a single node (that means it only contains the current location), then return False
        assert len(corridor) != 0
        if len(corridor) == 1:
            assert corridor[-1] == agent.curr_node
            return False, []

        # if you are here that means the corridor is of the length of 2 or higher
        assert len(corridor) >= 2
        assert corridor[0] == agent.curr_node  # the first node is agent's current location
        for n in corridor:
            assert new_map[n.x, n.y]

        # check if there are any agents inside the corridor
        c_agents: List[AlgAgentLMAPF] = get_agents_in_corridor(corridor, flex_agents)

        # if there are no agents inside the corridor, then update agent's path and return True with []
        if len(c_agents) == 0:
            agent.path.extend(corridor[1:])

            # CHECK FULL PATHS FORWARD
            if self.to_check_paths:
                agents_to_check = planned_agents[:]
                agents_to_check.append(agent)
                check_paths(agents_to_check, self.next_iteration)

            return True, []

        # if you are here that means there are other agents inside the corridor (lets call them c_agents)
        assert len(c_agents) > 0

        # the c_agents are not allowed to pass through current location of the agent
        corridor_for_c_agents = corridor[1:]
        new_map[agent.curr_node.x, agent.curr_node.y] = 0
        tubes: List[Tube] = []

        for c_agent in c_agents:
            # try to create a tube + free_node for c_agent (a new free_node must be different from other free_nodes)
            solvable, tube = get_tube(
                c_agent, new_map, tubes, corridor_for_c_agents, self.nodes_dict, flex_agents
            )

            # if there is no tube, then return False
            if not solvable:
                return False, []

            # tubes <- tube, free_node
            tubes.append(tube)

        # if you are here that means all c_agents found their tubes, and we are ready to move things
        for tube in tubes:
            assert len(tube) >= 2

        # up until now no one moved
        assert len(agent.path[self.next_iteration:]) == 0
        for f_agent in flex_agents:
            assert len(f_agent.path[self.next_iteration:]) == 0

        # let's define captured_agents to be the list of all agents that we will move in addition to the main agent
        captured_agents: List[AlgAgentLMAPF] = []

        for c_agent in c_agents:
            assert c_agent in flex_agents

        # NOW THE AGENTS WILL PLAN FUTURE STEPS
        for tube in tubes:
            # let's call all agents in the tube as t_agents
            t_agents: List[AlgAgentLMAPF] = find_t_agents(tube, flex_agents)
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
                if t_agent not in captured_agents:
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

        # agent_to_check = planned_agents[:]
        # agent_to_check.extend(captured_agents)
        # agent_to_check.append(agent)
        # check_vc_ec_neic_iter(agent_to_check, self.next_iteration)
        return True, captured_agents

        # ----------------------------- #


@use_profiler(save_dir='../stats/alg_LMAPF_gen_cor_v1.pstat')
def main():
    set_seed(random_seed_bool=False, seed=601)
    # set_seed(random_seed_bool=True)
    # N = 70
    # N = 100
    N = 300
    # N = 400
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 850
    # N = 2000
    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    # max_time = 20
    max_time = 100
    # corridor_size = 10
    corridor_size = 5
    # corridor_size = 3

    to_render: bool = True
    # to_render: bool = False

    # to_check_paths: bool = True
    to_check_paths: bool = False

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    # for rendering
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        total_unique_moves_list = []
        total_finished_goals_list = []

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = ALgLMAPFGenCor(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        if to_render:
            total_unique_moves_list.append(metrics['total_unique_moves'])
            total_finished_goals_list.append(metrics['total_finished_goals'])
            i_agent = alg.global_order[0]
            plot_info = {
                'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
                'n_agents': env.n_agents, 'agents': alg.agents,
                'total_unique_moves_list': total_unique_moves_list,
                'total_finished_goals_list': total_finished_goals_list,
                'i_agent': i_agent,
                'corridor': i_agent.path[i_step:]
            }
            plot_step_in_env(ax[0], plot_info)
            plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    if to_render:
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        plot_info = {
            'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
            'n_agents': env.n_agents, 'agents': env.agents,
            'total_unique_moves_list': total_unique_moves_list,
            'total_finished_goals_list': total_finished_goals_list,
        }
        plot_step_in_env(ax[0], plot_info)
        plot_total_finished_goals(ax[1], plot_info)
        # plot_unique_movements(ax[1], plot_info)
        plt.show()
    print(f'finished run')


if __name__ == '__main__':
    main()




# # move others out of the corridor
# succeeded, cc_paths_dict = clean_corridor(
#     next_agent, corridor, c_agents, occupied_nodes, l_agents,
#     node_name_to_agent_dict, self.nodes, self.nodes_dict, self.img_np,
#     curr_iteration=curr_iteration
# )
# if not succeeded:
#     return False, []
#
# # assign new paths to the next_agent and other captured agents
#
# path_through_corridor = get_path_through_corridor(next_agent, corridor, cc_paths_dict)
# next_agent.path = next_agent.path[:curr_iteration]
# next_agent.path.extend(path_through_corridor)
# for agent_name, alt_path in cc_paths_dict.items():
#     o_agent = self.agents_dict[agent_name]
#     o_agent.path = o_agent.path[:curr_iteration]
#     o_agent.path.extend(alt_path)
#     captured_agents.append(o_agent)
#
# return True, captured_agents