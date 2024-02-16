import matplotlib.pyplot as plt
from collections import deque

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from alg_gen_cor_v1 import get_full_tube, get_assign_agent_to_node_dict


class Tube:
    def __init__(self, nodes: List[Node], free_node: Node, tube_pattern: List[int]):
        self.nodes: List[Node] = nodes
        self.free_node: Node = free_node
        self.tube_pattern = tube_pattern
        assert free_node == nodes[0]

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, item):
        return item in self.nodes

    @property
    def node_names(self):
        return [n.xy_name for n in self.nodes]

    @property
    def creator_name(self):
        return self.free_node.xy_name

    def get_from_pattern(self, from_nodes: List[Node]) -> List[int]:
        from_pattern = []
        for n in self.nodes:
            if n in from_nodes:
                from_pattern.append(0)
            else:
                from_pattern.append(1)
        return from_pattern

    def get_to_nodes(self, from_nodes: List[Node], from_pattern: List[int]) -> List[Node]:
        assert from_pattern[0] == 1
        to_pattern = from_pattern[:]
        to_pattern[0] = 0
        to_pattern[-1] = 1
        to_nodes = []
        for n, pat in zip(self.nodes, to_pattern):
            if pat == 0:
                to_nodes.append(n)
            if len(to_nodes) == len(from_nodes):
                break
        # to_nodes = [self.free_node]
        # to_nodes.extend(from_nodes[:-1])
        return to_nodes

    def get_max_time(self, next_iteration: int, captured_agents: list) -> int:
        freedom_times = {n.xy_name: 0 for n in self.nodes}
        for cap_agent in captured_agents:
            for i, node in enumerate(cap_agent.path[next_iteration - 1:]):
                if node.xy_name in freedom_times:
                    curr_v = freedom_times[node.xy_name]
                    freedom_times[node.xy_name] = max(curr_v, i)
        freedom_times_v = list(freedom_times.values())
        max_time = max(freedom_times_v)
        # from_times = [len(t_agent.path[next_iteration:]) for t_agent in t_agents]
        # max_time = max(from_times)
        return max_time

    def move(self, t_agents: list, next_iteration: int, captured_agents: list) -> None:
        """
        move all t_agents forward such that the free node will be occupied, the last node will be free,
        and the rest of the nodes inside a tube will remain state the same state
        v- find start locations of t_agents
        v- all t_agents wait until the last of them will arrive to its start locations from movements of other tubes
        v- assign to each t_agent its final location
        v- move all agents along the tube until they reach their final locations
        :param t_agents:
        :param next_iteration:
        :param captured_agents:
        :return: None
        """
        agent_to_final_node_dict = {}
        if len(t_agents) > 0:

            # find start locations of t_agents
            from_nodes = [t_agent.path[-1] for t_agent in t_agents]
            from_pattern = self.get_from_pattern(from_nodes)

            # all t_agents wait until the last of them will arrive to its start locations from movements of other tubes
            max_time = self.get_max_time(next_iteration, captured_agents)
            for t_agent in t_agents:
                while len(t_agent.path[next_iteration:]) < max_time:
                    t_agent.path.append(t_agent.path[-1])
            for t_agent in t_agents:
                assert len(t_agent.path[next_iteration:]) == max_time
            to_nodes = self.get_to_nodes(from_nodes, from_pattern)
            # assign to each t_agent its final location
            assert self.free_node not in from_nodes  # the free node is supposed to be without agent in it
            # assert self.nodes[-1] in from_nodes
            # assert self.nodes[-1] == from_nodes[-1]

            agent_to_final_node_dict: Dict[str, Node] = {}
            for to_node, t_agent in zip(to_nodes, t_agents):
                agent_to_final_node_dict[t_agent.name] = to_node
            assert len(agent_to_final_node_dict) == len(from_nodes)

        # move all agents along the tube until they reach their final locations
        there_is_movement = True
        while there_is_movement:
            there_is_movement = False
            step_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}
            pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(self.nodes)
            for to_t_node, from_t_node in pairwise_tube:
                if from_t_node.xy_name in step_dict:
                    curr_agent = step_dict[from_t_node.xy_name]
                    if curr_agent.name in agent_to_final_node_dict and agent_to_final_node_dict[curr_agent.name] == from_t_node:
                        # curr_agent.path.append(from_t_node)
                        continue
                    elif to_t_node.xy_name in step_dict:
                        curr_agent.path.append(from_t_node)
                        # continue
                    else:
                        curr_agent.path.append(to_t_node)
                        there_is_movement = True
                        step_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}
        for t_agent in t_agents:
            assert t_agent.path[-1] == agent_to_final_node_dict[t_agent.name]


def find_t_agents(tube: Tube, flex_agents: list) -> list:
    t_agents = []
    node_name_to_agent_dict = {f_agent.path[-1].xy_name: f_agent for f_agent in flex_agents}
    for n in tube.nodes:
        if n.xy_name in node_name_to_agent_dict:
            t_agents.append(node_name_to_agent_dict[n.xy_name])
    return t_agents


def move_main_agent(agent, corridor: List[Node], captured_agents: list, next_iteration: int) -> None:
    """
    - build the list minimum times per locations in the corridor
    - proceed with the agent through the corridor while preserving the minimum times
    :param agent:
    :param corridor:
    :param captured_agents:
    :param next_iteration:
    :return: None
    """
    assert agent.path[-1] == agent.curr_node
    assert agent.curr_node == corridor[0]
    # build the list minimum times per locations in the corridor
    min_times_dict = OrderedDict([(n.xy_name, 0) for n in corridor])
    for cap_agent in captured_agents:
        for i, i_node in enumerate(cap_agent.path[next_iteration - 1:]):
            if i_node.xy_name in min_times_dict:
                min_times_dict[i_node.xy_name] = max(i, min_times_dict[i_node.xy_name])

    # proceed with the agent through the corridor while preserving the minimum times
    pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(corridor)
    path = [agent.curr_node]
    for from_t_node, to_t_node in pairwise_tube:
        min_time = min_times_dict[to_t_node.xy_name]
        while len(path) <= min_time:
            path.append(from_t_node)
        path.append(to_t_node)
    agent.path.extend(path[1:])


def create_new_map(img_np: np.ndarray, planned_agents: list, next_iteration: int) -> np.ndarray:
    # 1 - free space, 0 - occupied space
    new_map = np.copy(img_np)
    for p_agent in planned_agents:
        path: List[Node] = p_agent.path[next_iteration-1:]
        assert len(path) > 1
        for n in path:
            new_map[n.x, n.y] = 0
    return new_map


def calc_simple_corridor(agent, nodes_dict: Dict[str, Node], h_func, corridor_size: int, new_map: np.ndarray) -> List[Node] | None:
    corridor: List[Node] = [agent.curr_node]
    for i in range(corridor_size):
        next_node = corridor[-1]
        node_name_to_h_value_dict = {
            node_name: h_func(agent.next_goal_node, nodes_dict[node_name])
            for node_name in next_node.neighbours
        }
        min_node_name = min(node_name_to_h_value_dict, key=node_name_to_h_value_dict.get)
        min_node = nodes_dict[min_node_name]
        # 1 - free space, 0 - occupied space
        if new_map[min_node.x, min_node.y] == 0:
            return corridor
        corridor.append(min_node)
    return corridor


def get_agents_in_corridor(corridor: List[Node], flex_agents) -> list:
    agents_in_corridor = []
    node_name_to_f_agent_dict = {f_agent.curr_node.xy_name: f_agent for f_agent in flex_agents}
    for n in corridor:
        if n.xy_name in node_name_to_f_agent_dict:
            agents_in_corridor.append(node_name_to_f_agent_dict[n.xy_name])
    return agents_in_corridor


def tube_is_free_to_go(tube: List[Node], inner_captured_nodes: list) -> bool:
    # tube: free node -> init node
    for n in tube:
        if n in inner_captured_nodes:
            return False
    return True


def get_tube(
        c_agent,
        new_map: np.ndarray,
        tubes: List[Tube],
        corridor_for_c_agents: List[Node],
        nodes_dict: Dict[str, Node],
        flex_agents: list,
) -> Tuple[bool, Tube | None]:
    """
    - get a list of captured_free_nodes that are already taken by other agents
    - find a new free node
    - if no more nodes to search return False and None
    - if there is a free node, create a Tube for it and return True, Tube
    :param c_agent:
    :param new_map:
    :param tubes:
    :param corridor_for_c_agents:
    :param nodes_dict:
    :param flex_agents:
    :return: solvable, Tube(nodes, free_node)
    """
    captured_free_nodes: List[Node] = [tube.free_node for tube in tubes]
    flex_agents_nodes: List[Node] = [f_agent.curr_node for f_agent in flex_agents]

    for cap_node in captured_free_nodes:
        assert cap_node not in flex_agents_nodes

    spanning_tree_dict: Dict[str, str | None] = {c_agent.curr_node.xy_name: None}
    open_list: Deque[Node] = deque([c_agent.curr_node])
    closed_list: Deque[Node] = deque()
    small_iteration: int = 0
    while len(open_list) > 0:
        small_iteration += 1

        selected_node = open_list.pop()
        if selected_node not in corridor_for_c_agents and selected_node not in flex_agents_nodes and selected_node not in captured_free_nodes:
            nodes, tube_pattern = get_full_tube(selected_node, spanning_tree_dict, nodes_dict, flex_agents_nodes)
            tube = Tube(nodes, selected_node, tube_pattern)
            return True, tube

        corridor_nodes: List[Node] = []
        outer_nodes: List[Node] = []
        for nei_name in selected_node.neighbours:
            if nei_name == selected_node.xy_name:
                continue
            nei_node = nodes_dict[nei_name]
            # 1 - free space, 0 - occupied space
            if new_map[nei_node.x, nei_node.y] == 0:
                continue
            if nei_node in closed_list:
                continue
            if nei_node in open_list:
                continue
            # connect nei_note to selected one
            spanning_tree_dict[nei_node.xy_name] = selected_node.xy_name
            if nei_node in corridor_for_c_agents:
                corridor_nodes.append(nei_node)
            else:
                outer_nodes.append(nei_node)
        open_list.extendleft(outer_nodes)
        open_list.extendleft(corridor_nodes)
        closed_list.append(selected_node)
    return False, None


def roll_c_agent(
        next_c_agent,
        tube: List[Node],
        l_agents_paths_dict: Dict[str, List[Node]],
        l_agents: list,
        corridor: List[Node],
        executed_tubes_dict: Dict[str, List[Node]],
) -> None:

    node_name_to_agent_dict = {l_agents_paths_dict[agent.name][-1].xy_name: agent for agent in l_agents}
    t_agents: list = [node_name_to_agent_dict[n.xy_name] for n in tube if n.xy_name in node_name_to_agent_dict]
    assert next_c_agent in t_agents

    # if need to wait
    time_to_wait = 0
    for ex_agent_name, ex_tube in executed_tubes_dict.items():
        if len(intersection([n.xy_name for n in tube], [n.xy_name for n in ex_tube])) > 0:
            time_to_wait = max(time_to_wait, len(l_agents_paths_dict[ex_agent_name]))
    for t_agent in t_agents:
        while len(l_agents_paths_dict[t_agent.name]) < time_to_wait:
            l_agents_paths_dict[t_agent.name].append(l_agents_paths_dict[t_agent.name][-1])

    # out_of_t_agents: list = [agent for agent in l_agents if agent not in t_agents]
    assign_agent_to_t_node_dict = get_assign_agent_to_node_dict(tube, t_agents, corridor)
    there_is_movement = True
    while there_is_movement:
        there_is_movement = False
        node_name_to_agent_dict = {l_agents_paths_dict[t_agent.name][-1].xy_name: t_agent for t_agent in t_agents}
        pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(tube)
        for to_t_node, from_t_node in pairwise_tube:
            if to_t_node == tube[0] and to_t_node.xy_name in node_name_to_agent_dict:
                front_agent = node_name_to_agent_dict[to_t_node.xy_name]
                l_agents_paths_dict[front_agent.name].append(to_t_node)  # path change
            if from_t_node.xy_name in node_name_to_agent_dict:
                curr_agent = node_name_to_agent_dict[from_t_node.xy_name]
                if assign_agent_to_t_node_dict[curr_agent.name] == from_t_node:
                    l_agents_paths_dict[curr_agent.name].append(from_t_node)  # path change
                elif to_t_node.xy_name in node_name_to_agent_dict:
                    l_agents_paths_dict[curr_agent.name].append(from_t_node)  # path change
                else:
                    l_agents_paths_dict[curr_agent.name].append(to_t_node)  # path change
                    there_is_movement = True
                    node_name_to_agent_dict = {l_agents_paths_dict[t_agent.name][-1].xy_name: t_agent for t_agent in t_agents}

        len_list: List[int] = [len(l_agents_paths_dict[t_agent.name]) for t_agent in t_agents]
        assert len(set(len_list)) == 1
        assert list(set(len_list))[0] > 1


def clean_corridor(
        next_agent,
        input_corridor: List[Node],
        input_agents_in_corridor: list,
        input_occupied_nodes: List[Node],
        l_agents: list,
        node_name_to_agent_dict: dict,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        img_np: np.ndarray,
        # to_render: bool = False,
        to_render: bool = True,
        curr_iteration: int = 0
) -> Tuple[bool, dict | None]:

    assert len(input_corridor) > 1
    occupied_nodes = input_occupied_nodes[:]
    occupied_nodes.append(next_agent.curr_node)
    corridor: List[Node] = input_corridor[1:]
    agents_in_corridor: deque = deque(input_agents_in_corridor)
    l_agents_paths_dict = {a.name: [a.curr_node] for a in l_agents}

    for a in agents_in_corridor:
        assert a in l_agents

    # if to_render:
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    #     agents_to_plot = [a for a in l_agents if len(l_agents_paths_dict[a.name]) == 1]
    #     plot_info = {'img_np': img_np, 'agents': agents_to_plot, 'corridor': corridor,
    #                  'i_agent': next_agent, 'to_title': 'from clean_corridor 1',
    #                  'i': curr_iteration, 'n_agents': len(l_agents),
    #                  'occupied_nodes': occupied_nodes}
    #     plot_step_in_env(ax[0], plot_info)
    #     plt.show()
    #     plt.close()

    executed_tubes_dict: Dict[str, List[Node]] = {}
    counter = 0
    while len(agents_in_corridor) > 0:
        counter += 1
        # print(f'<{counter=}> {len(agents_in_corridor)=}')
        next_c_agent = agents_in_corridor.popleft()
        assert next_c_agent in l_agents
        # if to_render and curr_iteration > 4:
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        #     agents_to_plot = [a for a in l_agents if len(l_agents_paths_dict[a.name]) == 1]
        #     plot_info = {'img_np': img_np, 'agents': agents_to_plot, 'corridor': corridor,
        #                  'i_agent': next_c_agent, 'to_title': 'from clean_corridor 2',
        #                  'i': curr_iteration, 'n_agents': len(l_agents),
        #                  'occupied_nodes': occupied_nodes}
        #     plot_step_in_env(ax[0], plot_info)
        #     plt.show()
        #     plt.close()

        solvable, free_to_roll, tube = get_tube(
            next_c_agent, l_agents_paths_dict, l_agents, corridor, nodes, nodes_dict, occupied_nodes
        )

        if not solvable:
            # no tube
            assert tube is None
            return False, None

        if not free_to_roll:
            # there is a tube
            agents_in_corridor.append(next_c_agent)
            continue

        roll_c_agent(next_c_agent, tube, l_agents_paths_dict, l_agents, corridor, executed_tubes_dict)
        executed_tubes_dict[next_c_agent.name] = tube

    cc_paths_dict = {k: v for k, v in l_agents_paths_dict.items() if len(v) > 1}
    return True, cc_paths_dict


def get_path_through_corridor(
        next_agent,
        input_corridor: List[Node],
        cc_paths_dict: Dict[str, List[Node]],
) -> List[Node]:
    # corridor includes the curr_pos of the agent and is ordered correctly
    corridor: Deque[Node] = deque(input_corridor)
    free_time_for_node_dict: Dict[str, int] = {n.xy_name: 0 for n in corridor if n.xy_name != next_agent.curr_node.xy_name}
    for cca_name, cca_path in cc_paths_dict.items():
        for i, node in enumerate(cca_path):
            if node in corridor:
                curr_t = free_time_for_node_dict[node.xy_name]
                free_time_for_node_dict[node.xy_name] = max(curr_t, i)


    path_through_corridor: List[Node] = []
    while len(corridor) > 0:
        next_node = corridor.popleft()
        if next_node.xy_name in free_time_for_node_dict:
            free_t = free_time_for_node_dict[next_node.xy_name]
            if len(path_through_corridor) <= free_t:
                path_through_corridor.append(path_through_corridor[-1])
                corridor.appendleft(next_node)
                continue
        path_through_corridor.append(next_node)
    return path_through_corridor









































