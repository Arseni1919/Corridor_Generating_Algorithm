import heapq
import random

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

    def get_max_time(self, next_iteration: int, captured_agents: list) -> Tuple[int, dict]:
        freedom_times_dict = {n.xy_name: 0 for n in self.nodes}
        freedom_times_heap = list(freedom_times_dict.keys())
        heapq.heapify(freedom_times_heap)
        for cap_agent in captured_agents:
            for i, node in enumerate(cap_agent.path[next_iteration - 1:]):
                if node.xy_name in freedom_times_heap:
                    curr_v = freedom_times_dict[node.xy_name]
                    freedom_times_dict[node.xy_name] = max(curr_v, i)
        freedom_times_v = list(freedom_times_dict.values())
        max_time = max(freedom_times_v)
        # from_times = [len(t_agent.path[next_iteration:]) for t_agent in t_agents]
        # max_time = max(from_times)
        return max_time, freedom_times_dict

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
        freedom_times_dict = {n.xy_name: 0 for n in self.nodes}
        if len(t_agents) > 0:

            # find start locations of t_agents
            from_nodes = [t_agent.path[-1] for t_agent in t_agents]
            from_pattern = self.get_from_pattern(from_nodes)

            # all t_agents wait until the last of them will arrive to its start locations from movements of other tubes
            max_time, freedom_times_dict = self.get_max_time(next_iteration, captured_agents)
            # for t_agent in t_agents:
            #     while len(t_agent.path[next_iteration:]) < max_time:
            #         t_agent.path.append(t_agent.path[-1])
            #     assert len(t_agent.path[next_iteration:]) == max_time

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
        there_is_movement, i_time = True, 0
        while there_is_movement:
            there_is_movement = False
            i_time += 1
            step_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}
            pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(self.nodes)
            for to_t_node, from_t_node in pairwise_tube:
                if from_t_node.xy_name in step_dict:
                    curr_agent = step_dict[from_t_node.xy_name]
                    if len(curr_agent.path[next_iteration:]) >= i_time:
                        there_is_movement = True
                        continue
                    if curr_agent.name in agent_to_final_node_dict and agent_to_final_node_dict[curr_agent.name] == from_t_node:
                        # curr_agent.path.append(from_t_node)
                        continue
                    elif to_t_node.xy_name in step_dict:
                        curr_agent.path.append(from_t_node)
                        there_is_movement = True
                        continue
                    elif i_time <= freedom_times_dict[to_t_node.xy_name]:
                        curr_agent.path.append(from_t_node)
                        there_is_movement = True
                        continue
                    else:
                        curr_agent.path.append(to_t_node)
                        there_is_movement = True
                        step_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}
        for t_agent in t_agents:
            assert t_agent.path[-1] == agent_to_final_node_dict[t_agent.name]


def find_t_agents(tube: Tube, flex_agents) -> list:
    t_agents = []
    node_name_to_f_agent_dict = {f_agent.path[-1].xy_name: f_agent for f_agent in flex_agents}
    node_name_to_f_agent_heap = list(node_name_to_f_agent_dict.keys())
    for n in tube.nodes:
        if n.xy_name in node_name_to_f_agent_heap:
            t_agents.append(node_name_to_f_agent_dict[n.xy_name])
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
        # path: List[Node] = p_agent.path[next_iteration-1:]
        # path: List[Node] = p_agent.path[next_iteration:]
        # assert len(path) > 1
        for n in p_agent.path[next_iteration:]:
            new_map[n.x, n.y] = 0
    return new_map


def get_agents_in_corridor(corridor: List[Node], node_name_to_f_agent_dict, node_name_to_f_agent_heap) -> list:
    agents_in_corridor = []
    heapq.heapify(node_name_to_f_agent_heap)
    for n in corridor:
        if n.xy_name in node_name_to_f_agent_heap:
            agents_in_corridor.append(node_name_to_f_agent_dict[n.xy_name])
    return agents_in_corridor


def get_tube(
        c_agent,
        new_map: np.ndarray,
        tubes: List[Tube],
        corridor_for_c_agents: List[Node],
        nodes_dict: Dict[str, Node],
        node_name_to_f_agent_heap: list,
        to_assert: bool,
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
    :param node_name_to_f_agent_heap:
    :param to_assert:
    :return: solvable, Tube(nodes, free_node)
    """
    captured_free_nodes_heap: List[Node] = [tube.free_node.xy_name for tube in tubes]
    heapq.heapify(captured_free_nodes_heap)

    if to_assert:
        for cap_node in captured_free_nodes_heap:
            assert cap_node not in node_name_to_f_agent_heap

    spanning_tree_dict: Dict[str, str | None] = {c_agent.curr_node.xy_name: None}
    open_list: Deque[Node] = deque([c_agent.curr_node])
    closed_list_heap: List[str] = []
    heapq.heapify(closed_list_heap)
    small_iteration: int = 0
    while len(open_list) > 0:
        small_iteration += 1

        selected_node = open_list.pop()
        if selected_node not in corridor_for_c_agents and selected_node.xy_name not in node_name_to_f_agent_heap and selected_node.xy_name not in captured_free_nodes_heap:
            nodes, tube_pattern = get_full_tube(selected_node, spanning_tree_dict, nodes_dict, node_name_to_f_agent_heap)
            tube = Tube(nodes, selected_node, tube_pattern)
            return True, tube

        # corridor_nodes: List[Node] = []
        # outer_nodes: List[Node] = []
        nei_nodes: List[Node] = []
        for nei_name in selected_node.neighbours:
            if nei_name == selected_node.xy_name:
                continue
            nei_node = nodes_dict[nei_name]
            # 1 - free space, 0 - occupied space
            if new_map[nei_node.x, nei_node.y] == 0:
                continue
            if nei_node.xy_name in closed_list_heap:
                continue
            if nei_node in open_list:
                continue
            # connect nei_note to selected one
            spanning_tree_dict[nei_node.xy_name] = selected_node.xy_name
            nei_nodes.append(nei_node)
            # if nei_node in corridor_for_c_agents:
            #     corridor_nodes.append(nei_node)
            # else:
            #     outer_nodes.append(nei_node)
        random.shuffle(nei_nodes)
        open_list.extendleft(nei_nodes)
        # open_list.extendleft(outer_nodes)
        # open_list.extendleft(corridor_nodes)
        closed_list_heap.append(selected_node.xy_name)
    return False, None


def collapse_corridor(corridor: List[Node]):
    out_corridor: List[Node] = [corridor[0]]
    for from_n, to_n in pairwise_list(corridor):
        if to_n == from_n:
            continue
        assert to_n.xy_name in out_corridor[-1].neighbours
        out_corridor.append(to_n)

    return out_corridor


def calc_a_star_corridor(agent, nodes_dict: Dict[str, Node], h_dict, corridor_size: int, new_map: np.ndarray) -> List[Node] | None:
    """

    :param agent:
    :param nodes_dict:
    :param h_dict:
    :param corridor_size:
    :param new_map:
    :return: List of nodes where the first item is the agent's current location
    """
    next_goal_node = agent.next_goal_node
    curr_node = agent.curr_node
    goal_h_dict: np.ndarray = h_dict[next_goal_node.xy_name]
    initial_h = goal_h_dict[curr_node.x, curr_node.y]
    open_list = [((0 + initial_h, initial_h), curr_node)]
    heapq.heapify(open_list)
    open_names_list = [f'{curr_node.xy_name}_0']
    heapq.heapify(open_names_list)
    closed_list = []
    heapq.heapify(closed_list)
    closed_names_list = []
    heapq.heapify(closed_names_list)
    spanning_tree_dict: Dict[str, str | None] = {f'{curr_node.xy_name}_0': None}  # child: parent
    xyt_nodes_dict = {f'{curr_node.xy_name}_0': curr_node}

    iteration = 0
    i_t = 0
    i_node = curr_node
    while len(open_list) > 0:
        iteration += 1
        (i_f, i_h), i_node = heapq.heappop(open_list)
        i_t = int(i_f - i_h)
        open_names_list.remove(f'{i_node.xy_name}_{i_t}')
        # print()

        if i_node == next_goal_node or i_t >= corridor_size:
            # we have reached the end
            corridor: List[Node] = [i_node]
            j = i_t
            parent = spanning_tree_dict[f'{i_node.xy_name}_{j}']
            while parent is not None:
                parent_node = xyt_nodes_dict[parent]
                corridor.append(parent_node)
                parent = spanning_tree_dict[parent]
            corridor.reverse()
            corridor = collapse_corridor(corridor)
            return corridor

        node_current_neighbours = i_node.neighbours[:]
        random.shuffle(node_current_neighbours)
        for successor_xy_name in node_current_neighbours:
            new_t = i_t + 1
            successor_xyt_name = f'{successor_xy_name}_{new_t}'
            # if successor_xy_name == i_node.xy_name:
            #     continue
            if successor_xy_name in open_names_list:
                continue
            if successor_xy_name in closed_names_list:
                continue
            node_successor = nodes_dict[successor_xy_name]
            new_h = goal_h_dict[node_successor.x, node_successor.y]
            # 1 - free space, 0 - occupied space
            if new_map[node_successor.x, node_successor.y] == 0:
                continue
            # if new_h > initial_h + 1:
            #     continue

            spanning_tree_dict[successor_xyt_name] = f'{i_node.xy_name}_{i_t}'
            xyt_nodes_dict[f'{successor_xy_name}_{new_t}'] = node_successor
            heapq.heappush(open_list, ((new_t + new_h, new_h), node_successor))
            heapq.heappush(open_names_list, successor_xyt_name)

        heapq.heappush(closed_list, ((i_f, i_h), i_node))
        heapq.heappush(closed_names_list, f'{i_node.xy_name}_{i_t}')

    corridor: List[Node] = [i_node]
    j = i_t
    parent = spanning_tree_dict[f'{i_node.xy_name}_{j}']
    while parent is not None:
        parent_node = xyt_nodes_dict[parent]
        corridor.append(parent_node)
        parent = spanning_tree_dict[parent]
    corridor.reverse()
    corridor = collapse_corridor(corridor)
    return corridor


def calc_simple_corridor(agent, nodes_dict: Dict[str, Node], h_func, h_dict, corridor_size: int, new_map: np.ndarray) -> List[Node] | None:
    corridor: List[Node] = [agent.curr_node]
    goal_h_map: np.ndarray = h_dict[agent.next_goal_node.xy_name]
    goal_node: Node = agent.next_goal_node

    # def get_min_node(v_node, iterable_name):
    #     iterable_node = nodes_dict[iterable_name]
    #     if goal_h_map[iterable_node.x, iterable_node.y] < goal_h_map[v_node.x, v_node.y]:
    #         return iterable_node
    #     return v_node

    def get_min_value(min_v, iterable_name):
        iterable_node = nodes_dict[iterable_name]
        iterable_node_value = goal_h_map[iterable_node.x, iterable_node.y]
        if iterable_node_value < min_v:
            return iterable_node_value
        return min_v

    for i in range(corridor_size):
        next_node = corridor[-1]
        # node_name_to_h_value_dict = {
        #     node_name: h_func(agent.next_goal_node, nodes_dict[node_name])
        #     for node_name in next_node.neighbours
        # }
        # min_node_name = min(node_name_to_h_value_dict, key=node_name_to_h_value_dict.get)
        # min_node = nodes_dict[min_node_name]
        # min_node = reduce(get_min_node, next_node.neighbours, next_node)
        min_value: float = reduce(get_min_value, next_node.neighbours, goal_h_map[next_node.x, next_node.y])
        min_nodes_names: List[str] = list(filter(
            lambda n_name: goal_h_map[nodes_dict[n_name].x, nodes_dict[n_name].y] == min_value,
            next_node.neighbours))
        min_nodes: List[Node] = [nodes_dict[n_name] for n_name in min_nodes_names]
        min_nodes: List[Node] = list(filter(lambda n: new_map[n.x, n.y] != 0, min_nodes))
        if len(min_nodes) == 0:
            return corridor
        random.shuffle(min_nodes)
        min_node = min_nodes[0]
        corridor.append(min_node)
        if min_node == goal_node:
            return corridor
        # 1 - free space, 0 - occupied space
        # if new_map[min_node.x, min_node.y] == 0:
        #     return corridor
        # corridor.append(min_node)
    return corridor


def is_freedom_node(node: Node, nodes_dict: Dict[str, Node]) -> bool:
    assert len(node.neighbours) != 0
    assert len(node.neighbours) != 1
    if len(node.neighbours) == 2:
        return True

    prev_len = len(node.neighbours)
    init_nei_names = node.neighbours[:]
    init_nei_names.remove(node.xy_name)

    assert len(init_nei_names) < prev_len
    assert len(node.neighbours) == prev_len
    assert len(init_nei_names) != 0
    assert len(init_nei_names) > 1

    first_nei_name = init_nei_names[0]
    init_nei_names.remove(first_nei_name)
    first_nei = nodes_dict[first_nei_name]

    open_list: Deque[Node] = deque([first_nei])
    open_names_list_heap = [f'{first_nei.xy_name}']
    heapq.heapify(open_names_list_heap)
    closed_names_list_heap = [f'{node.xy_name}']
    heapq.heapify(closed_names_list_heap)

    iteration = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.pop()
        open_names_list_heap.remove(next_node.xy_name)
        if next_node.xy_name in init_nei_names:
            init_nei_names.remove(next_node.xy_name)
            if len(init_nei_names) == 0:
                return True
        for nei_name in next_node.neighbours:
            if nei_name == next_node.xy_name:
                continue
            if nei_name in closed_names_list_heap:
                continue
            if nei_name in open_names_list_heap:
                continue
            nei_node = nodes_dict[nei_name]

            open_list.appendleft(nei_node)
            heapq.heappush(open_names_list_heap, nei_name)
        heapq.heappush(closed_names_list_heap, next_node.xy_name)

    return False


def get_freedom_nodes_np(nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray) -> np.ndarray:
    freedom_nodes_np = np.zeros(img_np.shape)
    for node in nodes:
        if is_freedom_node(node, nodes_dict):
            freedom_nodes_np[node.x, node.y] = 1
    return freedom_nodes_np







































