import matplotlib.pyplot as plt
from collections import deque

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from alg_gen_cor_v1 import get_full_tube, get_assign_agent_to_node_dict


class Tube:
    def __init__(self, nodes: List[Node], free_node: Node):
        self.nodes: List[Node] = nodes
        self.free_node: Node = free_node
        assert free_node == nodes[0]

    def move(self, t_agents: list) -> None:
        pass

    def __len__(self):
        return len(self.nodes)


def find_t_agents(tube: Tube, flex_agents: list) -> list:
    t_agents = []
    return t_agents

def move_main_agent(agent, corridor: List[Node], captured_agents: list):
    pass


def create_new_map(img_np: np.ndarray, planned_agents: list, next_iteration: int) -> np.ndarray:
    # 1 - free space, 0 - occupied space
    new_map = np.copy(img_np)
    for agent in planned_agents:
        path: List[Node] = agent.path[next_iteration-1:]
        assert len(path) > 1
        for n in path:
            new_map[n.x, n.y] = 0
    return new_map


def calc_simple_corridor(agent, nodes_dict: Dict[str, Node], h_func, corridor_size: int, new_map: np.ndarray) -> List[Node] | None:
    corridor: List[Node] = [agent.curr_node]
    for i in range(corridor_size - 1):
        next_node = corridor[-1]
        node_name_to_h_value_dict = {
            node_name: h_func(agent.next_goal_node, nodes_dict[node_name])
            for node_name in next_node.neighbours
        }
        min_node_name = min(node_name_to_h_value_dict, key=node_name_to_h_value_dict.get)
        min_node = nodes_dict[min_node_name]
        # 1 - free space, 0 - occupied space
        if not new_map[min_node.x, min_node.y]:
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
            nodes = get_full_tube(selected_node, spanning_tree_dict, nodes_dict)
            tube = Tube(nodes, selected_node)
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









































