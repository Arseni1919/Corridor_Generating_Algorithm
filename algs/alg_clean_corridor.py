import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from alg_gen_cor_v1 import get_full_tube, get_assign_agent_to_node_dict


def tube_is_free_to_go(tube: List[Node], inner_captured_nodes: list) -> bool:
    # tube: free node -> init node
    for n in tube:
        if n in inner_captured_nodes:
            return False
    return True


def get_tube(
        next_c_agent,
        l_agents_paths_dict: Dict[str, List[Node]],
        l_agents: list,
        corridor: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        occupied_nodes: List[Node],
) -> Tuple[bool, bool, List[Node] | None]:
    # solvable, free_to_roll, tube

    next_c_agent_path = l_agents_paths_dict[next_c_agent.name]

    inner_captured_nodes: List[Node] = []
    outer_captured_nodes: List[Node] = []
    for l_agent in l_agents:
        if l_agent == next_c_agent:
            continue
        l_agent_path = l_agents_paths_dict[l_agent.name]
        if l_agent_path[-1] in corridor:
            inner_captured_nodes.append(l_agent_path[-1])
        else:
            outer_captured_nodes.append(l_agent_path[-1])

    spanning_tree_dict: Dict[str, str | None] = {next_c_agent_path[-1].xy_name: None}
    open_list: Deque[Node] = deque([next_c_agent_path[-1]])
    closed_list: Deque[Node] = deque()
    small_iteration: int = 0
    while len(open_list) > 0:
        small_iteration += 1

        selected_node = open_list.pop()
        if selected_node not in corridor and selected_node not in outer_captured_nodes:
            tube = get_full_tube(selected_node, spanning_tree_dict, nodes_dict)
            if tube_is_free_to_go(tube, inner_captured_nodes):
                return True, True, tube
            return True, False, tube

        corridor_nodes: List[Node] = []
        outer_nodes: List[Node] = []
        for nei_name in selected_node.neighbours:
            if nei_name == selected_node.xy_name:
                continue
            nei_node = nodes_dict[nei_name]
            if nei_node in occupied_nodes:
                continue
            if nei_node in closed_list:
                continue
            if nei_node in open_list:
                continue
            # connect nei_note to selected one
            spanning_tree_dict[nei_node.xy_name] = selected_node.xy_name
            if nei_node in corridor:
                corridor_nodes.append(nei_node)
            else:
                outer_nodes.append(nei_node)
        open_list.extendleft(outer_nodes)
        open_list.extendleft(corridor_nodes)
        closed_list.append(selected_node)
    return False, False, None


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









































