from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


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


def main():
    pass


if __name__ == '__main__':
    main()
