import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


class HeapList:
    def __init__(self, input_list: list = None, input_names_list: list = None):
        self.inner_list: list = input_list if input_list else []
        heapq.heapify(self.inner_list)
        self.inner_names_list: list = input_names_list if input_names_list else []
        heapq.heapify(self.inner_names_list)

    def add(self, i_t: int, i_h: int, i_node: Node):
        i_f = int(i_t + i_h)
        heapq.heappush(self.inner_list, ((i_f, i_h), i_node))
        heapq.heappush(self.inner_names_list, f'{i_node.xy_name}_{i_t}')

    def pop(self):
        (i_f, i_h), i_node = heapq.heappop(self.inner_list)
        i_t = int(i_f - i_h)
        self.inner_names_list.remove(f'{i_node.xy_name}_{i_t}')
        return i_t, i_h, i_f, i_node

    def __contains__(self, item: str):
        return item in self.inner_names_list

    def __len__(self):
        return len(self.inner_list)

    def __iter__(self):
        return iter(self.inner_list)


def reconstruct_path(i_node, i_t, spanning_tree_dict, xyt_nodes_dict):
    # we have reached the end
    path: List[Node] = [i_node]
    parent = spanning_tree_dict[f'{i_node.xy_name}_{i_t}']
    while parent is not None:
        parent_node = xyt_nodes_dict[parent]
        path.append(parent_node)
        parent = spanning_tree_dict[parent]
    path.reverse()
    return path


def get_latest_vc_on_node(i_node: Node, vc_np: np.ndarray | None) -> int:
    """
    :param i_node:
    :param vc_np: vertex constraints [x, y, t] = bool
    :return: int
    """
    if vc_np is None:
        return 0
    vc_times = vc_np[i_node.x, i_node.y, :]
    indices = np.argwhere(vc_times == 1)
    if len(indices) == 0:
        return 0
    return indices[-1][0]


def create_constraints(paths: List[List[Node]], map_dim: Tuple[int, int]) -> Tuple[
    np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    vc_np: vertex constraints [x, y, t] = bool
    ec_np: edge constraints [x, y, x, y, t] = bool
    pc_np: permanent constraints [x, y] = int or -1
    """
    if len(paths) == 0:
        return None, None, None
    max_path_len = max(map(lambda x: len(x), paths))
    if max_path_len == 0:
        return None, None, None
    vc_np = np.zeros((map_dim[0], map_dim[1], max_path_len))
    ec_np = np.zeros((map_dim[0], map_dim[1], map_dim[0], map_dim[1], max_path_len))
    pc_np = np.ones((map_dim[0], map_dim[1])) * -1
    for path in paths:
        # pc
        last_node = path[-1]
        last_time = len(path) - 1
        pc_np[last_node.x, last_node.y] = max(pc_np[last_node.x, last_node.y], last_time)
        prev_n = path[0]
        for t, n in enumerate(path):
            # vc
            vc_np[n.x, n.y, t] = 1
            # ec
            ec_np[prev_n.x, prev_n.y, n.x, n.y, t] = 1
            prev_n = n
    return vc_np, ec_np, pc_np


def get_node_successor(i_node: Node, successor_xy_name: str, new_t: int, nodes_dict: Dict[str, Node],
                       vc_np: np.ndarray | None, ec_np: np.ndarray | None, pc_np: np.ndarray | None) -> Node | None:
    """
    :param i_node:
    :param successor_xy_name:
    :param new_t:
    :param nodes_dict:
    :param vc_np: vertex constraints [x, y, t] = bool
    :param ec_np: edge constraints [x, y, x, y, t] = bool
    :param pc_np: permanent constraints [x, y] = int or -1
    :return:
    """
    if successor_xy_name not in nodes_dict:
        return None
    node_successor = nodes_dict[successor_xy_name]

    if vc_np is not None:
        if new_t < vc_np.shape[-1] and vc_np[node_successor.x, node_successor.y, new_t]:
            return None

    if ec_np is not None:
        if new_t < ec_np.shape[-1] and ec_np[node_successor.x, node_successor.y, i_node.x, i_node.y, new_t]:
            return None

    if pc_np is not None:
        pc_value = pc_np[node_successor.x, node_successor.y]
        if pc_value != -1 and new_t >= pc_value:
            return None

    return node_successor


def calc_temporal_a_star(curr_node: Node, goal_node: Node, nodes_dict: Dict[str, Node], h_dict, max_len: int,
                         vc_np: np.ndarray | None = None, ec_np: np.ndarray | None = None, pc_np: np.ndarray | None = None) -> Tuple[List[
    Node], dict] | None:
    """
    :param curr_node:
    :param goal_node:
    :param nodes_dict:
    :param h_dict:
    :param max_len:
    :param vc_np: vertex constraints [x, y, t] = bool
    :param ec_np: edge constraints [x, y, x, y, t] = bool
    :param pc_np: permanent constraints [x, y] = int
    :return: List of nodes where the first item is the agent's current location
    """
    start_time = time.time()
    goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
    initial_h = int(goal_h_dict[curr_node.x, curr_node.y])
    open_list = HeapList()
    open_list.add(i_t=0, i_h=initial_h, i_node=curr_node)
    closed_list = HeapList()
    spanning_tree_dict: Dict[str, str | None] = {f'{curr_node.xy_name}_0': None}  # child: parent
    xyt_nodes_dict = {f'{curr_node.xy_name}_0': curr_node}

    iteration = 0
    i_t = 0
    i_node = curr_node
    while len(open_list) > 0:
        iteration += 1
        print(f'\ropen: {len(open_list)}, closed: {len(closed_list)}', end='')
        i_t, i_h, i_f, i_node = open_list.pop()
        # i_node_name = i_node.xy_name
        if i_node == goal_node or i_t >= max_len:
            # if there is a future constraint on a goal
            latest_vc_on_node: int = get_latest_vc_on_node(i_node, vc_np)
            if i_t > latest_vc_on_node or i_t >= max_len:
                path = reconstruct_path(i_node, i_t, spanning_tree_dict, xyt_nodes_dict)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list}

        node_current_neighbours = i_node.neighbours[:]
        random.shuffle(node_current_neighbours)
        for successor_xy_name in node_current_neighbours:
            new_t = i_t + 1
            successor_xyt_name = f'{successor_xy_name}_{new_t}'
            # if successor_xy_name == i_node.xy_name:
            #     continue
            if successor_xyt_name in open_list:
                continue
            if successor_xyt_name in closed_list:
                continue

            node_successor = get_node_successor(i_node, successor_xy_name, new_t, nodes_dict, vc_np, ec_np, pc_np)
            if node_successor is None:
                continue
            # node_successor = nodes_dict[successor_xy_name]
            new_h = int(goal_h_dict[node_successor.x, node_successor.y])
            # if new_h > initial_h + 1:
            #     continue

            spanning_tree_dict[successor_xyt_name] = f'{i_node.xy_name}_{i_t}'
            xyt_nodes_dict[f'{successor_xy_name}_{new_t}'] = node_successor
            open_list.add(i_t=new_t, i_h=new_h, i_node=node_successor)

        closed_list.add(i_t=i_t, i_h=i_h, i_node=i_node)

    path = reconstruct_path(i_node, i_t, spanning_tree_dict, xyt_nodes_dict)
    runtime = time.time() - start_time
    return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list}


def calc_fastest_escape(curr_node: Node, goal_node: Node, nodes_dict: Dict[str, Node], h_dict,
                        vc_np: np.ndarray | None = None, ec_np: np.ndarray | None = None, pc_np: np.ndarray | None = None) -> Tuple[List[
    Node] | None, dict]:
    """
    :param curr_node:
    :param goal_node:
    :param nodes_dict:
    :param h_dict:
    :param vc_np: vertex constraints [x, y, t] = bool
    :param ec_np: edge constraints [x, y, x, y, t] = bool
    :param pc_np: permanent constraints [x, y] = int
    :return: List of nodes where the first item is the agent's current location
    """
    start_time = time.time()
    goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
    initial_h = int(goal_h_dict[curr_node.x, curr_node.y])
    open_list = HeapList()
    open_list.add(i_t=0, i_h=initial_h, i_node=curr_node)
    closed_list = HeapList()
    spanning_tree_dict: Dict[str, str | None] = {f'{curr_node.xy_name}_0': None}  # child: parent
    xyt_nodes_dict = {f'{curr_node.xy_name}_0': curr_node}

    iteration = 0
    i_t = 0
    i_node = curr_node
    while len(open_list) > 0:
        iteration += 1
        print(f'\ropen: {len(open_list)}, closed: {len(closed_list)}', end='')
        i_t, i_h, i_f, i_node = open_list.pop()
        i_node_name = i_node.xy_name
        if i_t > 0:
            # if there is a future constraint on a goal
            latest_vc_on_node: int = get_latest_vc_on_node(i_node, vc_np)
            if i_t > latest_vc_on_node:
                path = reconstruct_path(i_node, i_t, spanning_tree_dict, xyt_nodes_dict)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list}

        node_current_neighbours = i_node.neighbours[:]
        random.shuffle(node_current_neighbours)
        for successor_xy_name in node_current_neighbours:
            new_t = i_t + 1
            successor_xyt_name = f'{successor_xy_name}_{new_t}'
            # if successor_xy_name == i_node.xy_name:
            #     continue
            if successor_xyt_name in open_list:
                continue
            if successor_xyt_name in closed_list:
                continue

            node_successor = get_node_successor(i_node, successor_xy_name, new_t, nodes_dict, vc_np, ec_np, pc_np)
            if node_successor is None:
                continue
            # node_successor = nodes_dict[successor_xy_name]
            new_h = int(goal_h_dict[node_successor.x, node_successor.y])
            # if new_h > initial_h + 1:
            #     continue

            spanning_tree_dict[successor_xyt_name] = f'{i_node.xy_name}_{i_t}'
            xyt_nodes_dict[f'{successor_xy_name}_{new_t}'] = node_successor
            open_list.add(i_t=new_t, i_h=new_h, i_node=node_successor)

        closed_list.add(i_t=i_t, i_h=i_h, i_node=i_node)

    # path = reconstruct_path(i_node, i_t, spanning_tree_dict, xyt_nodes_dict)
    runtime = time.time() - start_time
    return None, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list}


def main():
    # set_seed(random_seed_bool=False, seed=7310)
    # set_seed(random_seed_bool=False, seed=123)
    set_seed(random_seed_bool=True)
    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    img_dir = 'maze-32-32-4.map'

    to_render: bool = True
    # to_render: bool = False

    # for the map
    path_to_maps = '../maps'
    path_to_heuristics = '../logs_for_heuristics'
    map_dim = get_dims_from_pic(img_dir=img_dir, path=path_to_maps)
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
    h_dict = parallel_build_heuristic_for_entire_map(nodes, nodes_dict, map_dim, img_dir=img_dir,
                                                     path=path_to_heuristics)

    # ------------------------- #
    node_start = random.choice(nodes)
    node_goal = random.choice(nodes)
    print(f'start: {node_start.x}, {node_start.y} -> goal: {node_goal.x}, {node_goal.y}')
    # ------------------------- #

    # vc_np = np.zeros((map_dim[0], map_dim[1], 40))
    # vc_np[8, 28, 35] = 1
    # vc_np[7, 1, 1] = 1
    # ec_np = np.zeros((map_dim[0], map_dim[1], map_dim[0], map_dim[1], 40))
    # ec_np[7, 1, 6, 1, 3] = 1
    # pc_np = np.ones((map_dim[0], map_dim[1])) * -1
    # pc_np[8, 2] = 1
    path = [
        nodes_dict['7_1'],
        nodes_dict['7_1'],
        nodes_dict['6_1'],
        nodes_dict['5_1'],
        nodes_dict['4_1'],
    ]
    paths = [path]
    vc_np, ec_np, pc_np = create_constraints(paths, map_dim)

    # path, info = calc_temporal_a_star(curr_node=node_start, goal_node=node_goal, nodes_dict=nodes_dict,
    #                                   h_dict=h_dict, max_len=1000, vc_np=vc_np, ec_np=ec_np, pc_np=pc_np)

    path, info = calc_fastest_escape(curr_node=node_start, goal_node=node_goal, nodes_dict=nodes_dict,
                                     h_dict=h_dict, vc_np=vc_np, ec_np=ec_np, pc_np=pc_np)

    if path:
        print(f'\nruntime: {info['runtime']: .2f}s.')
        print('The result is:', *[node.xy_name for node in path], sep='->')
        print('The result is:', *[i for i in range(len((path)))], sep='->')

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_info = {'path': path, 'img_np': img_np, **info}
        plot_temp_a_star(ax[0], plot_info)
        plt.show()


if __name__ == '__main__':
    main()
