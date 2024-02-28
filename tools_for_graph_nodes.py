import heapq

import numpy as np

from functions import *


class Node:
    def __init__(self, x: int, y: int, t: int = 0, neighbours: List[str] | None = None):
        self._x = x
        self._y = y
        self.t = t
        self.h = 0
        self.neighbours = [] if neighbours is None else neighbours
        self.parent = None
        self.g_dict = {}

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def xy(self):
        return self._x, self._y

    @property
    def g(self):
        return self.t

    @property
    def xyt_name(self):
        return f'{self.x}_{self.y}_{self.t}'

    @property
    def xy_name(self):
        return f'{self.x}_{self.y}'

    @property
    def f(self):
        # return self.t + self.h
        return self.t + self.h

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.xy_name < other.xy_name

    def __gt__(self, other):
        return self.xy_name > other.xy_name

    def reset(self, target_nodes: list | None = None, **kwargs):
        if 'start_time' in kwargs:
            self.t = kwargs['start_time']
        else:
            self.t = 0
        self.h = 0
        self.parent = None
        self.g_dict = {target_node.xy_name: 0 for target_node in target_nodes} if target_nodes else {}

    def get_pattern(self) -> dict:
        return {'x': self.x, 'y': self.y, 'neighbours': self.neighbours}


class ListNodes:
    def __init__(self, target_name=None):
        self.heap_list = []
        self.heap_names_list = []
        heapq.heapify(self.heap_names_list)
        # self.nodes_list = []
        self.dict = {}
        self.h_func_bool = False
        if target_name:
            self.h_func_bool = True
            self.target_name = target_name

    def __len__(self):
        return len(self.heap_list)

    def remove(self, node):
        if self.h_func_bool:
            self.heap_list.remove((node.g_dict[self.target_name], node.xy_name))
            self.heap_names_list.remove(node.xy_name)
            del self.dict[node.xy_name]
            return
        if node.xyt_name not in self.dict:
            raise RuntimeError('node.ID not in self.dict')
        self.heap_list.remove(((node.f, node.h), node.xyt_name))
        self.heap_names_list.remove(node.xyt_name)
        del self.dict[node.xyt_name]
        # self.nodes_list.remove(node)

    def add(self, node):
        if self.h_func_bool:
            heapq.heappush(self.heap_list, (node.g_dict[self.target_name], node.xy_name))
            heapq.heappush(self.heap_names_list, node.xy_name)
            self.dict[node.xy_name] = node
            return
        heapq.heappush(self.heap_list, ((node.f, node.h), node.xyt_name))
        heapq.heappush(self.heap_names_list, node.xyt_name)
        self.dict[node.xyt_name] = node
        # self.nodes_list.append(node)

    def pop(self):
        heap_tuple = heapq.heappop(self.heap_list)
        node = self.dict[heap_tuple[1]]
        if self.h_func_bool:
            del self.dict[node.xy_name]
            return node
        del self.dict[node.xyt_name]
        self.heap_names_list.remove(node.xyt_name)
        # self.nodes_list.remove(node)
        return node

    def get(self, xyt_name):
        return self.dict[xyt_name]

    def get_nodes_list(self):
        return [self.dict[item[1]] for item in self.heap_list]


def get_dims_from_pic(img_dir: str, path: str = 'maps') -> Tuple[int, int]:
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height = int(re.search(r'\d+', lines[1]).group())
        width = int(re.search(r'\d+', lines[2]).group())
    return height, width


def distance_nodes(node1, node2, h_func: dict = None):
    if h_func is None:
        # print('regular distance')
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    else:
        heuristic_dist = h_func[node1.x][node1.y][node2.x][node2.y]
        # direct_dist = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return heuristic_dist


def build_graph_nodes(img_dir: str, path: str = 'maps', show_map: bool = False) -> Tuple[List[Node], Dict[str, Node], np.ndarray]:
    # nodes, nodes_dict, img_np
    # print('Start build_graph_nodes...')
    img_np, (height, width) = get_np_from_dot_map(img_dir, path)
    return build_graph_from_np(img_np, show_map)


def get_np_from_dot_map(img_dir: str, path: str = 'maps') -> Tuple[np.ndarray, Tuple[int, int]]:
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height, width = get_dims_from_pic(img_dir, path)
        img_np = np.zeros((height, width))
        for height_index, line in enumerate(lines[4:]):
            for width_index, curr_str in enumerate(line):
                if curr_str == '.':
                    img_np[height_index, width_index] = 1
        return img_np, (height, width)


def build_graph_from_np(img_np: np.ndarray, show_map: bool = False) -> Tuple[List[Node], Dict[str, Node], np.ndarray]:
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    for node1, node2 in combinations(nodes, 2):
        if abs(node1.x - node2.x) > 1 or abs(node1.y - node2.y) > 1:
            continue
        if abs(node1.x - node2.x) == 1 and abs(node1.y - node2.y) == 1:
            continue
        node1.neighbours.append(node2.xy_name)
        node2.neighbours.append(node1.xy_name)
        # dist = distance_nodes(node1, node2)
        # if dist == 1:

    for curr_node in nodes:
        curr_node.neighbours.append(curr_node.xy_name)
        heapq.heapify(curr_node.neighbours)

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict, img_np


def get_edge_nodes(nodes_type, scen_name, nodes_dict, path='scens'):
    possible_dir = f"{path}/{scen_name}.json"
    nodes_to_return = []
    if os.path.exists(possible_dir):
        # Opening JSON file
        with open(possible_dir, 'r') as openfile:
            # Reading from json file
            scen = json.load(openfile)
            for pair in scen[nodes_type]:
                nodes_to_return.append(nodes_dict[f'{pair[0]}_{pair[1]}'])
    return nodes_to_return


def main():
    # img_dir = 'empty-32-32.map'  # 32-32
    img_dir = 'random-32-32-10.map'  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    # img_dir = 'random-32-32-20.map'  # 32-32
    # img_dir = 'room-32-32-4.map'  # 32-32
    # img_dir = 'maze-32-32-2.map'  # 32-32
    # img_dir = 'den312d.map'  # 65-81
    # img_dir = 'room-64-64-8.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'warehouse-10-20-10-2-2.map'  # 84-170
    # img_dir = 'warehouse-20-40-10-2-1.map'  # 123-321
    # img_dir = 'ht_chantry.map'  # 141-162
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'lak303d.map'  # 194-194
    # img_dir = 'warehouse-20-40-10-2-2.map'  # 164-340
    # img_dir = 'Berlin_1_256.map'  # 256-256
    # img_dir = 'den520d.map'  # 257-256
    # img_dir = 'ht_mansion_n.map'  # 270-133
    # img_dir = 'brc202d.map'  # 481-530
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path='maps', show_map=True)
    # start_nodes = get_edge_nodes(nodes_type='starts', scen_name='tree', nodes_dict=nodes_dict)
    # goal_nodes = get_edge_nodes(nodes_type='goals', scen_name='tree', nodes_dict=nodes_dict)
    print()


if __name__ == '__main__':
    main()
