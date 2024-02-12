import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_corridor_creation import SimEnvCC, get_random_corridor


def get_agents_in_corridor(agents: list, corridor: list) -> deque:
    # get agents inside the corridor
    init_agents_in_corridor = [agent for agent in agents if agent.curr_node in corridor]
    nodes_to_agents_dict = {agent.curr_node.xy_name: agent for agent in init_agents_in_corridor}
    agents_in_corridor: Deque[AlgAgentCC] = deque()
    for n in corridor:
        if n.xy_name in nodes_to_agents_dict:
            agents_in_corridor.append(nodes_to_agents_dict[n.xy_name])
    assert len(agents_in_corridor) == len(init_agents_in_corridor)
    return agents_in_corridor


def tube_is_full(tube: List[Node], prev_config: OrderedDict) -> bool:
    for n in tube[:-1]:
        if n.xy_name not in prev_config:
            return False
    return True


def find_closest_hanging_agent(from_t_node: Node, corridor: List[Node], prev_config: OrderedDict, next_config: OrderedDict, nodes_dict: Dict[str, Node]) -> Tuple[Any, Node]:
    next_config_agents = list(next_config.values())
    spanning_tree_dict: Dict[str, str | None] = {from_t_node.xy_name: None}
    open_list: Deque[Node] = deque([from_t_node])
    closed_list: Deque[Node] = deque([])
    small_iteration: int = 0
    while len(open_list) > 0:
        small_iteration += 1

        selected_node = open_list.pop()
        if selected_node.xy_name in prev_config:
            curr_agent = prev_config[selected_node.xy_name]
            if curr_agent in next_config_agents:
                continue
            sub_to_node = nodes_dict[spanning_tree_dict[selected_node.xy_name]]
            return curr_agent, sub_to_node

        for nei_name in selected_node.neighbours:
            if nei_name == selected_node.xy_name:
                continue
            nei_node = nodes_dict[nei_name]
            if nei_node in closed_list:
                continue
            if nei_node in open_list:
                continue
            if nei_node not in corridor:
                continue
            # connect nei_note to selected one
            spanning_tree_dict[nei_node.xy_name] = selected_node.xy_name
            open_list.appendleft(nei_node)
        closed_list.append(selected_node)

    raise RuntimeError('ashipka')


def get_tube_to_corridor(free_node: Node, spanning_tree_dict: Dict[str, str], corridor: List[Node], nodes_dict: Dict[str, Node]) -> List[Node]:
    tube_to_corridor: List[Node] = [free_node]
    parent = spanning_tree_dict[free_node.xy_name]
    while parent is not None:
        parent_node = nodes_dict[parent]
        tube_to_corridor.append(parent_node)
        if parent_node in corridor:
            break
        parent = spanning_tree_dict[parent]
    return tube_to_corridor


def get_tubes_to_corridor(agents_in_corridor: list, corridor: List[Node], nodes_dict: Dict[str, Node]) -> list:
    # create tubes
    tubes_to_corridor = []
    for cc_agent in agents_in_corridor:
        free_node = cc_agent.free_node
        spanning_tree_dict = cc_agent.spanning_tree_dict
        tube_to_corridor = get_tube_to_corridor(free_node, spanning_tree_dict, corridor, nodes_dict)
        tubes_to_corridor.append(tube_to_corridor)
    return tubes_to_corridor


def get_static_agents(tubes_to_corridor: List[list], corridor: list, agents: list) -> list:
    static_agents = []
    all_nodes_of_tubes = list(itertools.chain.from_iterable(tubes_to_corridor))
    for agent in agents:
        if agent.curr_node in corridor or agent.curr_node in all_nodes_of_tubes:
            continue
        static_agents.append(agent)
    return static_agents


class AlgAgentCC:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.path: List[Node] = [start_node]
        self.free_node: Node | None = None
        self.spanning_tree_dict: Dict[str, str | None] | None = None

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    def __eq__(self, other):
        return self.num == other.num


class ALgCC:
    def __init__(self, img_dir: str, env: SimEnvCC, **kwargs):
        self.img_dir = img_dir
        self.env = env
        path_to_maps: str = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        path_to_heuristics: str = kwargs[
            'path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'

        # for the map
        self.map_dim = get_dims_from_pic(img_dir=self.img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        self.h_func = self.env.h_func

        self.agents: List[AlgAgentCC] = []
        self.agents_dict: Dict[str, AlgAgentCC] = {}
        self.start_nodes: List[Node] = []
        self.corridor: List[Node] = []

    def initiate_problem(self, start_node_names: List[str], corridor_names: List[str]) -> None:
        self.start_nodes = [self.nodes_dict[snn] for snn in start_node_names]
        self.corridor = [self.nodes_dict[cn] for cn in corridor_names]
        self._create_agents()
        self._solve()

    def get_actions(self, obs: dict) -> Dict[str, str]:
        iteration = obs['iteration']
        actions = {
            agent.name: agent.path[iteration].xy_name for agent in self.agents
        }
        return actions

    def _create_agents(self) -> None:
        self.agents: List[AlgAgentCC] = []
        self.agents_dict: Dict[str, AlgAgentCC] = {}
        for i, start_node in enumerate(self.start_nodes):
            new_agent = AlgAgentCC(num=i, start_node=start_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _solve(self) -> None:
        """
        - create flow to find k empty locations
        - roll the agents upon the flow
        """
        agents_in_corridor, free_nodes, free_nodes_dict = self._create_flow_roadmap()
        self._roll_agents(agents_in_corridor, free_nodes, free_nodes_dict)

    def _create_flow_roadmap(self) -> Tuple[List[AlgAgentCC], List[Node], Dict[str, Node]]:

        # get agents inside the corridor
        agents_in_corridor = get_agents_in_corridor(self.agents, self.corridor)

        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor}
        # plot_flow_in_env(ax[0], plot_info)
        # plt.show()
        # plt.close()

        # find k free spots
        free_nodes, free_nodes_dict = self._find_k_free_locations(agents_in_corridor)

        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor,
        #              'free_nodes': free_nodes}
        # plot_flow_in_env(ax[0], plot_info)
        # plt.show()
        # plt.close()

        return agents_in_corridor, free_nodes, free_nodes_dict

    def _find_k_free_locations(self, agents_in_corridor: List[AlgAgentCC]) -> Tuple[List[Node], Dict[str, Node]]:
        k: int = len(agents_in_corridor)
        counter: int = 0
        free_nodes: List[Node] = []
        free_nodes_dict: Dict[str, Node] = {}
        others_locs = [agent.curr_node for agent in self.agents if agent not in agents_in_corridor]

        for cc_agent in agents_in_corridor:
            spanning_tree_dict: Dict[str, str | None] = {cc_agent.curr_node.xy_name: None}
            open_list: Deque[Node] = deque([cc_agent.curr_node])
            closed_list: Deque[Node] = deque([])
            small_iteration: int = 0
            while len(open_list) > 0:
                small_iteration += 1

                selected_node = open_list.pop()
                if selected_node not in self.corridor and selected_node not in others_locs and selected_node not in free_nodes:
                    counter += 1
                    free_nodes.append(selected_node)
                    free_nodes_dict[selected_node.xy_name] = selected_node
                    cc_agent.free_node = selected_node
                    cc_agent.spanning_tree_dict = spanning_tree_dict
                    break

                corridor_nodes: List[Node] = []
                outer_nodes: List[Node] = []
                for nei_name in selected_node.neighbours:
                    if nei_name == selected_node.xy_name:
                        continue
                    nei_node = self.nodes_dict[nei_name]
                    if nei_node in closed_list:
                        continue
                    if nei_node in open_list:
                        continue
                    # connect nei_note to selected one
                    spanning_tree_dict[nei_node.xy_name] = selected_node.xy_name
                    if nei_node in self.corridor:
                        corridor_nodes.append(nei_node)
                    else:
                        outer_nodes.append(nei_node)
                open_list.extendleft(outer_nodes)
                open_list.extendleft(corridor_nodes)
                closed_list.append(selected_node)

        assert counter == k
        return free_nodes, free_nodes_dict

    def _roll_agents(self, agents_in_corridor: List[AlgAgentCC], free_nodes: List[Node], free_nodes_dict: Dict[str, Node]):

        # create tubes
        tubes_to_corridor = get_tubes_to_corridor(agents_in_corridor, self.corridor, self.nodes_dict)
        still_in_corridor: List[AlgAgentCC] = agents_in_corridor[:]

        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor,
        #              'free_nodes': free_nodes, 'tubes_to_corridor': tubes_to_corridor}
        # plot_flow_in_env(ax[0], plot_info)
        # plt.show()
        # plt.close()

        static_agents = get_static_agents(tubes_to_corridor, self.corridor, self.agents)
        moving_agents: List[AlgAgentCC] = [agent for agent in self.agents if agent not in static_agents]

        # roll any agent through the tube
        path_time: int = 0

        corridor_is_empty: bool = False
        prev_config: OrderedDict[str, AlgAgentCC] = OrderedDict([(agent.curr_node.xy_name, agent) for agent in self.agents])
        next_config: OrderedDict[str, AlgAgentCC] = OrderedDict([(agent.curr_node.xy_name, agent) for agent in static_agents])

        tube_index: int = 0
        while not corridor_is_empty:

            tube = tubes_to_corridor[tube_index]
            if tube_is_full(tube, prev_config):
                tube_index += 1

            # at the edges (free nodes) - agents cannot move forward
            if tube[0].xy_name in prev_config:
                p_agent = prev_config[tube[0].xy_name]
                next_config[tube[0].xy_name] = p_agent

            # tube: [free node ---> node inside the corridor]
            pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(tube)
            for to_t_node, from_t_node in pairwise_tube:
                if from_t_node.xy_name in prev_config:
                    curr_agent: AlgAgentCC = prev_config[from_t_node.xy_name]
                    if to_t_node.xy_name in next_config:
                        next_config[from_t_node.xy_name] = curr_agent
                    else:
                        next_config[to_t_node.xy_name] = curr_agent
                else:
                    if from_t_node in self.corridor:
                        curr_agent, sub_to_node = find_closest_hanging_agent(from_t_node, self.corridor, prev_config, next_config, self.nodes_dict)
                        if sub_to_node.xy_name in next_config:
                            next_config[from_t_node.xy_name] = curr_agent
                        else:
                            next_config[sub_to_node.xy_name] = curr_agent
            next_config_agents = list(next_config.values())
            for m_agent in moving_agents:
                # if not moving_agents_dict[m_agent.name]:
                if m_agent not in next_config_agents:
                    to_node = m_agent.path[-1]
                    assert to_node.xy_name not in next_config
                    next_config[to_node.xy_name] = m_agent

            assert len(next_config) == len(self.agents)
            for next_node_name, agent in next_config.items():
                next_node = self.nodes_dict[next_node_name]
                agent.path.append(next_node)

            # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor,
            #              'free_nodes': free_nodes, 'tubes_to_corridor': tubes_to_corridor, 'tube': tube}
            # plot_flow_in_env(ax[0], plot_info)
            # plt.show()
            # plt.close()

            prev_config = next_config
            next_config: OrderedDict[str, AlgAgentCC] = OrderedDict([(agent.curr_node.xy_name, agent) for agent in static_agents])

            still_in_corridor = [agent for agent in agents_in_corridor if agent.path[-1] in self.corridor]
            corridor_is_empty = len(still_in_corridor) == 0
            path_time += 1
            print(f'{path_time=} | still_in_corridor: {len(still_in_corridor)}')

        print()


def main():
    # set_seed(random_seed_bool=False, seed=973)
    set_seed(random_seed_bool=True)
    # N = 80
    N = 100
    # N = 600
    # N = 700
    # N = 750
    iterations = 100
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    img_dir = 'maze-32-32-2.map'

    # problem creation
    env = SimEnvCC(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    corridor = get_random_corridor(env)

    # alg creation + init
    alg = ALgCC(img_dir=img_dir, env=env)
    alg.initiate_problem(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])

    # for rendering
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.1
    total_unique_moves_list = []

    # the run
    i_step = 0
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])
    while True:
        i_step += 1
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        total_unique_moves_list.append(metrics['total_unique_moves'])
        plot_info = {
            'i': i_step, 'iterations': iterations, 'img_dir': img_dir, 'img_np': env.img_np,
            'n_agents': env.n_agents, 'agents': env.agents, 'corridor': corridor,
            'total_unique_moves_list': total_unique_moves_list,
        }
        plot_step_in_env(ax[0], plot_info)
        plot_unique_movements(ax[1], plot_info)
        plt.pause(plot_rate)

        if terminated:
            break

    plot_info = {
        'i': iterations, 'iterations': iterations, 'img_dir': img_dir, 'img_np': env.img_np,
        'n_agents': env.n_agents, 'agents': env.agents, 'corridor': corridor,
        'total_unique_moves_list': total_unique_moves_list,
    }
    plot_step_in_env(ax[0], plot_info)
    plot_unique_movements(ax[1], plot_info)
    plt.pause(plot_rate)
    plt.show()
    print(f'finished run, metrics: {metrics}')


if __name__ == '__main__':
    main()



