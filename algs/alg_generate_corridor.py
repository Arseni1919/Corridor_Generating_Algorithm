import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_corridor_creation import SimEnvCC, get_random_corridor


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


class AlgAgentCC:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.path: List[Node] = []
        self.free_node: Node | None = None
        self.spanning_tree_dict: Dict[str, str | None] | None = None

    @property
    def name(self):
        return f'agent_{self.num}'


class ALgCC:
    def __init__(self, img_dir: str, **kwargs):
        self.img_dir = img_dir
        path_to_maps: str = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        path_to_heuristics: str = kwargs[
            'path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'

        # for the map
        self.map_dim = get_dims_from_pic(img_dir=self.img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.nodes_dict, self.map_dim,
                                                              img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)

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
        pass

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
        self._create_flow_roadmap()
        self._roll_agents()

    def _create_flow_roadmap(self):

        # get agents inside the corridor
        agents_in_corridor = [agent for agent in self.agents if agent.curr_node in self.corridor]

        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor}
        # plot_flow_in_env(ax[0], plot_info)
        # plt.show()
        # plt.close()

        # find k free spots
        free_nodes, free_nodes_dict = self._find_k_free_locations(agents_in_corridor)

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor,
                     'free_nodes': free_nodes}
        plot_flow_in_env(ax[0], plot_info)
        plt.show()
        plt.close()

        # complete search from all agents to all k locations
        self._roll_agents(agents_in_corridor, free_nodes, free_nodes_dict)

        plt.close()

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
        for cc_agent in agents_in_corridor:
            free_node = cc_agent.free_node
            spanning_tree_dict = cc_agent.spanning_tree_dict
            tube_to_corridor = get_tube_to_corridor(free_node, spanning_tree_dict, self.corridor, self.nodes_dict)

            # roll any agent through the tube
            pass
            print()


def main():
    set_seed(random_seed_bool=False, seed=123)
    N = 100
    # N = 700
    iterations = 100
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'

    # problem creation
    env = SimEnvCC(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    corridor = get_random_corridor(env)

    # alg creation + init
    alg = ALgCC(img_dir=img_dir)
    alg.initiate_problem(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])

    # for rendering
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.1
    total_unique_moves_list = []

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])
    for i_step in range(iterations):
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

    plt.show()
    print(f'finished run, metrics: {metrics}')


if __name__ == '__main__':
    main()



