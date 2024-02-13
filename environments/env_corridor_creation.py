import random

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt


class SimAgentCC:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.path: List[Node] = []
        self.unique_moves: List[Node] = []

    @property
    def name(self):
        return f'agent_{self.num}'


class SimEnvCC:
    def __init__(self, img_dir: str, **kwargs):
        self.img_dir = img_dir
        path_to_maps: str = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        path_to_heuristics: str = kwargs['path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'

        # for the map
        self.map_dim = get_dims_from_pic(img_dir=self.img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.nodes_dict, self.map_dim, img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)

        self.terminated: bool = False
        self.n_runs: int = 0
        self.iteration: int = 0
        self.agents: List[SimAgentCC] = []
        self.agents_dict: Dict[str, SimAgentCC] = {}
        self.start_nodes: List[Node] = []
        self.corridor: List[Node] = []

    @property
    def n_agents(self):
        return len(self.agents)

    def reset(self, start_node_names: List[str], corridor_names: List[str]) -> dict:
        self.start_nodes = [self.nodes_dict[snn] for snn in start_node_names]
        self.corridor = [self.nodes_dict[cn] for cn in corridor_names]
        self._check_solvability()
        self._create_agents()
        self.terminated = False
        self.n_runs += 1
        self.iteration = 1
        obs = self._get_obs()
        return obs

    def sample_actions(self) -> Dict[str, str]:
        actions = {}
        conf_v_list: List[Tuple[int, int]] = [agent.curr_node.xy for agent in self.agents]
        conf_e_list: List[Tuple[int, int, int, int]] = []
        for agent in self.agents:
            # pick a random next node
            next_node_name: str = random.choice(agent.curr_node.neighbours[:-1])
            next_node: Node = self.nodes_dict[next_node_name]
            # check if it not in conflict with previous agents
            in_conf_v = next_node.xy in conf_v_list
            in_conf_e = (agent.curr_node.x, agent.curr_node.y, next_node.x, next_node.y) in conf_e_list
            if not in_conf_v and not in_conf_e:
                actions[agent.name] = next_node.xy_name
                # insert new conflicts
                conf_v_list.append(next_node.xy)
                conf_e_list.append((next_node.x, next_node.y, agent.curr_node.x, agent.curr_node.y))
            else:
                actions[agent.name] = agent.curr_node.xy_name

        return actions

    def step(self, actions: Dict[str, str]) -> Tuple[dict, dict, bool, dict]:
        assert not self.terminated
        self._execute_actions(actions)
        self.iteration += 1
        obs = self._get_obs()
        metrics = self._get_metrics()
        self.terminated = self._check_termination()
        info = {}
        print(f'[ENV]: iteration: {self.iteration}')
        if self.terminated:
            print(f'[ENV]: finished.')
        return obs, metrics, self.terminated, info

    def _check_solvability(self):
        assert len(self.nodes) - len(self.start_nodes) >= len(self.corridor), 'UNSOLVABLE'

    def _create_agents(self) -> None:
        self.agents: List[SimAgentCC] = []
        self.agents_dict: Dict[str, SimAgentCC] = {}
        for i, start_node in enumerate(self.start_nodes):
            new_agent = SimAgentCC(num=i, start_node=start_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _execute_actions(self, actions: Dict[str, str]) -> None:
        for agent_name, next_node_name in actions.items():
            agent = self.agents_dict[agent_name]
            next_node = self.nodes_dict[next_node_name]
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            # extend path
            agent.path.append(agent.curr_node)
            if agent.prev_node.xy_name != agent.curr_node.xy_name:
                agent.unique_moves.append(agent.curr_node)
        # checks
        check_if_nei_pos(self.agents)
        check_if_vc(self.agents)
        check_if_ec(self.agents)

    def _get_obs(self) -> dict:
        obs = {agent.name: agent.curr_node.xy_name for agent in self.agents}
        obs['iteration'] = self.iteration
        return obs

    def _get_metrics(self) -> dict:
        total_unique_moves = sum([len(agent.unique_moves) for agent in self.agents])
        return {'total_unique_moves': total_unique_moves}

    def _check_termination(self) -> bool:
        # return False
        for agent in self.agents:
            if agent.curr_node in self.corridor:
                return False
        return True


def get_random_corridor(env: SimEnvCC) -> List[Node]:
    node_start = random.choice(env.nodes)
    node_goal = random.choice(env.nodes)
    v_constr_dict = {node.xy_name: [] for node in env.nodes}
    e_constr_dict = {node.xy_name: [] for node in env.nodes}
    perm_constr_dict = {node.xy_name: [] for node in env.nodes}
    result, info = a_star_xyt(start=node_start, goal=node_goal, nodes=env.nodes, h_func=env.h_func,
                              v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                              perm_constr_dict=perm_constr_dict,
                              plotter=None, middle_plot=False, nodes_dict=env.nodes_dict,
                              agent_name='agent_name', xyt_problem=False)
    return result


def main():
    N = 100
    iterations = 100
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'

    # problem creation
    env = SimEnvCC(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    corridor = get_random_corridor(env)

    # alg creation + init

    # for rendering
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.1
    total_unique_moves_list = []

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])
    for i_step in range(iterations):
        actions = env.sample_actions()  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # render
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
