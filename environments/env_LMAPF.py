import random

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt


class SimAgentLMAPF:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_goal_node: Node | None = None
        self.path: List[Node] = []
        self.unique_moves: List[Node] = []
        self.finished_goals: List[Node] = []
        self.arrived: bool = False

    @property
    def name(self):
        return f'agent_{self.num}'


class SimEnvLMAPF:
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

        self.n_runs: int = 0
        self.iteration: int = 0
        self.agents: List[SimAgentLMAPF] = []
        self.agents_dict: Dict[str, SimAgentLMAPF] = {}
        self.start_nodes: List[Node] = []
        self.max_time: int | None = None
        self.corridor_size: int = 1

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def _if_terminated(self) -> bool:
        return self.iteration > self.max_time

    @property
    def start_nodes_names(self):
        return [n.xy_name for n in self.start_nodes]

    @property
    def agents_names(self):
        return [a.name for a in self.agents]

    def reset(self, start_node_names: List[str], max_time: int, corridor_size: int) -> Dict[str, Any]:
        self.start_nodes = [self.nodes_dict[snn] for snn in start_node_names]
        self.max_time = max_time
        self.corridor_size = corridor_size
        self._check_solvability()
        self._create_agents()
        # set first goals
        self._update_goals()
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

    def step(self, actions: Dict[str, str]) -> Tuple[Dict[str, Any], dict, bool, dict]:
        assert not self._if_terminated
        self._execute_actions(actions)
        self._update_goals()
        self.iteration += 1
        obs = self._get_obs()
        metrics = self._get_metrics()
        info = {}
        print(f'[ENV]: iteration: {self.iteration}')
        if self._if_terminated:
            print(f'[ENV]: finished.')
        return obs, metrics, self._if_terminated, info

    def assign_next_goal(self, curr_agent: SimAgentLMAPF) -> None:
        occupied_nodes_odict: OrderedDict[str, Node] = OrderedDict()
        for o_agent in self.agents:
            if o_agent == curr_agent:
                continue
            if o_agent.next_goal_node:
                occupied_nodes_odict[o_agent.next_goal_node.xy_name] = o_agent.next_goal_node

        possible_nodes: List[Node] = [n for n in self.nodes if n.xy_name not in occupied_nodes_odict]
        next_goal_node: Node = random.choice(possible_nodes)
        curr_agent.next_goal_node = next_goal_node

        # if curr_agent.num != 0:
        #     curr_agent.next_goal_node = curr_agent.start_node

    def _update_goals(self):
        for agent in self.agents:
            if agent.next_goal_node is None:
                self.assign_next_goal(agent)
                continue
            agent.arrived = agent.curr_node == agent.next_goal_node
            if agent.arrived:
                agent.finished_goals.append(agent.next_goal_node)
                self.assign_next_goal(agent)

    def _check_solvability(self):
        assert len(self.nodes) - len(self.start_nodes) >= self.corridor_size

    def _create_agents(self) -> None:
        self.agents: List[SimAgentLMAPF] = []
        self.agents_dict: Dict[str, SimAgentLMAPF] = {}
        for i, start_node in enumerate(self.start_nodes):
            new_agent = SimAgentLMAPF(num=i, start_node=start_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _execute_actions(self, actions: Dict[str, str]) -> None:
        for agent in self.agents:
        # for agent_name, next_node_name in actions.items():
            next_node_name = actions[agent.name]
            next_node = self.nodes_dict[next_node_name]
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            # extend path
            agent.path.append(agent.curr_node)
            if agent.prev_node.xy_name != agent.curr_node.xy_name:
                agent.unique_moves.append(agent.curr_node)
        # checks
        check_vc_ec_neic(self.agents)
        # check_if_nei_pos(self.agents) check_if_vc(self.agents) check_if_ec(self.agents)

    def _get_obs(self) -> dict:
        obs = {agent.name:
                   AgentTuple(**{
                       'num': agent.num,
                       'start_node_name': agent.start_node.xy_name,
                       'curr_node_name': agent.curr_node.xy_name,
                       'next_goal_node_name': agent.next_goal_node.xy_name,
                       'arrived': agent.arrived,
                   })
               for agent in self.agents
               }
        obs['iteration'] = self.iteration
        obs['start_nodes_names'] = self.start_nodes_names
        obs['agents_names'] = self.agents_names
        return obs

    def _get_metrics(self) -> dict:
        total_unique_moves: int = 0
        total_finished_goals: int = 0
        for agent in self.agents:
            total_unique_moves += len(agent.unique_moves)
            total_finished_goals += len(agent.finished_goals)
        return {'total_unique_moves': total_unique_moves, 'total_finished_goals': total_finished_goals}


def get_random_corridor(env: SimEnvLMAPF) -> List[Node]:
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
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'
    max_time = 100
    corridor_size = 5

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)

    # alg creation + init

    # for rendering
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.1
    total_unique_moves_list = []
    total_finished_goals_list = []

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    for i_step in range(max_time):
        actions = env.sample_actions()  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # render
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        plot_info = {
            'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
            'n_agents': env.n_agents, 'agents': env.agents,
            'total_unique_moves_list': total_unique_moves_list,
            'total_finished_goals_list': total_finished_goals_list,
        }
        plot_step_in_env(ax[0], plot_info)
        plot_total_finished_goals(ax[1], plot_info)
        # plot_unique_movements(ax[1], plot_info)
        plt.pause(plot_rate)

        if terminated:
            break

    plt.show()
    print(f'finished run, metrics: {metrics}')


if __name__ == '__main__':
    main()
