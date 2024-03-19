import random
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.old_alg_a_star_space_time import a_star_xyt


class SimAgentMAPF:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node | None = None
        self.path: List[Node] = []
        self.unique_moves: List[Node] = []

    @property
    def name(self):
        return f'agent_{self.num}'


class SimEnvMAPF:
    def __init__(self, img_dir: str, to_check_collisions: bool = True, **kwargs):
        self.img_dir = img_dir
        self.to_check_collisions = to_check_collisions
        path_to_maps: str = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        path_to_heuristics: str = kwargs[
            'path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'

        # for the map
        self.map_dim = get_dims_from_pic(img_dir=self.img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.nodes_dict, self.map_dim,
                                                              img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)
        self.agents: List[SimAgentMAPF] = []
        self.agents_dict: Dict[str, SimAgentMAPF] = {}
        self.start_nodes: List[Node] = []

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def start_nodes_names(self):
        return [n.xy_name for n in self.start_nodes]

    @property
    def agents_names(self):
        return [a.name for a in self.agents]

    def reset(self, start_node_names: List[str]) -> Dict[str, Any]:
        self.start_nodes = [self.nodes_dict[snn] for snn in start_node_names]
        self._create_agents()
        # set first goals
        self._update_goals()
        obs = self._get_obs()
        return obs

    def assign_next_goal(self, curr_agent: SimAgentMAPF) -> None:
        occupied_nodes_odict: OrderedDict[str, Node] = OrderedDict()
        for o_agent in self.agents:
            if o_agent == curr_agent:
                continue
            if o_agent.goal_node:
                occupied_nodes_odict[o_agent.goal_node.xy_name] = o_agent.goal_node

        possible_nodes: List[Node] = [n for n in self.nodes]
        next_goal_node: Node = random.choice(possible_nodes)
        while next_goal_node == curr_agent.curr_node:
            next_goal_node: Node = random.choice(possible_nodes)
        curr_agent.goal_node = next_goal_node

    def _update_goals(self):
        for agent in self.agents:
            if agent.goal_node is None:
                self.assign_next_goal(agent)

    def _create_agents(self) -> None:
        self.agents: List[SimAgentMAPF] = []
        self.agents_dict: Dict[str, SimAgentMAPF] = {}
        for i, start_node in enumerate(self.start_nodes):
            new_agent = SimAgentMAPF(num=i, start_node=start_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _get_obs(self) -> dict:
        obs = {agent.name:
                   AgentTupleMAPF(**{
                       'num': agent.num,
                       'start_node_name': agent.start_node.xy_name,
                       'curr_node_name': agent.curr_node.xy_name,
                       'goal_node_name': agent.goal_node.xy_name,
                   })
               for agent in self.agents
               }
        obs['start_nodes_names'] = self.start_nodes_names
        obs['agents_names'] = self.agents_names
        return obs

    def _get_metrics(self) -> dict:
        return {}


def main():
    N = 100
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'

    # problem creation
    env = SimEnvMAPF(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes])
    pprint(obs)


if __name__ == '__main__':
    main()
