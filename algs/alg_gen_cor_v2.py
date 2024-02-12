import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_corridor_creation import SimEnvCC, get_random_corridor
from alg_gen_cor_v1 import *


def get_assign_agent_to_node_dict(tube: List[Node], t_agents: List[AlgAgentCC], corridor: List[Node]) -> Dict[str, Node]:
    copy_t_agents = t_agents[:]
    assign_agent_to_node_dict: Dict[str, Node] = {}
    for n in tube:
        if n in corridor:
            continue
        next_agent = copy_t_agents.pop(0)
        assign_agent_to_node_dict[next_agent.name] = n
    assert len(copy_t_agents) == 0
    return assign_agent_to_node_dict


def get_full_tube(free_node: Node, spanning_tree_dict: Dict[str, str], nodes_dict: Dict[str, Node]) -> List[Node]:
    tube: List[Node] = [free_node]
    parent = spanning_tree_dict[free_node.xy_name]
    while parent is not None:
        parent_node = nodes_dict[parent]
        tube.append(parent_node)
        parent = spanning_tree_dict[parent]
    return tube


def tube_is_free_to_go(tube: List[Node], inner_captured_nodes: list, next_agent: AlgAgentCC) -> bool:
    # tube: free node -> init node
    sub_tube = tube[:-1]
    assert next_agent.path[-1] not in sub_tube
    for n in sub_tube:
        if n in inner_captured_nodes:
            return False
    return True


class ALgCCv2(ALgCC):
    def __init__(self, img_dir: str, env: SimEnvCC, **kwargs):
        super().__init__(img_dir, env, **kwargs)

    def _solve(self) -> None:
        """
        - create flow to find k empty locations
        - roll the agents upon the flow
        """
        agents_in_corridor: deque = get_agents_in_corridor(self.agents, self.corridor)

        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor}
        # plot_flow_in_env(ax[0], plot_info)
        # plt.show()
        # plt.close()
        effective_order: List[AlgAgentCC] = []
        counter = 0
        while len(agents_in_corridor) > 0:
            counter += 1
            print(f'{counter=} | {len(agents_in_corridor)=}')
            next_agent = agents_in_corridor.pop()

            # if counter > 45:
            #     fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            #     plot_info = {'img_np': self.img_np, 'agents': self.agents, 'corridor': self.corridor,
            #                  'next_agent': next_agent}
            #     plot_flow_in_env(ax[0], plot_info)
            #     plt.show()
            #     plt.close()

            # find free location
            solvable, tube = self._get_tube(next_agent)
            if not solvable:
                agents_in_corridor.appendleft(next_agent)
                continue

            # roll the agent to the location
            self._roll_agent(next_agent, tube)
            effective_order.append(next_agent)
            # agents_in_corridor = get_agents_in_corridor(self.agents, self.corridor)

        self._tuning(effective_order)

    def _get_tube(self, next_agent: AlgAgentCC) -> Tuple[bool, List[Node] | None]:
        inner_captured_nodes, outer_captured_nodes = [], []
        for agent in self.agents:
            if agent.path[-1] in self.corridor:
                inner_captured_nodes.append(agent.path[-1])
            else:
                outer_captured_nodes.append(agent.path[-1])
        spanning_tree_dict: Dict[str, str | None] = {next_agent.path[-1].xy_name: None}
        open_list: Deque[Node] = deque([next_agent.path[-1]])
        closed_list: Deque[Node] = deque()
        small_iteration: int = 0
        while len(open_list) > 0:
            small_iteration += 1

            selected_node = open_list.pop()
            if selected_node not in self.corridor and selected_node not in outer_captured_nodes:
                next_agent.free_node = selected_node
                next_agent.spanning_tree_dict = spanning_tree_dict
                tube = get_full_tube(selected_node, spanning_tree_dict, self.nodes_dict)
                if tube_is_free_to_go(tube, inner_captured_nodes, next_agent):
                    return True, tube
                return False, None

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
        return False, None

    def _roll_agent(self, next_agent: AlgAgentCC, tube: List[Node]) -> None:
        next_agent.start_time = len(next_agent.path)
        node_to_t_agent_dict = {agent.path[-1].xy_name: agent for agent in self.agents}
        t_agents: List[AlgAgentCC] = [node_to_t_agent_dict[n.xy_name] for n in tube if n.xy_name in node_to_t_agent_dict]
        assert next_agent in t_agents
        next_agent.t_agents = t_agents
        next_agent.tube = tube
        out_of_t_agents: List[AlgAgentCC] = [agent for agent in self.agents if agent not in t_agents]
        assign_agent_to_t_node_dict = get_assign_agent_to_node_dict(tube, t_agents, self.corridor)
        there_is_movement = True
        while there_is_movement:
            there_is_movement = False
            node_to_t_agent_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}
            pairwise_tube: List[Tuple[Node, Node]] = pairwise_list(tube)
            for to_t_node, from_t_node in pairwise_tube:
                if to_t_node == tube[0] and to_t_node.xy_name in node_to_t_agent_dict:
                    front_agent = node_to_t_agent_dict[to_t_node.xy_name]
                    front_agent.path.append(to_t_node)
                if from_t_node.xy_name in node_to_t_agent_dict:
                    curr_agent = node_to_t_agent_dict[from_t_node.xy_name]
                    if assign_agent_to_t_node_dict[curr_agent.name] == from_t_node:
                        curr_agent.path.append(from_t_node)
                    elif to_t_node.xy_name in node_to_t_agent_dict:
                        curr_agent.path.append(from_t_node)
                    else:
                        curr_agent.path.append(to_t_node)
                        there_is_movement = True
                        node_to_t_agent_dict = {t_agent.path[-1].xy_name: t_agent for t_agent in t_agents}

            len_list: List[int] = [len(t_agent.path) for t_agent in t_agents]
            assert len(set(len_list)) == 1
        next_agent.finish_time = len(next_agent.path)
        for o_agent in out_of_t_agents:
            while len(o_agent.path) < next_agent.finish_time:
                o_agent.path.append(o_agent.path[-1])

        # for t_agent in t_agents:
        #     assert t_agent.path[-1] not in self.corridor

    def _tuning(self, effective_order: List[AlgAgentCC]) -> None:
        for agent1 in effective_order:
            to_cut = True
            for agent2 in self.agents:
                if agent1 == agent2:
                    continue
                if len(intersection(agent1.tube_names, agent2.tube_names)) != 0:
                    to_cut = False
                    break
            if not to_cut:
                continue
            for t_agent in agent1.t_agents:
                t_agent.path = t_agent.path[agent1.start_time - 1:]

        max_len = max([len(agent.path) for agent in self.agents])
        for agent in self.agents:
            while len(agent.path) < max_len:
                agent.path.append(agent.path[-1])
        len_list: List[int] = [len(agent.path) for agent in self.agents]
        assert len(set(len_list)) == 1

        prev_config: List[Tuple[int, int]] = [a.path[0].xy for a in self.agents]
        finished = False
        curr_index = 1
        while not finished:
            if len(self.agents[0].path) < 4:
                break
            next_config = [a.path[curr_index].xy for a in self.agents]
            if same_configs(prev_config, next_config):
                for agent in self.agents:
                    agent.path.pop(curr_index)
                curr_index = min(curr_index, len(self.agents[0].path) - 1)
                continue
            curr_index += 1
            prev_config = next_config
            if curr_index >= len(self.agents[0].path) - 1:
                finished = True



        # for agent1, agent2 in combinations(self.agents, 2):
        #     print(f'{agent1.name}-{agent2.name}')
        #     assert two_plans_have_no_confs(agent1.path, agent2.path)


def main():
    set_seed(random_seed_bool=False, seed=552)
    # set_seed(random_seed_bool=True)
    # N = 80
    # N = 100
    # N = 300
    # N = 400
    # N = 500
    # N = 600
    N = 620
    # N = 700
    # N = 750
    # N = 850
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    img_dir = 'maze-32-32-2.map'

    # problem creation
    env = SimEnvCC(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    corridor = get_random_corridor(env)

    # alg creation + init
    alg = ALgCCv2(img_dir=img_dir, env=env)
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
            'i': i_step, 'img_dir': img_dir, 'img_np': env.img_np,
            'n_agents': env.n_agents, 'agents': env.agents, 'corridor': corridor,
            'total_unique_moves_list': total_unique_moves_list,
        }
        plot_step_in_env(ax[0], plot_info)
        plot_unique_movements(ax[1], plot_info)
        plt.pause(plot_rate)

        if terminated:
            break

    plot_info = {
        'i': i_step, 'img_dir': img_dir, 'img_np': env.img_np,
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



