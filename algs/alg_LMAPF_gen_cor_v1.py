import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_LMAPF import SimEnvLMAPF
from alg_gen_cor_v1 import copy_nodes
from alg_clean_corridor import clean_corridor, get_path_through_corridor


def get_agents_in_corridor(corridor, node_name_to_agent_dict):
    corridor_names = list(set([n.xy_name for n in corridor]))
    agents_in_corridor = [node_name_to_agent_dict[node_name] for node_name in corridor_names if node_name in node_name_to_agent_dict]
    return agents_in_corridor


def build_perm_constr_list(curr_iteration: int, finished_agents: list, nodes: List[Node]) -> Tuple[Dict[str, List[Node]], List[Node]]:
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    occupied_nodes: List[Node] = []
    for agent in finished_agents:
        a_path = agent.path[curr_iteration:]
        for n in a_path:
            perm_constr_dict[n.xy_name] = [0]
            occupied_nodes.append(n)
    return perm_constr_dict, occupied_nodes


def build_occupied_nodes(curr_iteration: int, finished_agents: list) -> List[Node]:
    occupied_nodes: List[Node] = []
    for agent in finished_agents:
        a_path = agent.path[curr_iteration:]
        for n in a_path:
            occupied_nodes.append(n)
    return occupied_nodes


def calc_corridor(next_agent, nodes, nodes_dict, h_func, corridor_size, perm_constr_dict: Dict[str, List[Node]]) -> \
        List[Node] | None:
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    result, info = a_star_xyt(
        start=next_agent.curr_node, goal=next_agent.next_goal_node, nodes=nodes, h_func=h_func,
        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict, perm_constr_dict=perm_constr_dict,
        plotter=None, middle_plot=False, nodes_dict=nodes_dict,
        xyt_problem=True, k_time=corridor_size,
    )
    return result


def calc_simple_corridor(next_agent, nodes, nodes_dict, h_func, corridor_size, occupied_nodes: List[Node]) -> List[Node] | None:
    corridor: List[Node] = [next_agent.curr_node]
    for i in range(corridor_size):
        next_node = corridor[-1]
        node_name_to_h_value_dict = {
            node_name: h_func(next_agent.next_goal_node, nodes_dict[node_name])
            for node_name in next_node.neighbours
        }
        min_node_name = min(node_name_to_h_value_dict, key=node_name_to_h_value_dict.get)
        min_node = nodes_dict[min_node_name]
        if min_node in occupied_nodes:
            return corridor
        corridor.append(min_node)

    return corridor


class AlgAgentLMAPF:

    def __init__(self, num: int, start_node: Node, next_goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.path: List[Node] = [start_node]
        self.arrived: bool = False

        # self.free_node: Node | None = None
        # self.spanning_tree_dict: Dict[str, str | None] | None = None
        # self.t_agents: list = []
        # self.tube: List[Node] = []
        # self.start_time: int = 0
        # self.finish_time: int = 0

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def last_path_node_name(self):
        return self.path[-1].xy_name

    @property
    def a_curr_node_name(self):
        return self.curr_node.xy_name

    @property
    def a_next_goal_node_name(self):
        return self.next_goal_node.xy_name

    # @property
    # def t_agents_names(self):
    #     return [a.name for a in self.t_agents]
    #
    # @property
    # def tube_names(self):
    #     return [n.xy_name for n in self.tube]

    def __eq__(self, other):
        return self.num == other.num


class ALgLMAPFGenCor:
    def __init__(self, env: SimEnvLMAPF, **kwargs):

        self.env = env
        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np = self.env.img_np
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func

        self.agents: List[AlgAgentLMAPF] = []
        self.agents_dict: Dict[str, AlgAgentLMAPF] = {}
        self.start_nodes: List[Node] = []

        self.max_time: int | None = self.env.max_time
        self.corridor_size: int = self.env.corridor_size

        self.global_order: List[AlgAgentLMAPF] = []

    def initiate_problem(self, obs: dict) -> None:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._check_solvability()
        self._create_agents(obs)
        self.global_order = self.agents[:]

    def get_actions(self, obs: dict) -> Dict[str, str]:
        next_iteration = obs['iteration']
        self._update_agents(obs, next_iteration)
        self.calc_next_steps(next_iteration)
        actions = {
            agent.name: agent.path[next_iteration].xy_name for agent in self.agents
        }
        return actions

    def _check_solvability(self) -> None:
        assert len(self.nodes) - len(self.start_nodes) >= self.corridor_size, 'UNSOLVABLE'

    def _create_agents(self, obs: dict) -> None:
        self.agents: List[AlgAgentLMAPF] = []
        self.agents_dict: Dict[str, AlgAgentLMAPF] = {}
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            new_agent = AlgAgentLMAPF(num=num, start_node=start_node, next_goal_node=next_goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _update_agents(self, obs: dict, next_iteration: int) -> None:
        for env_agent in self.agents:
            obs_agent = obs[env_agent.name]
            env_agent.curr_node = self.nodes_dict[obs_agent.curr_node_name]
            env_agent.next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            env_agent.arrived = obs_agent.arrived
            assert next_iteration != 0
            assert env_agent.curr_node == env_agent.path[next_iteration - 1]

    def _update_global_order(self):
        unfinished_agents, finished_agents = [], []
        for agent in self.global_order:
            if agent.arrived:
                finished_agents.append(agent)
            else:
                unfinished_agents.append(agent)
        self.global_order = unfinished_agents
        self.global_order.extend(finished_agents)

    def _plan_for_agent(self, next_agent: AlgAgentLMAPF, next_iteration: int, finished_agents: List[AlgAgentLMAPF],
                        unfinished_agents: Deque[AlgAgentLMAPF], failed_agents: List[AlgAgentLMAPF],
                        node_name_to_agent_dict: dict) -> Tuple[bool, List[AlgAgentLMAPF]]:
        l_agents = list(unfinished_agents)
        l_agents.extend(failed_agents)
        assert next_agent not in finished_agents
        assert next_agent not in l_agents
        for a in finished_agents:
            assert a not in l_agents

        curr_iteration = next_iteration - 1

        # the given path cannot be longer than the corridor size
        # assert self.corridor_size > len(next_agent.path[curr_iteration:])

        corridor = next_agent.path[curr_iteration:]
        # perm_constr_dict, occupied_nodes = build_perm_constr_list(curr_iteration, finished_agents, self.nodes)
        # corridor = calc_corridor(next_agent, self.nodes, self.nodes_dict, self.h_func, self.corridor_size, perm_constr_dict)
        occupied_nodes = build_occupied_nodes(curr_iteration, finished_agents)
        corridor = calc_simple_corridor(next_agent, self.nodes, self.nodes_dict, self.h_func, self.corridor_size, occupied_nodes)
        # calc the new corridor
        # there is not a plan up until the goal location
        # if next_agent.path[-1] != next_agent.next_goal_node:
        if not corridor:
            return False, []
        # node_name_to_unfinished_dict = {a.curr_node.xy_name: a for a in self.agents}
        agents_in_corridor = get_agents_in_corridor(corridor, node_name_to_agent_dict)
        agents_in_corridor.remove(next_agent)
        if len(agents_in_corridor) == 0:
            # no need to move anybody
            next_agent.path = next_agent.path[:curr_iteration]
            next_agent.path.extend(corridor)
            return True, []

        for a in agents_in_corridor:
            assert a in l_agents

        for a in finished_agents:
            assert a not in l_agents

        # FOR SURE THERE ARE OTHER AGENT ON THE ROAD HERE!

        # move others out of the corridor
        succeeded, cc_paths_dict = clean_corridor(
            next_agent, corridor, agents_in_corridor, occupied_nodes, l_agents,
            node_name_to_agent_dict, self.nodes, self.nodes_dict, self.img_np,
            curr_iteration=curr_iteration
        )
        if not succeeded:
            return False, []

        # assign new paths to the next_agent and other captured agents
        captured_agents: List[AlgAgentLMAPF] = []
        path_through_corridor = get_path_through_corridor(next_agent, corridor, cc_paths_dict)
        next_agent.path = next_agent.path[:curr_iteration]
        next_agent.path.extend(path_through_corridor)
        for agent_name, alt_path in cc_paths_dict.items():
            o_agent = self.agents_dict[agent_name]
            o_agent.path = o_agent.path[:curr_iteration]
            o_agent.path.extend(alt_path)
            captured_agents.append(o_agent)

        # agents_to_check = [a for a in self.agents if a in finished_agents]
        # agents_to_check.extend([a for a in self.agents if a in captured_agents])
        # agents_to_check.append(next_agent)
        # check_if_nei_pos_iter(agents_to_check, next_iteration)
        # check_if_vc_iter(agents_to_check, next_iteration)
        # check_if_ec_iter(agents_to_check, next_iteration)
        return True, captured_agents

    def calc_next_steps(self, next_iteration: int) -> None:
        # first - is the highest
        self._update_global_order()
        unfinished_agents: Deque[AlgAgentLMAPF] = deque(self.global_order)
        failed_agents: List[AlgAgentLMAPF] = []
        finished_agents: List[AlgAgentLMAPF] = []
        node_name_to_agent_dict = {a.curr_node.xy_name: a for a in self.agents}
        while len(unfinished_agents) > 0:
            next_agent = unfinished_agents.popleft()
            if next_agent in finished_agents:
                continue
            planned, captured_agents = self._plan_for_agent(
                next_agent, next_iteration, finished_agents, unfinished_agents, failed_agents, node_name_to_agent_dict)
            if planned:
                finished_agents.append(next_agent)
                finished_agents.extend(captured_agents)
                failed_agents = list(filter(lambda n: n not in finished_agents, failed_agents))
                unfinished_agents = deque(filter(lambda n: n not in finished_agents, unfinished_agents))
            else:
                failed_agents.append(next_agent)
                # unfinished_agents.append(next_agent)
        for f_agent in failed_agents:
            f_agent.path = f_agent.path[:next_iteration]
            f_agent.path.append(f_agent.path[-1])

        # checks
        check_if_nei_pos_iter(self.agents, next_iteration)
        check_if_vc_iter(self.agents, next_iteration)
        check_if_ec_iter(self.agents, next_iteration)


@use_profiler(save_dir='../stats/alg_LMAPF_gen_cor_v1.pstat')
def main():
    set_seed(random_seed_bool=False, seed=218)
    # set_seed(random_seed_bool=True)
    # N = 70
    N = 100
    # N = 300
    # N = 400
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 850
    # N = 2000
    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    max_time = 20
    corridor_size = 5

    # to_render: bool = True
    to_render: bool = False

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)

    # for rendering
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.1
        total_unique_moves_list = []
        total_finished_goals_list = []

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = ALgLMAPFGenCor(env=env)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        if to_render:
            total_unique_moves_list.append(metrics['total_unique_moves'])
            total_finished_goals_list.append(metrics['total_finished_goals'])
            i_agent = alg.agents[49]
            plot_info = {
                'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
                'n_agents': env.n_agents, 'agents': alg.agents,
                'total_unique_moves_list': total_unique_moves_list,
                'total_finished_goals_list': total_finished_goals_list,
                'i_agent': i_agent,
                'corridor': i_agent.path[i_step:]
            }
            plot_step_in_env(ax[0], plot_info)
            plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    if to_render:
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
        plt.show()
    print(f'finished run')


if __name__ == '__main__':
    main()
