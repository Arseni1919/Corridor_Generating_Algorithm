import heapq

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_temporal_a_star import calc_temporal_a_star, create_constraints
from environments.env_LMAPF import SimEnvLMAPF
from create_animation import do_the_animation
from algs.params import *


def fill_paths(agents, from_iter, max_path_len):
    for agent in agents:
        while len(agent.path[from_iter:]) < max_path_len:
            agent.path.append(agent.path[-1])


class AgentPrPLMAPF:
    """
    Public methods:
    .update_obs(obs, **kwargs)
    .clean_nei()
    .add_nei(agent1)
    .build_plan(h_agents)
    .choose_action()
    """

    def __init__(self, num: int, start_node, next_goal_node, **kwargs):
        # h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.name_start_node = start_node.xy_name
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.prev_goal_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.name_next_goal_node = next_goal_node.xy_name
        self.first_goal_node: Node = next_goal_node
        self.closed_goal_nodes: List[Node] = []
        self.path: List[Node] = [self.curr_node]
        self.arrived: bool = False
        self.nodes = kwargs['nodes']
        self.nodes_dict = kwargs['nodes_dict']
        self.h_func = kwargs['h_func']
        self.h_dict = kwargs['h_dict']
        self.params = kwargs['params']
        self.map_dim = kwargs['map_dim']
        self.nei_list, self.nei_dict, self.nei_paths_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}
        self.h, self.w = set_h_and_w(self)

    def update_obs(self, obs, **kwargs):
        self.prev_node = self.curr_node
        self.curr_node = self.nodes_dict[obs.curr_node_name]
        self.next_goal_node = self.nodes_dict[obs.next_goal_node_name]
        self.heuristic_value = self.h_dict[self.next_goal_node.xy_name][self.curr_node.x, self.curr_node.y]
        self.arrived = obs.arrived

    def clean_nei(self):
        self.nei_list, self.nei_dict, self.nei_paths_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}

    def add_nei(self, nei_agent):
        self.nei_list.append(nei_agent)
        self.nei_dict[nei_agent.name] = nei_agent
        self.nei_paths_dict[nei_agent.name] = nei_agent.path
        self.nei_h_dict[nei_agent.name] = nei_agent.heuristic_value
        self.nei_num_dict[nei_agent.name] = nei_agent.num


class AlgPrPLMAPF:
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    name = 'PrP'

    def __init__(self, env, **kwargs):
        self.env = env
        self.params = kwargs['params']
        self.map_dim = self.env.map_dim
        self.nodes, self.nodes_dict = self.env.nodes, self.env.nodes_dict
        self.h_dict = self.env.h_dict
        self.h_func = self.env.h_func
        self.agents: List[AgentPrPLMAPF] | None = None
        self.agents_dict: Dict[str, AgentPrPLMAPF] = {}
        self.next_iteration = 0

        # RHCR part
        self.h, self.w = self.params['h'], self.params['w']

    @property
    def n_agents(self):
        return len(self.agents)

    def initiate_problem(self, obs):
        self.agents: List[AgentPrPLMAPF] = []
        for env_agent in self.env.agents:
            new_agent = AgentPrPLMAPF(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim, params=self.params
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.next_iteration = 1

    def get_actions(self, obs):
        """
        observations[agent.name] = {
                'num': agent.num,
                'curr_node': agent.curr_node,
                'next_goal_node': agent.next_goal_node,
            }
        actions: {agent_name: node_name, ...}
        """
        self.next_iteration = obs['iteration']

        # update the current state
        for agent in self.agents:
            agent.update_obs(obs[agent.name], agents_dict=self.agents_dict)

        # update neighbours - RHCR part
        self._update_neighbours()

        # build the plans - PF part
        self._build_plan()

        # choose the actions
        actions = {agent.name: agent.path[self.next_iteration].xy_name for agent in self.agents}

        # checks
        # check_actions_if_vc(self.agents, actions)
        # check_actions_if_ec(self.agents, actions)

        return actions

    def _update_neighbours(self):
        _ = [agent.clean_nei() for agent in self.agents]
        for agent1, agent2 in combinations(self.agents, 2):
            distance = manhattan_distance_nodes(agent1.curr_node, agent2.curr_node)
            if distance <= 2 * self.h + 1:
                agent1.add_nei(agent2)
                agent2.add_nei(agent1)

    def _build_plan(self) -> None:
        if self.next_iteration > 1 and self.next_iteration % self.h != 0:
            return
        self._reshuffle_agents()

        # clean first
        for agent in self.agents:
            agent.path = agent.path[:self.next_iteration]

        planned_agents = []
        unplanned_agents = []
        for agent in self.agents:
            paths = [a.path[max(0, self.next_iteration-1):] for a in planned_agents]
            vc_np, ec_np, pc_np = create_constraints(paths, self.map_dim)
            future_path, info = calc_temporal_a_star(
                curr_node=agent.curr_node, goal_node=agent.next_goal_node,
                nodes_dict=self.nodes_dict, h_dict=self.h_dict, max_len=self.h,
                vc_np=vc_np, ec_np=ec_np, pc_np=pc_np)
            agent.path.extend(future_path[1:])
            if len(future_path) > 1:
                planned_agents.append(agent)
                fill_paths([agent], max(0, self.next_iteration - 1), self.h + 1)
            else:
                unplanned_agents.append(agent)

        # IStay
        there_is_a_conf = True
        from_iter = max(0, self.next_iteration-1)
        agents_i_stay = []
        heapq.heapify(agents_i_stay)
        i_stay_iters = 0
        while there_is_a_conf:
            i_stay_iters += 1
            print(f'\r{i_stay_iters=}', end='')
            there_is_a_conf = False
            for agent1, agent2 in combinations(self.agents, 2):
                if agent1.name not in agent2.nei_dict:
                    continue
                if agent1.num in agents_i_stay and agent2.num in agents_i_stay:
                    continue
                if not two_plans_have_no_confs(agent1.path[from_iter:], agent2.path[from_iter:]):
                    agent1.path = agent1.path[:self.next_iteration]
                    agent1.path.append(agent1.path[-1])
                    agent2.path = agent2.path[:self.next_iteration]
                    agent2.path.append(agent2.path[-1])
                    fill_paths([agent1, agent2], max(0, self.next_iteration - 1), self.h + 1)
                    if agent1.num not in agents_i_stay:
                        heapq.heappush(agents_i_stay, agent1.num)
                    if agent2.num not in agents_i_stay:
                        heapq.heappush(agents_i_stay, agent2.num)
                    there_is_a_conf = True
                    break
        # for agent in self.agents:
        #     assert len(agent.path[self.next_iteration - 1:]) == self.h + 1
        # self._implement_istay()

    def _reshuffle_agents(self):
        # print(f'\n**************** random reshuffle ****************\n')
        good_agents, stuck_agents = [], []
        for agent in self.agents:
            if len(agent.path[self.next_iteration:]) > 0:
                good_agents.append(agent)
                continue
            stuck_agents.append(agent)
        random.shuffle(stuck_agents)
        random.shuffle(good_agents)
        stuck_agents.extend(good_agents)
        self.agents = stuck_agents


@use_profiler(save_dir='../stats/alg_LMAPF_PrP.pstat')
def main():
     # LMAPF
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_LMAPF()
    params = {
        'w': 5,
        'k': 5,
        'h': 5,
    }
    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    # for rendering
    total_unique_moves_list = []
    total_finished_goals_list = []
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = AlgPrPLMAPF(env=env, to_check_paths=to_check_paths, params=params)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        if to_render:
            i_agent = alg.agents[0]
            plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                         'n_agents': env.n_agents, 'agents': alg.agents, 'total_unique_moves_list': total_unique_moves_list,
                         'total_finished_goals_list': total_finished_goals_list, 'i_agent': i_agent, 'corridor': i_agent.path[i_step:]}
            plot_step_in_env(ax[0], plot_info)
            plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    # if to_render:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_info = {'i': max_time, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                 'n_agents': env.n_agents, 'agents': env.agents, 'total_unique_moves_list': total_unique_moves_list,
                 'total_finished_goals_list': total_finished_goals_list, }
    plot_step_in_env(ax[0], plot_info)
    plot_total_finished_goals(ax[1], plot_info)
    # plot_unique_movements(ax[1], plot_info)
    plt.show()
    if env.is_sacg:
        max_time = len(env.agents_dict['agent_0'].path)
    do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
                           'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')


if __name__ == '__main__':
    main()





