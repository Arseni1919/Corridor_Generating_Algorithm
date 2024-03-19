from typing import List, Dict
from tools_for_graph_nodes import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.old_alg_a_star_space_time import a_star_xyt
from algs.params import *
from environments.env_SACG_LMAPF import SimEnvLMAPF
# from create_animation import do_the_animation


class PrPAgent:
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
        self.next_goal_node: Node = next_goal_node
        self.name_next_goal_node = next_goal_node.xy_name
        self.first_goal_node: Node = next_goal_node
        # self.closed_goal_nodes: List[Node] = []
        self.plan = None
        self.plan_succeeded = False
        self.nodes = kwargs['nodes']
        self.nodes_dict = kwargs['nodes_dict']
        self.h_func = kwargs['h_func']
        self.h_dict = kwargs['h_dict']
        self.map_dim = kwargs['map_dim']
        self.heuristic_value = None
        self.heuristic_value_init = self.h_dict[self.next_goal_node.xy_name][self.curr_node.x, self.curr_node.y]
        self.pf_field = None
        self.nei_list, self.nei_dict, self.nei_plans_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}
        self.nei_pfs = None
        self.h = 5
        self.w = 5

    def update_obs(self, obs, **kwargs):
        self.prev_node = self.curr_node
        self.curr_node = self.nodes_dict[obs.curr_node_name]
        self.next_goal_node = self.nodes_dict[obs.next_goal_node_name]
        self.heuristic_value = self.h_dict[self.next_goal_node.xy_name][self.curr_node.x, self.curr_node.y]

    def clean_nei(self):
        self.nei_list, self.nei_dict, self.nei_plans_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}

    def add_nei(self, nei_agent):
        self.nei_list.append(nei_agent)
        self.nei_dict[nei_agent.name] = nei_agent
        self.nei_plans_dict[nei_agent.name] = nei_agent.plan
        self.nei_h_dict[nei_agent.name] = nei_agent.heuristic_value
        self.nei_num_dict[nei_agent.name] = nei_agent.num
        self.nei_pf_dict[nei_agent.name] = None  # in the _all_exchange_plans method
        self.nei_succ_dict[nei_agent.name] = None  # in the _all_exchange_plans method

    def build_plan(self, h_agents, goal=None, nodes=None, nodes_dict=None):
        # self._execute_a_star(h_agents)
        # {'runtime': time.time() - start_time, 'n_open': len(open_nodes), 'n_closed': len(closed_nodes)}
        a_s_info = {'runtime': 0, 'n_open': 0, 'n_closed': 0}
        if h_agents is None:
            h_agents = []
        if self.plan is None or len(self.plan) == 0:
            nei_h_agents = [agent for agent in h_agents if agent.name in self.nei_dict]
            sub_results = create_sub_results(nei_h_agents)
            v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = build_constraints(self.nodes, sub_results)
            a_s_info = self.execute_a_star(v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem,
                                goal=goal, nodes=nodes, nodes_dict=nodes_dict)

        assert self.plan is not None
        return a_s_info

    def correct_nei_pfs(self):
        if self.nei_pfs is not None:
            self.nei_pfs = self.nei_pfs[:, :, 1:]
            if self.nei_pfs.shape[2] == 0:
                self.nei_pfs = None

    def choose_action(self):
        next_node: Node = self.plan.pop(0)
        return next_node.xy_name

    def get_full_plan(self):
        full_plan = [self.curr_node]
        full_plan.extend(self.plan)
        return full_plan

    def execute_a_star(self, v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem,
                       goal=None, nodes=None, nodes_dict=None) -> dict:
        if goal is None:
            goal = self.next_goal_node
        if nodes is None or nodes_dict is None:
            nodes, nodes_dict = self.nodes, self.nodes_dict
        # v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = self._create_constraints(h_agents)
        new_plan, a_s_info = a_star_xyt(start=self.curr_node, goal=goal,
                                        nodes=nodes, nodes_dict=nodes_dict, h_func=self.h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        agent_name=self.name, k_time=self.w + 1, xyt_problem=xyt_problem)
        if new_plan is not None:
            # pop out the current location, because you will order to move to the next location
            self.plan_succeeded = True
            new_plan.pop(0)
            self.plan = new_plan
            self.fulfill_the_plan()
        else:
            # self.plan = None
            # IStay
            self.set_istay()
        return a_s_info

    def fulfill_the_plan(self):
        if len(self.plan) == 0:
            self.plan = [self.curr_node]
        if self.h and self.h < 1000:
            while len(self.plan) < self.h:
                self.plan.append(self.plan[-1])

    def set_istay(self):
        self.plan = [self.curr_node]
        self.fulfill_the_plan()
        self.plan_succeeded = False
        # print(f' \r\t --- [{self.name}]: I stay!', end='')


class AlgPrP:
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    name = 'PrP'

    def __init__(self, env, **kwargs):
        # limits
        self.time_to_think_limit = 5
        # RHCR part
        self.h = 5
        self.w = 5

        self.env = env
        self.agents = None
        self.agents_dict = {}
        self.curr_iteration = None
        self.agents_names_with_new_goals = []
        self.i_agent = None

        self.env = env
        self.n_agents = env.n_agents
        self.map_dim = env.map_dim
        self.nodes, self.nodes_dict = env.nodes, env.nodes_dict
        self.h_dict = env.h_dict
        self.h_func = env.h_func

        self.agents = None
        self.agents_dict = {}
        self.curr_iteration = None
        self.agents_names_with_new_goals = []


        self.logs = {}

    def reset(self) -> None:
        """
        sets: agents,
        """
        self.agents: List[PrPAgent] = []
        for env_agent in self.env.agents:
            new_agent = PrPAgent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.curr_iteration = 0
        self.i_agent = self.agents_dict['agent_0']

    def initiate_problem(self, obs: dict) -> bool:
        self.logs = {
            'runtime': 0,
            'expanded_nodes': 0,
        }
        self.curr_iteration = 0
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self.agents: List[PrPAgent] = []
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            new_agent = PrPAgent(
                num=num, start_node=start_node, next_goal_node=next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

        self.i_agent = self.agents_dict['agent_0']
        return True

    # @check_time_limit()
    def get_actions(self, observations, **kwargs):
        """
        observations[agent.name] = {
                'num': agent.num,
                'curr_node': agent.curr_node,
                'next_goal_node': agent.next_goal_node,
            }
        actions: {agent_name: node_name, ...}
        """
        self.curr_iteration = observations['iteration'] - 1

        # update the current state
        self.agents_names_with_new_goals = observations['agents_names_with_new_goals']
        for agent in self.agents:
            agent.update_obs(observations[agent.name], agents_dict=self.agents_dict)

        # update neighbours - RHCR part
        self._update_neighbours()

        # build the plans - PF part
        build_plans_info = self._build_plans()

        # choose the actions
        actions = {agent.name: agent.choose_action() for agent in self.agents}

        # alg_info = {
        #     'i_agent': self.i_agent,
        #     'i_nodes': self.nodes,
        #     'time_to_think_limit': self.time_to_think_limit,
        # }
        # alg_info.update(build_plans_info)

        # checks
        check_actions_if_vc(self.agents, actions)
        check_actions_if_ec(self.agents, actions)

        return actions

    def _update_neighbours(self):
        _ = [agent.clean_nei() for agent in self.agents]
        if self.h is None or self.h >= 1e6:
            for agent1, agent2 in combinations(self.agents, 2):
                agent1.add_nei(agent2)
                agent2.add_nei(agent1)
        else:
            for agent1, agent2 in combinations(self.agents, 2):
                distance = manhattan_distance_nodes(agent1.curr_node, agent2.curr_node)
                if distance <= 2 * self.h + 1:
                    agent1.add_nei(agent2)
                    agent2.add_nei(agent1)

    def _update_order(self):
        finished_list = []
        unfinished_list = []
        for agent in self.agents:
            if agent.plan is not None and len(agent.plan) == 0:
                finished_list.append(agent)
            else:
                unfinished_list.append(agent)
        self.agents = unfinished_list
        self.agents.extend(finished_list)

    def _reshuffle_agents(self):
        # print(f'\n**************** random reshuffle ****************\n')

        stuck_agents = [agent for agent in self.agents if not agent.plan_succeeded]
        good_agents = [agent for agent in self.agents if agent.plan_succeeded]
        random.shuffle(stuck_agents)
        random.shuffle(good_agents)
        stuck_agents.extend(good_agents)
        self.agents = stuck_agents

        for agent in self.agents:
            agent.plan = None

    def _implement_istay(self):
        # IStay
        there_is_conf = True
        # pairs_list = list(combinations(self.agents, 2))
        standing_agents = set()
        while there_is_conf:
            there_is_conf = False
            for agent1, agent2 in combinations(self.agents, 2):
                if agent1.name not in agent2.nei_dict:
                    continue
                if agent1.name in standing_agents:
                    if agent2.name in standing_agents:
                        continue
                    if not plan_has_no_conf_with_vertex(agent2.plan, agent1.curr_node):
                        there_is_conf = True
                        agent2.set_istay()
                        standing_agents.add(agent2.name)
                        break
                    else:
                        continue
                if agent2.name in standing_agents:
                    if not plan_has_no_conf_with_vertex(agent1.plan, agent2.curr_node):
                        there_is_conf = True
                        agent1.set_istay()
                        standing_agents.add(agent1.name)
                        break
                    else:
                        continue
                if not two_plans_have_no_confs(agent1.plan, agent2.plan):
                    there_is_conf = True
                    agent1.set_istay()
                    agent2.set_istay()
                    standing_agents.add(agent1.name)
                    standing_agents.add(agent2.name)
                    break

    def _cut_up_to_the_limit(self, i):
        if len(self.agents) >= i:
            for failed_agent in self.agents[i + 1:]:
                failed_agent.set_istay()
        self._implement_istay()

    def initial_prp_assignment(self, start_time):
        # Persist
        h_agents = []
        for i, agent in enumerate(self.agents):
            # a_s_info = {'runtime': 0, 'n_open': 0, 'n_closed': 0}
            a_s_info = agent.build_plan(h_agents)
            h_agents.append(agent)

            # limit check
            self.logs['expanded_nodes'] += a_s_info['n_open'] + a_s_info['n_closed']
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True
        return False

    def _build_plans_persist(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return
        start_time = time.time()
        # self._update_order()
        self._reshuffle_agents()

        # Persist
        time_limit_crossed = self.initial_prp_assignment(start_time)
        if time_limit_crossed:
            return

        # IStay
        self._implement_istay()

    def _build_plans(self):
        self._build_plans_persist()
        return {}


@use_profiler(save_dir='../stats/out_PrP.pstat')
def main():

    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_LMAPF()

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time,
                    corridor_size=corridor_size)

    # alg creation + init
    alg = AlgPrP(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        if to_render:
            i_agent = alg.i_agent
            plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                         'n_agents': env.n_agents, 'agents': alg.agents, 'i_agent': i_agent, 'corridor': i_agent.plan[i_step:]}
            plot_step_in_env(ax[0], plot_info)
            # plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    # if to_render:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_info = {'i': max_time, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                 'n_agents': env.n_agents, 'agents': env.agents, }
    plot_step_in_env(ax[0], plot_info)
    # plot_total_finished_goals(ax[1], plot_info)
    # plot_unique_movements(ax[1], plot_info)
    plt.show()
    # do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
    #                        'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')

    # ------------------------------------------------------ #
    # ------------------------------------------------------ #
    # ------------------------------------------------------ #

    # test_single_alg(
    #     # GENERAL
    #     # random_seed=True,
    #     random_seed=False,
    #     seed=321,
    #     PLOT_PER=1,
    #     PLOT_RATE=0.001,
    #     PLOT_FROM=1,
    #     middle_plot=True,
    #     # middle_plot=False,
    #     final_plot=True,
    #     # final_plot=False,
    #
    #     # FOR ENV
    #     iterations=200,
    #     # iterations=100,
    #     # iterations=50,
    #     n_agents=100,
    #     n_problems=1,
    #     # classical_rhcr_mapf=True,
    #     classical_rhcr_mapf=False,
    #     global_time_limit=100000,
    #     time_to_think_limit=100000,  # seconds
    #     rhcr_mapf_limit=10000,
    #
    #     # Map
    #     # img_dir='empty-32-32.map',  # 32-32
    #     img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    #     # img_dir='random-32-32-20.map',  # 32-32
    #     # img_dir='room-32-32-4.map',  # 32-32
    #     # img_dir='maze-32-32-2.map',  # 32-32
    #     # img_dir='empty-48-48.map',
    # )


if __name__ == '__main__':
    main()





