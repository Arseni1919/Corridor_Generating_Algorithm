from typing import List, Dict
from tools_for_graph_nodes import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.old_alg_a_star_space_time import a_star_xyt
from algs.params import *
from environments.env_LMAPF import SimEnvLMAPF
from create_animation import do_the_animation
from algs.out_PrP import AlgPrP, PrPAgent


class LNS2Agent(PrPAgent):
    def __init__(self, num: int, start_node, next_goal_node, **kwargs):
        super().__init__(num, start_node, next_goal_node, **kwargs)


class AlgLNS2(AlgPrP):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """

    name = 'LNS2'

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.big_N = 5

        self.conf_matrix = None
        self.conf_agents_names_list = None
        self.conf_vv_random_walk = None
        self.conf_neighbourhood = None

    def reset(self):
        self.agents: List[PrPAgent] = []
        for env_agent in self.env.agents:
            new_agent = LNS2Agent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim)
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
        self.agents: List[LNS2Agent] = []
        agents_names = obs['agents_names']
        for agent_name in agents_names:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            new_agent = LNS2Agent(
                num=num, start_node=start_node, next_goal_node=next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

        self.i_agent = self.agents_dict['agent_0']
        return True

    def _build_G_c(self):
        num_of_agents = len(self.agents)
        self.conf_matrix = np.zeros((num_of_agents, num_of_agents))
        self.conf_agents_names_list = []
        num_of_confs = 0
        for agent1, agent2 in combinations(self.agents, 2):
            if agent1.name not in agent2.nei_dict:
                continue
            if not two_plans_have_no_confs(agent1.plan, agent2.plan):
                num_of_confs += 1
                self.conf_matrix[agent1.num, agent2.num] = 1
                self.conf_matrix[agent2.num, agent1.num] = 1
                self.conf_agents_names_list.append(agent1.name)
                self.conf_agents_names_list.append(agent2.name)
        self.conf_agents_names_list = list(set(self.conf_agents_names_list))
        return num_of_confs

    def _select_random_conf_v(self):
        self.conf_vv_random_walk = []
        v = self.agents_dict[random.choice(self.conf_agents_names_list)]
        V_v, V_v_nums = [], []
        new_leaves = [v.num]
        while len(new_leaves) > 0:
            next_leave = new_leaves.pop(0)
            if len(self.conf_vv_random_walk) < self.big_N and next_leave != v.num:
                self.conf_vv_random_walk.append(self.agents_dict[f'agent_{next_leave}'])
                V_v_nums.append(next_leave)
            elif len(self.conf_vv_random_walk) == self.big_N:
                break
            children = np.where(self.conf_matrix[next_leave] == 1)[0]
            new_leaves.extend([c for c in children if c not in V_v_nums])
            new_leaves = list(set(new_leaves))
        V_v = [self.agents_dict[f'agent_{num}'] for num in V_v_nums]
        return V_v, v

    def _fill_the_neighbourhood(self, V_v):
        self.conf_neighbourhood = V_v
        not_in_nbhd = [agent for agent in self.agents if agent not in self.conf_neighbourhood]
        while len(self.conf_neighbourhood) < self.big_N:
            new_one = random.choice(not_in_nbhd)
            self.conf_neighbourhood.append(new_one)

    def _solve_with_PrP(self):
        random.shuffle(self.conf_neighbourhood)
        # reset
        for agent in self.conf_neighbourhood:
            agent.plan = None

        h_agents = [agent for agent in self.agents if agent not in self.conf_neighbourhood]
        for agent in self.agents:
            a_s_info = agent.build_plan(h_agents)
            h_agents.append(agent)

            self.logs['expanded_nodes'] += a_s_info['n_open'] + a_s_info['n_closed']

    def _replace_old_plans(self):
        pass

    def _build_plans(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return {}
        """
        LNS2:
        - build PrP
        - While there are collisions:
            - Build G_c
            - v <- select a random vertex v from Vc with deg(v) > 0
            - V_v <- find the connected component that has v
            - If |V_v| ≤ N:
                - put all inside
                - select random agent from V_v and do the random walk until fulfill V_v up to N
            - Otherwise:
                - Do the random walk from v in G_c until you have N agents
            - Solve the neighbourhood V_v with PrP (random priority)
            - Replace old plans with the new plans iff the number of colliding pairs (CP) of the paths in the new plan
              is no larger than that of the old plan
        return: valid plans
        """
        start_time = time.time()
        # self._update_order()
        self._reshuffle_agents()

        time_limit_crossed = self.initial_prp_assignment(start_time)
        if time_limit_crossed:
            return {}

        while (num_of_confs := self._build_G_c()) > 0:
            V_v, v = self._select_random_conf_v()
            if len(V_v) >= self.big_N:
                self.conf_neighbourhood = self.conf_vv_random_walk
            else:
                self._fill_the_neighbourhood(V_v)

            self._solve_with_PrP()
            self._replace_old_plans()
            # print(f'\n{num_of_confs=}')
            print(f'\r{num_of_confs=}', end='')

            # limit check
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._implement_istay()
                return {}
        return {}


@use_profiler(save_dir='../stats/out_LNS2.pstat')
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
    alg = AlgLNS2(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        if to_render:
            i_agent = alg.agents[0]
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
    # Alg params

    # params_dict = {
    #     'ParObs-LNS2': {
    #         'big_N': big_N,
    #         # For RHCR
    #         'h': 5,  # my step
    #         'w': 5,  # my planning
    #     },
    # }
    #
    # alg = AlgLNS2(params=params_dict[alg_name], alg_name=alg_name)
    # test_single_alg(
    #     alg,
    #
    #     # GENERAL
    #     # random_seed=True,
    #     random_seed=False,
    #     seed=321,
    #     PLOT_PER=1,
    #     # PLOT_PER=20,
    #     PLOT_RATE=0.001,
    #     # PLOT_RATE=5,
    #     PLOT_FROM=1,
    #     middle_plot=True,
    #     # middle_plot=False,
    #     final_plot=True,
    #     # final_plot=False,
    #
    #     # FOR ENV
    #     # iterations=50,  # !!!
    #     # iterations=200,
    #     iterations=100,
    #     n_agents=100,
    #     n_problems=1,
    #     # classical_rhcr_mapf=True,
    #     classical_rhcr_mapf=False,
    #     time_to_think_limit=3000,  # seconds
    #     rhcr_mapf_limit=10000,
    #     global_time_limit=6000,  # seconds
    #
    #     # Map
    #     # img_dir='10_10_my_rand.map',  # 10-10
    #     # img_dir='10_10_my_empty.map',  # 10-10
    #
    #     # img_dir='empty-32-32.map',  # 32-32
    #     img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    #     # img_dir='random-32-32-20.map',  # 32-32
    #     # img_dir='room-32-32-4.map',  # 32-32
    #     # img_dir='maze-32-32-2.map',  # 32-32
    #
    #     # img_dir='warehouse-10-20-10-2-1.map',
    # )


if __name__ == '__main__':
    main()

