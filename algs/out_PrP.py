from typing import List, Dict
from tools_for_graph_nodes import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.old_alg_a_star_space_time import a_star_xyt
from algs.params import *
from environments.env_LMAPF import SimEnvLMAPF
from create_animation import do_the_animation


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
        if h_agents is None:
            h_agents = []
        if self.plan is None or len(self.plan) == 0:
            nei_h_agents = [agent for agent in h_agents if agent.name in self.nei_dict]
            sub_results = create_sub_results(nei_h_agents)
            v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = build_constraints(self.nodes, sub_results)
            self.execute_a_star(v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem,
                                goal=goal, nodes=nodes, nodes_dict=nodes_dict)
        assert self.plan is not None
        return self.plan

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
                       goal=None, nodes=None, nodes_dict=None):
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

        # limits
        self.time_to_think_limit = 5

        # RHCR part
        self.h = 5
        self.w = 5

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

    def _build_plans_restart(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return
        start_time = time.time()
        self._update_order()

        h_agents = []
        need_to_shuffle = False
        for i, agent in enumerate(self.agents):
            agent.build_plan(h_agents)
            h_agents.append(agent)
            if not agent.plan_succeeded:
                need_to_shuffle = True

            # limit check
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return

        # IStay
        self._implement_istay()

        if need_to_shuffle:
            random.shuffle(self.agents)

    def initial_prp_assignment(self, start_time):
        # Persist
        h_agents = []
        for i, agent in enumerate(self.agents):
            agent.build_plan(h_agents)
            h_agents.append(agent)

            # limit check
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
        if self.h is None:
            self._build_plans_restart()
        else:
            self._build_plans_persist()
        return {}



class SimAgent:
    def __init__(self, num, start_node, next_goal_node):
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.prev_goal_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.plan = []
        self.reached_the_goal = False
        self.latest_arrival = None
        self.time_passed_from_last_goal = 0
        # self.nei_list, self.nei_dict = [], {}

    def latest_arrival_at_the_goal(self, iteration):
        if self.curr_node.xy_name != self.next_goal_node.xy_name:
            self.reached_the_goal = False
            return
        if not self.reached_the_goal:
            self.reached_the_goal = True
            self.latest_arrival = iteration

    def build_plan(self, **kwargs):
        nodes = kwargs['nodes']
        nodes_dict = kwargs['nodes_dict']
        h_func = kwargs['h_func']
        v_constr_dict = {node.xy_name: [] for node in nodes}
        e_constr_dict = {node.xy_name: [] for node in nodes}
        perm_constr_dict = {node.xy_name: [] for node in nodes}
        new_plan, a_s_info = a_star_xyt(start=self.curr_node, goal=self.next_goal_node,
                                        nodes=nodes, nodes_dict=nodes_dict, h_func=h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        # magnet_w=magnet_w, mag_cost_func=mag_cost_func,
                                        # plotter=self.plotter, middle_plot=self.middle_plot,
                                        # iter_limit=self.iter_limit, k_time=k_time,
                                        agent_name=self.name)
        new_plan.pop(0)
        self.plan = new_plan


class EnvLifelongMAPF:
    def __init__(self, n_agents, img_dir, **kwargs):
        self.n_agents = n_agents
        self.agents: List[SimAgent] = None
        self.img_dir = img_dir
        self.classical_rhcr_mapf = kwargs['classical_rhcr_mapf'] if 'classical_rhcr_mapf' in kwargs else False
        if self.classical_rhcr_mapf:
            self.rhcr_mapf_limit = kwargs['rhcr_mapf_limit']
            self.global_time_limit = kwargs['global_time_limit']
        path_to_maps = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        self.map_dim = get_dims_from_pic(img_dir=img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        path_to_heuristics = kwargs['path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.nodes_dict, self.map_dim,
                                                              img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)

        # for a single run
        self.start_nodes = None
        self.first_goal_nodes = None
        self.iteration = None
        self.start_time = None

        # for plotting
        self.middle_plot = kwargs['middle_plot']
        if self.middle_plot:
            self.plot_per = kwargs['plot_per']
            self.plot_rate = kwargs['plot_rate']
            self.plot_from = kwargs['plot_from']
            # self.fig, self.ax = plt.subplots()
            # self.fig, self.ax = plt.subplots(figsize=(14, 7))
            # self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))

    def reset(self, same_start, predefined=False, scen_name=None):
        self.iteration = 0
        self.start_time = time.time()
        first_run = same_start and self.start_nodes is None
        if predefined:
            self.start_nodes = get_edge_nodes('starts', scen_name, self.nodes_dict, path='../scens')
            self.first_goal_nodes = get_edge_nodes('goals', scen_name, self.nodes_dict, path='../scens')
        elif first_run or not same_start:
            self.start_nodes = random.sample(self.nodes, self.n_agents)
            # available_nodes = [node for node in self.nodes if node not in self.start_nodes]
            # self.first_goal_nodes = random.sample(available_nodes, self.n_agents)
            self.first_goal_nodes = random.sample(self.nodes, self.n_agents)
        self._create_agents()
        observations = self._get_observations([a.name for a in self.agents])
        return observations

    def sample_actions(self, **kwargs):
        actions = {}
        for agent in self.agents:
            if len(agent.plan) == 0:
                agent.build_plan(nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func)
            next_node = agent.plan.pop(0)
            actions[agent.name] = next_node.xy_name
        return actions

    def _classical_rhcr_mapf_termination(self):
        if self.classical_rhcr_mapf:
            if self.iteration >= self.rhcr_mapf_limit:
                return True, 0
            end_time = time.time() - self.start_time
            if end_time >= self.global_time_limit:
                return True, 0
            for agent in self.agents:
                if not agent.reached_the_goal:
                    return False, 0
            return True, 1
        return False, 0

    def _process_single_shot(self, actions):
        to_continue = True
        observations, succeeded, termination, info = None, None, None, None
        if 'one_shot' in actions and actions['one_shot']:
            to_continue = False
            succeeded = actions['succeeded']
            if succeeded:
                for agent in self.agents:
                    agent.latest_arrival = actions['latest_arrivals'][agent.name]
            observations = self._get_observations([])
            termination, info = True, {}
        return to_continue, (observations, succeeded, termination, info)

    def step(self, actions):
        """
        Events might be:
        (1) reaching the goal by any agent, and receiving next assignment
        (2) proceeding to the next moving horizon (h/w in RHCR)
        (3) a collision
        (4) no plan for any agent
        """
        to_continue, return_values = self._process_single_shot(actions)
        if not to_continue:
            observations, succeeded, termination, info = return_values
            return observations, succeeded, termination, info
        self.iteration += 1
        self._execute_actions(actions)
        agents_names_with_new_goals = self._execute_event_new_goal()
        observations = self._get_observations(agents_names_with_new_goals)
        termination, succeeded = self._classical_rhcr_mapf_termination()
        info = {}
        return observations, succeeded, termination, info

    def render(self, info):
        if self.middle_plot and info['i'] >= self.plot_from and info['i'] % self.plot_per == 0:
            plot_env_field(self.ax[0], info)
            plot_magnet_agent_view(self.ax[1], info)
            plt.pause(self.plot_rate)
        # n_closed_goals = sum([len(agent.closed_goal_nodes) for agent in self.agents])
        classical_mapf_str = ''
        if self.classical_rhcr_mapf:
            n_finished_agents = sum([agent.reached_the_goal for agent in self.agents])
            time_passed = time.time() - self.start_time
            classical_mapf_str = f'DONE: {n_finished_agents} / {len(self.agents)}, TIME: {time_passed: .2f}s., '
        print(f"\n\n[{len(self.agents)}][] "
              f"PROBLEM: {info['i_problem'] + 1}/{info['n_problems']}, "
              f"{classical_mapf_str}"
              f"ITERATION: {info['i'] + 1}\n"
              f"Total closed goals --------------------------------> \n"
              f"Total time --------------------------------> {info['runtime']: .2f}s\n")

    def _create_agents(self):
        self.agents = []
        for i, (start_node, goal_node) in enumerate(zip(self.start_nodes, self.first_goal_nodes)):
            new_agent = SimAgent(num=i, start_node=start_node, next_goal_node=goal_node)
            self.agents.append(new_agent)

    def _get_observations(self, agents_names_with_new_goals):
        observations = {
            'agents_names': [agent.name for agent in self.agents],
            'agents_names_with_new_goals': agents_names_with_new_goals
        }
        for agent in self.agents:
            observations[agent.name] = {
                'num': agent.num,
                'curr_node': agent.curr_node,
                'prev_node': agent.prev_node,
                'next_goal_node': agent.next_goal_node,
                'prev_goal_node': agent.prev_goal_node,
                # 'closed_goal_nodes': agent.closed_goal_nodes,
                'latest_arrival': agent.latest_arrival,
                'time_passed_from_last_goal': agent.time_passed_from_last_goal,
                # 'nei_list': [nei.name for nei in agent.nei_list]
            }
        return observations

    def _execute_actions(self, actions):
        for agent in self.agents:
            next_node_name = actions[agent.name]
            agent.prev_node = agent.curr_node
            agent.curr_node = self.nodes_dict[next_node_name]
            agent.time_passed_from_last_goal += 1
            if self.classical_rhcr_mapf:
                agent.latest_arrival_at_the_goal(self.iteration)
        # checks
        check_if_nei_pos(self.agents)
        check_if_vc(self.agents)
        check_if_ec(self.agents)

    def _execute_event_new_goal(self):
        if self.classical_rhcr_mapf:
            for agent in self.agents:
                if agent.curr_node.xy_name == agent.next_goal_node.xy_name:
                    agent.time_passed_from_last_goal = 0
            return []
        goals_names_list = [agent.next_goal_node.xy_name for agent in self.agents]
        available_nodes = [node for node in self.nodes if node.xy_name not in goals_names_list]
        random.shuffle(available_nodes)
        agents_names_with_new_goals = []
        for agent in self.agents:
            if agent.curr_node.xy_name == (closed_goal := agent.next_goal_node).xy_name:
                # agent.closed_goal_nodes.append(closed_goal)
                agent.prev_goal_node = closed_goal
                new_goal_node = available_nodes.pop()
                agent.next_goal_node = new_goal_node
                agent.plan = []
                agent.time_passed_from_last_goal = 0
                agents_names_with_new_goals.append(agent.name)
        return agents_names_with_new_goals

    def close(self):
        pass


def test_single_alg(**kwargs):
    # --------------------------------------------------- #
    # params
    # --------------------------------------------------- #

    # General
    random_seed = kwargs['random_seed']
    seed = kwargs['seed']
    PLOT_PER = kwargs['PLOT_PER']
    PLOT_RATE = kwargs['PLOT_RATE']
    middle_plot = kwargs['middle_plot']
    final_plot = kwargs['final_plot']
    PLOT_FROM = kwargs['PLOT_FROM'] if 'PLOT_FROM' in kwargs else 0

    # --------------------------------------------------- #

    # For env
    iterations = kwargs['iterations']
    n_agents = kwargs['n_agents']
    n_problems = kwargs['n_problems']
    classical_rhcr_mapf = kwargs['classical_rhcr_mapf']
    time_to_think_limit = kwargs['time_to_think_limit']
    rhcr_mapf_limit = kwargs['rhcr_mapf_limit']
    global_time_limit = kwargs['global_time_limit']
    predefined_nodes = kwargs['predefined_nodes'] if 'predefined_nodes' in kwargs else False
    scen_name = kwargs['scen_name'] if predefined_nodes else None

    # Map
    img_dir = kwargs['img_dir']

    # for save
    # to_save_results = True
    # to_save_results = False
    # file_dir = f'logs_for_plots/{datetime.now().strftime("%Y-%m-%d--%H-%M")}_MAP-{img_dir[:-4]}.json'

    if classical_rhcr_mapf:
        iterations = int(1e6)
    # --------------------------------------------------- #
    # --------------------------------------------------- #

    # init
    set_seed(random_seed, seed)
    env = EnvLifelongMAPF(
        n_agents=n_agents, img_dir=img_dir,
        classical_rhcr_mapf=classical_rhcr_mapf, rhcr_mapf_limit=rhcr_mapf_limit, global_time_limit=global_time_limit,
        plot_per=PLOT_PER, plot_rate=PLOT_RATE, plot_from=PLOT_FROM,
        middle_plot=middle_plot, final_plot=final_plot,
    )

    # !!!!!!!!!!!!!!!!!
    alg = AlgPrP(env, time_to_think_limit=time_to_think_limit)

    start_time = time.time()

    info = {
        'iterations': iterations,
        'n_problems': n_problems,
        'n_agents': n_agents,
        'img_dir': img_dir,
        'map_dim': env.map_dim,
        'img_np': env.img_np,
    }

    # loop for n_agents

    for i_problem in range(n_problems):

        observations = env.reset(same_start=False, predefined=predefined_nodes, scen_name=scen_name)

        # !!!!!!!!!!!!!!!!!
        alg.reset()

        # loop for algs
        # observations = env.reset(same_start=True)

        # main loop
        for i in range(iterations):

            # !!!!!!!!!!!!!!!!!
            actions, alg_info = alg.get_actions(observations, iteration=i)  # here is the agents' decision

            # step
            observations, rewards, termination, step_info = env.step(actions)

            # render
            info.update(observations)
            info.update(alg_info)
            info['i_problem'] = i_problem
            info['i'] = i
            info['runtime'] = time.time() - start_time
            env.render(info)

            # unexpected termination
            if termination:
                break

    plt.show()


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





