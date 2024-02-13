import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_LMAPF import SimEnvLMAPF
from alg_gen_cor_v1 import copy_nodes


class AlgAgentLMAPF:

    def __init__(self, num: int, start_node: Node, next_goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.path: List[Node] = [start_node]

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

    def initiate_problem(self, obs: dict) -> None:
        self.start_nodes = [self.nodes_dict[s_name] for s_name in obs['start_nodes_names']]
        self._check_solvability()
        self._create_agents(obs)

    def get_actions(self, obs: dict) -> Dict[str, str]:
        iteration = obs['iteration']
        self._update_agents(obs, iteration)
        self.calc_next_steps(iteration)
        actions = {
            agent.name: agent.path[iteration].xy_name for agent in self.agents
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

    def _update_agents(self, obs: dict, iteration: int) -> None:
        for env_agent in self.agents:
            obs_agent = obs[env_agent.name]
            env_agent.curr_node = self.nodes_dict[obs_agent.curr_node_name]
            env_agent.next_goal_node = self.nodes_dict[obs_agent.next_goal_node_name]
            assert iteration != 0
            assert env_agent.curr_node == env_agent.path[iteration - 1]

    def calc_next_steps(self, iteration: int) -> None:
        pass


@use_profiler(save_dir='../stats/alg_LMAPF_gen_cor_v1.pstat')
def main():
    # set_seed(random_seed_bool=False, seed=218)
    set_seed(random_seed_bool=True)
    N = 70
    # N = 100
    # N = 300
    # N = 400
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 850
    # N = 2000
    img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    max_time = 100
    corridor_size = 5

    to_render: bool = True
    # to_render = False

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



