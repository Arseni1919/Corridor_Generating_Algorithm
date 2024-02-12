import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_corridor_creation import SimEnvCC, get_random_corridor
from alg_gen_cor_v1 import *


class ALgCCv2(ALgCC):
    def __init__(self, img_dir: str, env: SimEnvCC, **kwargs):
        super().__init__(img_dir, env, **kwargs)

    def _solve(self) -> None:
        """
        - create flow to find k empty locations
        - roll the agents upon the flow
        """
        agents_in_corridor: deque = get_agents_in_corridor(self.agents, self.corridor)

        while len(agents_in_corridor) > 0:

            next_agent = agents_in_corridor.pop()

            # find free location
            solvable, tube = self._get_tube(next_agent)
            if not solvable:
                agents_in_corridor.appendleft(next_agent)
                continue

            # roll the agent to the location
            self._roll_agent(next_agent, tube)

    def _get_tube(self, next_agent: AlgAgentCC) -> Tuple[bool, List[Node]]:
        pass

    def _roll_agent(self, ext_agent: AlgAgentCC, tube: List[Node]) -> None:
        pass




def main():
    # set_seed(random_seed_bool=False, seed=973)
    set_seed(random_seed_bool=True)
    # N = 80
    # N = 100
    N = 600
    # N = 700
    # N = 750
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



