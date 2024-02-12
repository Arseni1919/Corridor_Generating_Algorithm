import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_corridor_creation import SimEnvCC, get_random_corridor
from alg_gen_cor_v1 import *


class ALgCCv3(ALgCC):
    def __init__(self, img_dir: str, env: SimEnvCC, **kwargs):
        super().__init__(img_dir, env, **kwargs)

    def _solve(self) -> None:
        pass
@use_profiler(save_dir='../stats/alg_gen_cor_v3.pstat')
def main():
    set_seed(random_seed_bool=False, seed=552)
    # set_seed(random_seed_bool=True)
    # N = 80
    # N = 100
    # N = 300
    # N = 400
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 850
    N = 2000
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-2.map'
    img_dir = 'random-64-64-20.map'

    # to_render = True
    to_render = False


    # problem creation
    env = SimEnvCC(img_dir=img_dir)
    start_nodes = random.sample(env.nodes, N)
    corridor = get_random_corridor(env)

    # alg creation + init
    alg = ALgCCv3(img_dir=img_dir, env=env)
    alg.initiate_problem(start_node_names=[n.xy_name for n in start_nodes], corridor_names=[n.xy_name for n in corridor])

    # for rendering
    if to_render:
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
        if to_render:
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

    if to_render:
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

