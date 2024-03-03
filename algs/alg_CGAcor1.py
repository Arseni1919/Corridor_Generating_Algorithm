from environments.env_LMAPF import SimEnvLMAPF
from algs.alg_gen_cor_v1 import copy_nodes
from algs.alg_clean_corridor import *
from create_animation import do_the_animation
from algs.params import *
from algs.alg_CGA import ALgCGA


class AlgCGAcor1(ALgCGA):

    name = 'CGA_1'

    def __init__(self, env: SimEnvLMAPF, **kwargs):
        super().__init__(env, **kwargs)
        self.freedom_nodes_np: np.ndarray = np.ones(self.img_np.shape)


@use_profiler(save_dir='../stats/alg_AlgCGAcor1.pstat')
def main():
    # SACG
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_SACG()
    # LMAPF
    # N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_LMAPF()

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
    alg = AlgCGAcor1(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        if to_render:
            i_agent = alg.global_order[0]
            plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
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