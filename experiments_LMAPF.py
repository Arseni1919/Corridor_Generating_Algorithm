from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from main_show_results import show_results
from environments.env_SACG_LMAPF import SimEnvLMAPF
from algs.alg_CGA import ALgCGA
from algs.alg_CGAcor1 import AlgCGAcor1
from algs.out_PrP import AlgPrP
from algs.out_LNS2 import AlgLNS2
from algs.alg_PIBT import AlgPIBT


def run_the_problem(env: SimEnvLMAPF, obs: dict, alg: Any, max_time: int) -> None:
    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        if terminated:
            break


@use_profiler(save_dir='stats/experiments_LMAPF.pstat')
def main():
    # ---------------------------------------------------- #
    set_seed(random_seed_bool=False, seed=9922)
    # set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)
    # ---------------------------------------------------- #
    # n_agents_list = [50, 100, 150, 200, 250, 300, 350, 400]
    # n_agents_list = [500, 600, 700, 800, 900, 1000]  # empty
    # n_agents_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]  # empty
    # n_agents_list = [100, 200, 300, 400, 500, 600, 700, 800]  # rand
    # n_agents_list = [100, 200, 300, 400, 500, 600, 700]  # maze
    # n_agents_list = [100, 200, 300, 400, 500, 600]  # room
    # n_agents_list = [100, 200, 300, 400, 500, 600]  # maze 32-32-2
    # n_agents_list = [100, 200]

    # n_agents_list = [50, 75, 100]  # 15-15-four-rooms
    # n_agents_list = [50, 75, 100, 125, 150]  # 15-15-eight-rooms
    # n_agents_list = [50, 75, 100, 125, 150, 175]  # 15-15-six-rooms ~
    n_agents_list = [50, 75, 100, 125, 150, 175, 200]  # 15-15-two-rooms ~
    # n_agents_list = [175]
    # ---------------------------------------------------- #
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'room-32-32-4.map'

    # img_dir = '15-15-four-rooms.map'  # 100
    # img_dir = '15-15-eight-rooms.map'  # 150
    # img_dir = '15-15-six-rooms.map'   # 175
    img_dir = '15-15-two-rooms.map'   # 200

    # img_dir = '10_10_my_rand.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    # ---------------------------------------------------- #
    # max_time = 20
    # max_time = 50
    max_time = 100
    # max_time = 200
    # ---------------------------------------------------- #
    k = 5
    # ---------------------------------------------------- #
    # runs_per_n_agents = 2
    # runs_per_n_agents = 5
    # runs_per_n_agents = 15
    # runs_per_n_agents = 15
    runs_per_n_agents = 25
    # ---------------------------------------------------- #
    # algorithms = [ALgCBS, AlgLNS2, AlgPrP]
    # algorithms = [ALgCBS]
    # algorithms = [ALgCBS, AlgCGAcor1]
    # algorithms = [AlgPrP, AlgPIBT, ALgCBS]
    # algorithms = [AlgPIBT, ALgCBS]
    algorithms = [AlgPrP]
    # ---------------------------------------------------- #
    time_to_think_limit = 5
    # ---------------------------------------------------- #
    to_save_results = True
    # to_save_results = False
    # ---------------------------------------------------- #
    to_check_collisions = False
    # to_check_collisions = True
    # ---------------------------------------------------- #
    # middle_plot = True
    middle_plot = False
    # ---------------------------------------------------- #
    if middle_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ---------------------------------------------------- #
    path_to_maps = 'maps'
    # ---------------------------------------------------- #
    path_to_heuristics = 'logs_for_heuristics'
    # ---------------------------------------------------- #
    logs_dict: Dict[str, Any] = {
        alg.name: {
            f'{n_agents}': {
                'runtime': [],
                'throughput': [],
                'expanded_nodes': [],
                'max_waiting': [],
            } for n_agents in n_agents_list
        } for alg in algorithms
    }
    logs_dict['alg_names'] = [alg.name for alg in algorithms]
    logs_dict['n_agents_list'] = n_agents_list
    logs_dict['runs_per_n_agents'] = runs_per_n_agents
    logs_dict['img_dir'] = img_dir
    logs_dict['time_to_think_limit'] = time_to_think_limit
    logs_dict['max_time'] = max_time
    # ---------------------------------------------------- #

    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=False, path_to_maps=path_to_maps, path_to_heuristics=path_to_heuristics,
                      to_check_collisions=to_check_collisions)

    for n_agents in n_agents_list:

        # init failed count
        pass

        for i_run in range(runs_per_n_agents):

            # init
            start_nodes = random.sample(env.nodes, n_agents)

            for algorithm in algorithms:

                # the run
                obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time,
                                corridor_size=1)
                # alg creation + init
                alg = algorithm(env=env)
                alg.initiate_problem(obs=obs)

                # run the problem
                run_the_problem(env, obs, alg, max_time)

                # logs
                logs_dict[alg.name][f'{n_agents}']['runtime'].append(alg.logs['runtime'])
                logs_dict[alg.name][f'{n_agents}']['expanded_nodes'].append(alg.logs['expanded_nodes'])
                logs_dict[alg.name][f'{n_agents}']['throughput'].append(sum(map(lambda a: len(a.finished_goals), env.agents)))
                logs_dict[alg.name][f'{n_agents}']['max_waiting'].append(sum(map(lambda a: a.stuck_count, env.agents)))

                print(f'\n=============================================')
                print('^'*40)
                print(f'>>>{algorithm.name}<<< | {n_agents=}, {i_run=}')
                print(f'=============================================')
                if middle_plot:
                    plot_throughput(ax[0], logs_dict)
                    plot_max_waiting(ax[1], logs_dict)
                    # plot_time_metric_cactus(ax[1], logs_dict)
                    # plot_sq_metric_cactus(ax[1], logs_dict)
                    # plot_en_metric_cactus(ax[1], logs_dict)
                    plt.pause(0.01)

    if to_save_results:
        file_dir = save_results(
            algorithms=algorithms, runs_per_n_agents=runs_per_n_agents, img_dir=img_dir, logs_dict=logs_dict
        )
        show_results(file_dir=file_dir, lmapf=True)
    plt.show()
    # final print
    print('\n###################################################')
    print('###################################################')
    print('###################################################')
    print('###################################################')
    print('###################################################')
    print('Finished.')


if __name__ == '__main__':
    main()

