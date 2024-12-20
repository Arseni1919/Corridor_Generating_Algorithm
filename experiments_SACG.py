import matplotlib.pyplot as plt

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from main_show_results import show_results
from algs.alg_PrP_SACG import ALgPrPSACG
from algs.alg_CGA import ALgCGA
from algs.alg_PIBT import AlgPIBT
from algs.alg_CBS_SACG import ALgCBS
from environments.env_SACG_LMAPF import SimEnvLMAPF


def run_the_problem(env: SimEnvLMAPF, obs: dict, alg: ALgCGA | ALgPrPSACG) -> bool:
    for i_step in range(1000):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        if terminated:
            return True
    return False


@use_profiler(save_dir='stats/experiments_SACG.pstat')
def main():
    # ---------------------------------------------------- #
    set_seed(random_seed_bool=False, seed=9922)
    # set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)
    # ---------------------------------------------------- #
    n_agents_list = [10, 20 ,30, 40, 50, 60, 70, 80, 90, 100]
    # n_agents_list = [50, 100, 150, 200, 250, 300, 350, 400]
    # n_agents_list = [500, 600, 700, 800, 900, 1000]  # empty
    # n_agents_list = [300, 400, 500, 600, 700, 800]  # rand
    # n_agents_list = [200, 300, 400, 500, 600, 700]  # maze
    # n_agents_list = [100, 200, 300, 400, 500, 600]  # room
    # n_agents_list = [100, 200]
    # n_agents_list = [1000]
    # ---------------------------------------------------- #
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'room-32-32-4.map'

    # img_dir = '15-15-four-rooms.map'  # 100
    # img_dir = '15-15-eight-rooms.map'  # 150
    # img_dir = '15-15-six-rooms.map'   # 175
    # img_dir = '15-15-two-rooms.map'  # 200

    # img_dir = '10_10_my_rand.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    # ---------------------------------------------------- #
    # runs_per_n_agents = 2
    runs_per_n_agents = 5
    # runs_per_n_agents = 15
    # runs_per_n_agents = 25
    # ---------------------------------------------------- #
    # algorithms = [ALgPrPSACG, AlgPIBT, ALgCGA]
    algorithms = [ALgCGA, ALgCBS]
    # algorithms = [AlgPIBT, ALgCBS]
    # algorithms = [AlgPIBT]
    # ---------------------------------------------------- #
    time_to_think_limit = 5
    # ---------------------------------------------------- #
    # to_save_results = True
    to_save_results = False
    # ---------------------------------------------------- #
    to_check_collisions = False
    # to_check_collisions = True
    # ---------------------------------------------------- #
    logs_dict: Dict[str, Any] = {
        alg.name: {
            f'{n_agents}': {
                'runtime': [],
                'sr': [],
                'sq': [],
                'soc': [],
                'expanded_nodes': [],
            } for n_agents in n_agents_list
        } for alg in algorithms
    }
    logs_dict['alg_names'] = [alg.name for alg in algorithms]
    logs_dict['n_agents_list'] = n_agents_list
    logs_dict['runs_per_n_agents'] = runs_per_n_agents
    logs_dict['img_dir'] = img_dir
    logs_dict['time_to_think_limit'] = time_to_think_limit
    # ---------------------------------------------------- #
    middle_plot = True
    # middle_plot = False
    # ---------------------------------------------------- #
    if middle_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ---------------------------------------------------- #
    path_to_maps = 'maps'
    # ---------------------------------------------------- #
    path_to_heuristics = 'logs_for_heuristics'
    # ---------------------------------------------------- #

    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=True, path_to_maps=path_to_maps, path_to_heuristics=path_to_heuristics,
                      to_check_collisions=to_check_collisions)

    for n_agents in n_agents_list:

        # init failed count
        pass

        for i_run in range(runs_per_n_agents):

            # init
            init_nodes = random.sample(env.nodes, n_agents + 1)
            start_nodes = init_nodes[0:n_agents]
            main_goal_node = init_nodes[-1]

            for algorithm in algorithms:

                # the run
                obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], main_goal_node=main_goal_node, max_time=10000,
                                corridor_size=1)
                # alg creation + init
                alg = algorithm(env=env)
                solved = alg.initiate_problem(obs=obs)

                if solved:
                    # run the problem
                    solved = run_the_problem(env, obs, alg)

                # logs
                logs_dict[alg.name][f'{n_agents}']['sr'].append(solved)
                if solved:
                    logs_dict[alg.name][f'{n_agents}']['runtime'].append(alg.logs['runtime'])
                    # logs_dict[alg.name][f'{n_agents}']['expanded_nodes'].append(alg.logs['expanded_nodes'])
                    # logs_dict[alg.name][f'{n_agents}']['sq'].append(len(env.agents_dict['agent_0'].path))
                    # logs_dict[alg.name][f'{n_agents}']['sq'].append(sum([len(a.unique_moves) for a in env.agents]))
                    soc = alg.logs['soc']
                    logs_dict[alg.name][f'{n_agents}']['soc'].append(soc)

                print(f'\n=============================================')
                print(f'{n_agents=}, {i_run=}, {algorithm.name}')
                print(f'=============================================')
                if middle_plot:
                    plot_sr(ax[0], logs_dict)
                    plot_soc(ax[1], logs_dict)
                    # plot_sq_metric_cactus(ax[1], logs_dict)
                    # plot_en_metric_cactus(ax[2], logs_dict)
                    # plot_time_metric_cactus(ax[2], logs_dict)
                    plt.pause(0.001)

    if to_save_results:
        file_dir = save_results(
            algorithms=algorithms, runs_per_n_agents=runs_per_n_agents, img_dir=img_dir, logs_dict=logs_dict
        )
        show_results(file_dir=file_dir)
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

