from globals import *
from tools_for_plotting import *


def show_results(file_dir: str, lmapf: bool = False) -> None:
    plt.close()
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        logs_dict = json.load(openfile)

        if lmapf:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_throughput(ax[0], logs_dict)
            plot_en_metric_cactus(ax[1], logs_dict)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_sr(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_sq_metric_cactus(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_en_metric_cactus(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_time_metric_cactus(ax, logs_dict)
            plt.show()

        plt.show()


def main():
    # file_dir = '2024-02-29--02-08_ALGS-1_RUNS-25_MAP-empty-32-32.json'
    # file_dir = '2024-02-29--05-25_ALGS-1_RUNS-25_MAP-random-32-32-20.json'
    # file_dir = '2024-02-29--10-50_ALGS-1_RUNS-25_MAP-maze-32-32-4.json'
    # show_results(file_dir=f'logs_experiments/{file_dir}', lmapf=True)

    # SACG - old
    # file_dir = 'SACG_2024-02-27--16-18_ALGS-2_RUNS-25_MAP-empty-32-32.json'
    # file_dir = 'SACG_2024-02-27--16-44_ALGS-2_RUNS-25_MAP-random-32-32-20.json'
    # file_dir = 'SACG_2024-02-27--17-14_ALGS-2_RUNS-25_MAP-maze-32-32-4.json'
    # file_dir = 'SACG_2024-02-27--15-39_ALGS-2_RUNS-25_MAP-room-32-32-4.json'

    # SACG
    # file_dir = '2024-03-11--11-57_ALGS-3_RUNS-25_MAP-empty-32-32.json'
    # file_dir = '2024-03-11--12-15_ALGS-3_RUNS-25_MAP-random-32-32-20.json'
    # file_dir = '2024-03-11--12-39_ALGS-3_RUNS-25_MAP-maze-32-32-4.json'
    file_dir = '2024-03-11--13-27_ALGS-3_RUNS-25_MAP-room-32-32-4.json'
    show_results(file_dir=f'final_logs/sacg/{file_dir}')

    # LMAPF
    # file_dir = '2024-03-11--17-12_ALGS-2_RUNS-25_MAP-15-15-four-rooms.json'
    # file_dir = '2024-03-11--18-05_ALGS-1_RUNS-25_MAP-15-15-four-rooms.json'

    # file_dir = '2024-03-11--17-23_ALGS-2_RUNS-25_MAP-15-15-eight-rooms.json'
    # file_dir = '2024-03-11--18-47_ALGS-1_RUNS-25_MAP-15-15-eight-rooms.json'

    # file_dir = '2024-03-11--17-38_ALGS-2_RUNS-25_MAP-15-15-six-rooms.json'
    # file_dir = '2024-03-11--19-00_ALGS-1_RUNS-25_MAP-15-15-six-rooms.json'

    # file_dir = '2024-03-11--17-55_ALGS-2_RUNS-25_MAP-15-15-two-rooms.json'
    # file_dir = '2024-03-11--19-18_ALGS-1_RUNS-25_MAP-15-15-two-rooms.json'

    # show_results(file_dir=f'final_logs/lmapf/{file_dir}', lmapf=True)
    pass


if __name__ == '__main__':
    main()


