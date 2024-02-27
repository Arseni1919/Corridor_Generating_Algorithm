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
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_sr(ax[0], logs_dict)
            plot_time_metric_cactus(ax[1], logs_dict)

        plt.show()


def main():
    file_dir = '2024-02-27--15-04_ALGS-2_RUNS-5_MAP-room-32-32-4.json'
    show_results(file_dir=f'logs_for_plots/{file_dir}')

    # LMAPF
    # file_dir = ''

    # SACG
    # file_dir = ''

    # show_results(file_dir=f'final_logs/{file_dir}')


if __name__ == '__main__':
    main()


