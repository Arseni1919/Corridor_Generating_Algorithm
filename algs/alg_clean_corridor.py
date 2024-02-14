import matplotlib.pyplot as plt
from collections import deque
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


def clean_corridor(
        next_agent,
        input_corridor: List[Node],
        agents_in_corridor: list,
        input_occupied_nodes: List[Node],
        l_agents: list,
        node_name_to_agent_dict: dict,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        img_np: np.ndarray,
        # to_render: bool = False,
        to_render: bool = True,
        curr_iteration=0
) -> Tuple[bool, dict | None]:

    assert len(input_corridor) > 1
    occupied_nodes = input_occupied_nodes[:]
    occupied_nodes.append(next_agent.curr_node)
    corridor: List[Node] = input_corridor[1:]

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_info = {'img_np': img_np, 'agents': l_agents, 'corridor': corridor,
                     'i_agent': next_agent, 'to_title': 'from clean_corridor',
                     'i': curr_iteration, 'n_agents': len(l_agents),
                     'occupied_nodes': occupied_nodes}
        plot_step_in_env(ax[0], plot_info)
        plt.show()
        plt.close()

    alt_paths_dict = {}
    return False, None


def get_path_through_corridor() -> List[Node]:
    pass









































