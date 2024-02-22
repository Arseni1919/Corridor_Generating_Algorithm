from functions import *


def do_the_animation(info):
    img_np = info['img_np']
    agents = info['agents']
    max_time = info['max_time']
    img_dir = info['img_dir']
    n_agents = len(agents)
    i_agent = agents[0]

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    field = img_np * -1
    ax.imshow(field, origin='lower')

    others_y_list, others_x_list, others_cm_list = [], [], []
    for agent in agents:
        curr_node = agent.start_node
        others_y_list.append(curr_node.y)
        others_x_list.append(curr_node.x)
        others_cm_list.append(get_color(agent.num))
    scat1 = ax.scatter(others_y_list, others_x_list, s=100, c='k')
    scat2 = ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))

    agent_scat1 = ax.scatter([i_agent.start_node.y], [i_agent.start_node.x], s=120, c='w')
    agent_scat2 = ax.scatter([i_agent.start_node.y], [i_agent.start_node.x], s=70, c='r')

    def update(frame):
        # for each frame, update the data stored on each artist.
        fr_y_list, fr_x_list = [], []
        for agent in agents:
            fr_node = agent.path[frame]
            fr_y_list.append(fr_node.y)
            fr_x_list.append(fr_node.x)
        # update the scatter plot:
        data = np.stack([fr_y_list, fr_x_list]).T
        scat1.set_offsets(data)
        scat2.set_offsets(data)

        fr_i_node = i_agent.path[frame]
        data = np.stack([[fr_i_node.y], [fr_i_node.x]]).T
        agent_scat1.set_offsets(data)
        agent_scat2.set_offsets(data)

        return scat1, scat2
    ani = animation.FuncAnimation(fig=fig, func=update, frames=max_time, interval=250)
    ani.save(filename=f"../videos/{n_agents}_agents_in_{img_dir[:-4]}.mp4", writer="ffmpeg")
    plt.show()

