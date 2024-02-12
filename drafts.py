# import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#
#    if terminated or truncated:
#       observation, info = env.reset()
#
#    env.render()
#
# env.close()


from itertools import tee


def pairwise_list(input_list: list) -> list[tuple]:
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    result = list(map(lambda x: (x[0], x[1]), pairwise(input_list)))
    return result


test_list = [0, 1, 2, 3, 4, 5]
print(pairwise_list(test_list))
