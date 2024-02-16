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

# from itertools import tee, combinations
#
# agents = [1,2,3]
# for a1, a2 in combinations(agents, 2):
#     print(a1, a2)
#
#
#
#
#
# def pairwise_list(input_list: list) -> list[tuple]:
#     def pairwise(iterable):
#         a, b = tee(iterable)
#         next(b, None)
#         return zip(a, b)
#     result = list(map(lambda x: (x[0], x[1]), pairwise(input_list)))
#     return result
#
#
# test_list = [0, 1, 2, 3, 4, 5]
# print(pairwise_list(test_list))

test_list = [0, 1, 2, 3, 4,]

print(5 < len(test_list))

# for i in range(0, 10):
#     print(i)



