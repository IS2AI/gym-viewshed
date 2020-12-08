from gym.envs.registration import register
# register(
#     id='viewshed-v0',
#     entry_point='gym_viewshed.envs:ViewshedEnv',
# )
# register(
#     id='viewshed-v1',
#     entry_point='gym_viewshed.envs:ViewshedBasicEnv',
# )
# register(
#     id='viewshed-v2',
#     entry_point='gym_viewshed.envs:ViewshedRandomEnv',
# )
# register(
#     id='viewshed-v3',
#     entry_point='gym_viewshed.envs:ViewshedGreedyEnv',
# )
# register(
#     id='viewshed-v4',
#     entry_point='gym_viewshed.envs:ViewshedCoverageEnv',
# )
register(
    id='viewshed-v5',
    entry_point='gym_viewshed.envs:VectorCoverageEnv',
)
