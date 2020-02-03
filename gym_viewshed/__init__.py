from gym.envs.registration import register

register(
    id='viewshed-v0',
    entry_point='gym_viewshed.envs:ViewshedEnv',
)