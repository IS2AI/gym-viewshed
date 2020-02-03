import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from PIL import Image


class ViewshedEnv(gym.Env):
    """
    Description:
        Viewshed analysis on raster data
        
    Source:
        ArcGIS function
    
    Observation:
        Type: Image
        
    Actions:
        Type: Discrete
        Num Action
        0   Rotate +5 deg 
        1   Rotate -5 deg
        2   Move +5 pixel
        3   Move -5 pixel
    
    Reward: 
        Reward 1 for game over
    
    Starting State: 
        Init image of the city
        
    Episode termination:
        Episode > 100
        
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self):
    
      # self.const = const
      self.city_image = Image.open("../data/sample_city.png").resize((500,500))
      self.im_width, self.im_height  = self.city_image.size
      self.city_array = np.array(city_image)
      
      self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_width,self.im_height, 3), dtype = np.uint8)
      self.action_space = spaces.Discrete(4)
      self.state = None # or init image
      
      self.seed(0)
      
    def step(self, action):
        assert self.action_space.contains(action)
        state = self.state
        
        done = bool(done)
        
        if not dot:
            reward = 1.0
        else:
            reward = 0.0
        
        return np.array(self.state), reward, done, {}
    
    def seed(self, seed = None):
        self.np_random , seed = seeding.np_random
        return [seed]
      
    def reset(self):
        self.state = # init image
        
    def render(self, mode='human'):
        pass
    def close(self):
        pass