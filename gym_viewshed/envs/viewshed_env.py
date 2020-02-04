import numpy as np
from PIL import Image
import csv

import arcpy
from arcpy import env
from arcpy.sa import Viewshed2
#from arcpy.da import *

import gym
from gym import error, spaces, utils
from gym.utils import seeding

#import torch

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
      self.city_array = np.array(self.city_image)
      self.input_rasterC = arcpy.NumPyArrayToRaster(self.city_array)
      
      self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_width,self.im_height, 1), dtype = np.uint8)
      self.action_space = spaces.Discrete(4)
      self.state = self.city_array # or init image
      
      self.points = [10,10,10]
      self.angle_start = 0
      self.angle_end = self.angle_start + 90
      
      self.seed(0)
      
    def step(self, action):
        assert self.action_space.contains(action)
        state = self.state
        
        self.act(self, action)
        
        done = 0
        done = bool(done)
        
        if not done:
            reward = 1.0
        else:
            reward = 0.0
        
        return np.array(state), reward, done, {}
    
    def seed(self, seed = None):
        self.np_random , seed = seeding.np_random()
        return [seed]
      
    def reset(self):
        self.state = self.city_array
        
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def act(self, action):
        self.create_points(action)
        output_viewshed = self.create_viewshed(self)    
        next_state = output_viewshed # ? 
        
        return next_state
    
    def create_points(self, action):
        # choose action and change the points location
        if action == 0:
            # rotate +5 deg
            # change the shape file
            self.angle_start = self.angle_start + 5
            self.angle_end = self.angle_start + 90
            pass
        elif action == 1:
            # rotate -5 deg
            self.angle_start = self.angle_start - 5
            self.angle_end = self.angle_start + 90
            pass
        elif action == 2:
            # move up x
            self.points=self.points+[5,0,0]
            pass
        elif action == 3:
            # move down x
            self.points=self.points-[5,0,0]
            pass
        elif action == 4:
            # move up y
            self.points=self.points+[0,5,0]
            pass
        elif action == 5:
            # move down y
            self.points=self.points-[0,5,0]
            pass
        
        fn = 'xyz'
        with open("../data/points.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows([str(fn)])
            writer.writerows(self.points)
    
    def create_viewshed(self):
        
        input_rasterC = self.input_rasterC
        # define the workspace
        env.workspace = "../data/raster/"
        env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 18N")
        env.geographicTransformations = "Arc_1950_To_WGS_1984_5; PSAD_1956_To_WGS_1984_6"
        env.overwriteOutput = True
    
        #create observer
        inCSV =   r'../data/points.csv'
        shapeSaveName = "camera_points"
        shape_file = arcpy.MakeXYEventLayer_management(table=inCSV, in_x_field="x", in_y_field="y", 
                                          out_layer=shapeSaveName, spatial_reference=arcpy.SpatialReference("WGS 1984 UTM Zone 18N"), 
                                          in_z_field="z")
        
        outViewshed2 = Viewshed2(input_rasterC, shape_file, "", "FREQUENCY", "", observer_elevation = 50, analysis_method="ALL_SIGHTLINES", 
                                 horizontal_start_angle= self.angle_start, horizontal_end_angle=self.angle_end, 
                                 vertical_lower_angle=-90, vertical_upper_angle=30, inner_radius=1, outer_radius=100)
        
        return outViewshed2