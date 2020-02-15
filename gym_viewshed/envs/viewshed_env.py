"""
Created on Mon Feb  3 13:41:56 2020

@author: Daulet Baimukashev
"""

import numpy as np
from PIL import Image
import cv2

import arcpy
from arcpy import env
from arcpy.sa import Viewshed2
#from arcpy.da import *

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt

#import torch

env.overwriteOutput = True
env.workspace = r"C:/Users/Akmaral/Desktop/visibility/RL_visibility_analysis/data/input_raster"
env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 18N")
env.geographicTransformations = "Arc_1950_To_WGS_1984_5; PSAD_1956_To_WGS_1984_6"


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
        2   Move +5 pixel x
        3   Move -5 pixel x
        4   Move +5 pixel y 
        5   Move -5 pixel y 
    
    Reward: 
        Reward 1 for game over
    
    Starting State: 
        Init image of the city
        
    Episode termination:
        Episode > 100
        
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self):

        # inputs Raster and ShapeFile        
        #self.city_array = 255 - np.array((Image.open("D:/windows/dev/projects/Visibility_analysis/python/RL_visibility_analysis/data/sample_city_1.png").convert('L')), dtype=np.uint16)  #.resize((900,600))     
        self.city_array = 255 - np.array((Image.open("C:/Users/Akmaral/Desktop/visibility/RL_visibility_analysis/data/sample_city_1.png").convert('L')), dtype=np.uint16)  #.resize((900,600))             
        self.im_height, self.im_width  = self.city_array.shape # reshape (width, height) [300,500] --> example: height = 500, width = 300
        self.input_raster = arcpy.NumPyArrayToRaster(self.city_array)
        #print('city size', self.city_array.shape)
        self.shape_file = r"../data/input_shapefile/1/points_XYTableToPoint_second.shp" 
        # viewshed params
        self.info = 0
        self.info_x = 0.0
        self.info_y = 0.0
        self.init_x = 200 #self.im_width/2 #310
        self.init_y = 200 #self.im_height/2 #80
        self.init_azimuth1 = 0
        self.init_azimuth2 = 360
        self.analysis_type = "FREQUENCY"
        self.analysis_method = "ALL_SIGHTLINES"
        self.radius_is_3d = 'False'
        self.observer_height_ = 100         
        self.vertical_lower_angle  = -70
        self.vertical_upper_angle = 45
        self.inner_radius =  30
        self.outer_radius = 70
        
        # camera params
        self.camera_number = 1
        self.action_number = 6
        self.delta_theta = 8
        self.delta_x = 15
        self.delta_y = 15
        self.delta_fv = 80 # Field of View
        self.max_render = 20
        
        # gym env params
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_width,self.im_height, 1), dtype = np.uint8)
        self.action_space = spaces.Discrete(6)
        self.state = np.zeros((self.im_height, self.im_width)) # self.city_Array
      
        self.is_render = 'True'
        self.iteration = 0
        self.seed(0)
        
    def step(self, action):
        #assert self.action_space.contains(action)
        
        state, reward, done = self.act_discrete(self.input_raster, self.shape_file, action)
        
        self.iteration = self.iteration + 1
        
        return state, reward, done
    
    def seed(self, seed = None):
        self.np_random , seed = seeding.np_random() 
        return [seed]
      
    def reset(self):
        self.reset_shapefile(self.shape_file)
        self.state = np.zeros((self.im_height, self.im_width)) # self.state
        self.iteration = 0
        
        return self.state
        
    def render(self, mode='human'):
        #print('reward::: ', self.info)
        # how to show th array
        show_array = self.state * 100
        
        self.is_render = 'True'
        if self.is_render == 'True' and self.iteration < self.max_render :
            print('render --- ratio --- ', self.info)
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            cv2.imshow("preview", show_array)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
    
    def close(self):
        pass
    
    def reset_shapefile(self, shape_file):
        
        #print('Reset init camera locations')
        fieldlist=['AZIMUTH1','AZIMUTH2']
        tokens=['SHAPE@X','SHAPE@Y']
        with arcpy.da.UpdateCursor(shape_file,tokens+fieldlist) as cursor:
            for row in cursor:
                row[0]= self.init_x
                row[1]= self.init_y
                row[2]= self.init_azimuth1
                row[3]= self.init_azimuth2
                cursor.updateRow(row)        
        del cursor
        
    def act_discrete(self, input_raster, shape_file, action):
        
        # this function needs to do:
        # map the "action" to CELL value update in shapefile (actions x observers)
        # action [0 ... N] --- > action type x observerN
        # here assumption is that action will be 1xD array for all N cameras, and should be interpreted as which action to which observer
        
        # for 1 camera 
        action_type = action #%cameraN
        observer_n = self.camera_number #action//actionN + 1
        
        #print('action', action) # [0 ... 5]
        #print('action_type',action_type) # [0 ... 5]
        #print('observerN',observerN ) # [1 ... ]
        
        self.update_shapefile_discrete(shape_file, action_type, observer_n)
        # create the viewshed
        output_array, visible_area = self.create_viewshed(input_raster, shape_file)


        # interpret the viewshed output to some value - state , reward etc
        
        # next_state ?
        next_state = output_array
        ratio = visible_area/output_array.size

        # for rendering        
        self.state = output_array        
        self.info = ratio
        
        #reward ?
        #reward = visible_area/output_array.size
        
        #done ?
        
        if ratio > 0.02:
            reward = 10
        else:
            reward = 0.001
                
        if self.iteration > 500 or reward > 5:
            done = 1
        else:
            done = 0
            
        
        return next_state, reward, done

    def update_shapefile_discrete(self, shape_file, action_type, observer_n):
        
        delta_theta = self.delta_theta
        delta_x = self.delta_x
        delta_y = self.delta_y
        angle_fv = self.delta_fv # Field of View

        # matrix_action_observer [1,5] ---> for camera N, update 1 action
        if action_type == 0:
            # rotate + delta deg
            # change the shape file
            tokens=["AZIMUTH1", "AZIMUTH2"]
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor :
                    s = s + 1
                    if s == observer_n:
                        temp1_old = row[0]
                        temp1 =  temp1_old + delta_theta
                        if temp1 > 360:
                            temp1 = temp1 - 360
                        temp2 = temp1 + angle_fv             
                        #print('set A+')
                        row[0] =  temp1
                        row[1] =  temp2
                        cursor.updateRow(row)
                    
            del cursor        
        if action_type == 1:
            # rotate - delta deg
            # change the shape file
            tokens=["AZIMUTH1", "AZIMUTH2"]
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor :
                    s = s + 1
                    if s == observer_n:
                        temp1_old = row[0]
                        temp1 =  temp1_old - delta_theta
                        if temp1 < 0:
                            temp1 = temp1 + 360
                        temp2 = temp1 + angle_fv
                        #print('set A-')
                        row[0] =  temp1
                        row[1] =  temp2
                        cursor.updateRow(row)
            del cursor 
        if action_type == 2:
            # move in x -> + delta right
            # change the shape file
            tokens=['SHAPE@X']
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor:
                    s = s + 1
                    if s == observer_n:
                        #print('set x+')
                        row[0]= row[0] + delta_x
                        if row[0] >= self.im_width:
                            row[0] = self.im_width - 1 
                            #print('wall right')
                        self.info_x = row[0]
                        cursor.updateRow(row)
            del cursor
        if action_type == 3:
            # move in x <- - delta left
            # change the shape file
            tokens=['SHAPE@X']
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor:
                    s = s + 1
                    if s == observer_n:
                        #print('set x-')
                        row[0]= row[0] - delta_x
                        if row[0] <= 0:
                            row[0] = 1 
                            #print('wall left')
                        self.info_x = row[0]
                        cursor.updateRow(row)
            del cursor            
        if action_type == 4:
            # move in y + delta up
            # change the shape file
            tokens=['SHAPE@Y']
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor:
                    s = s + 1
                    if s == observer_n:
                        #print('set y+',row[0])
                        row[0]= row[0] + delta_y
                        #print('after plus', row[1])
                        if row[0] >= self.im_height:
                            row[0] = self.im_height - 1 
                            #print('wall up')
                        self.info_y = row[0]
                        cursor.updateRow(row)
            del cursor
        if action_type == 5:
            # move in y - delta up
            # change the shape file
            tokens=['SHAPE@Y']
            with arcpy.da.UpdateCursor(shape_file,tokens) as cursor:
                s = 0
                for row in cursor:
                    s = s + 1
                    if s == observer_n:
                        #print('set y-', row[0])
                        row[0]= row[0] - delta_y
                        if row[0] <= 0:
                            row[0] = 1 
                            #print('wall down')
                        self.info_y = row[0]
                        cursor.updateRow(row)
            del cursor

    def create_viewshed(self, input_raster, shape_file):
        # define the workspace
        #env.overwriteOutput = True
        #env.workspace = r"D:/windows/dev/projects/Visibility_analysis/python/RL_visibility_analysis/data/input_raster"
        #env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 18N")
        #env.geographicTransformations = "Arc_1950_To_WGS_1984_5; PSAD_1956_To_WGS_1984_6"
        
        analysis_type_ = self.analysis_type 
        analysis_method_ = self.analysis_method 
        radius_is_3d_ = self.radius_is_3d
        observer_height_ = self.observer_height_
        vertical_lower_angle_ = self.vertical_lower_angle 
        vertical_upper_angle_ = self.vertical_upper_angle 
        inner_radius_ = self.inner_radius 
        outer_radius_ = self.outer_radius
        
        # import raster
        #input_rasterC 
        #import observer
        #shape_file 
        
        outViewshed2 = Viewshed2(in_raster=input_raster, in_observer_features= shape_file, out_agl_raster= "", analysis_type= analysis_type_,
                                 vertical_error= 0, out_observer_region_relationship_table= "", refractivity_coefficient= 0.13,
                                 surface_offset= 0, observer_offset = 0, observer_elevation = observer_height_, inner_radius= inner_radius_,
                                 outer_radius= outer_radius_, inner_radius_is_3d = radius_is_3d_, outer_radius_is_3d = radius_is_3d_,
                                 horizontal_start_angle= "AZIMUTH1", horizontal_end_angle= "AZIMUTH2", vertical_upper_angle = vertical_upper_angle_, 
                                 vertical_lower_angle= vertical_lower_angle_, analysis_method=analysis_method_)
    
        output_array = arcpy.RasterToNumPyArray(outViewshed2) # output array -> each cell how many observer can see that pixel
        output_array[output_array == 255] = 0
                
        visible_points = output_array > 0
        visible_area = visible_points.sum()
        
        return output_array, visible_area
