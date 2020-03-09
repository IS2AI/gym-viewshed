"""
Created on Mon Feb  3 13:41:56 2020

@author: Daulet Baimukashev
"""


import numpy as np
from PIL import Image
import cv2
import time
import arcpy
from arcpy import env
from arcpy.sa import Viewshed2
#from arcpy.da import *
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt

env.overwriteOutput = True
env.scratchWorkspace = r"in_memory"
env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 18N")
env.geographicTransformations = "Arc_1950_To_WGS_1984_5; PSAD_1956_To_WGS_1984_6"
#env.parallelProcessingFactor = "200%"
env.processorType = "GPU"
env.gpuID = "0"
env.compression = "LZ77" #"LZ77" #"JPEG" # LZW
env.tileSize = "128 128"
env.pyramid = "PYRAMIDS -1 CUBIC LZ77 NO_SKIP"

class ViewshedBasicEnv(gym.Env):
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

        # input Raster
        self.city_array = np.array((Image.open(r"../data/total_7x7_correct_small.png").convert('L')), dtype=np.uint8)  #.resize((900,600))
        #self.city_array = np.zeros((800,800))
        #self.city_array[0:500,600:790] = 10
        self.im_height, self.im_width  = self.city_array.shape # reshape (width, height) [300,500] --> example: height = 500, width = 300
        self.input_raster = arcpy.NumPyArrayToRaster(self.city_array)
        # input shapefile
        self.shape_file = r"../data/input_shapefile/10/points_XYTableToPoint_second.shp"
        # viewshed params
        self.analysis_type = "FREQUENCY"
        self.analysis_method = "PERIMETER_SIGHTLINES"
        self.observer_height = 0
        self.vertical_lower_angle  = -90
        self.vertical_upper_angle = 90
        self.radius_is_3d = 'True'
        self.inner_radius = 0
        self.outer_radius = 300
        # init params
        self.init_x = 150 #self.im_width/2
        self.init_y = 150 #self.im_height/2
        self.init_observer_dist = 10 # how far init observer from each other
        self.init_azimuth1 = 0
        self.init_azimuth2 = 360
        # info extra about the env
        self.info = 0
        self.info_x = 0.0
        self.info_y = 0.0
        # observer params
        self.camera_number = 1
        self.action_number = 4
        self.delta_x = 15
        self.delta_y = 15
        # gym env params
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_width,self.im_height, 1), dtype = np.uint8)
        self.action_space = spaces.Discrete(4)
        self.state = np.zeros((self.im_height, self.im_width)) # self.city_array
        self.iteration = 0
        # reward
        self.ratio_threshhold = 0.05
        self.reward_good_step = 5
        self.reward_bad_step = -0.1
        self.max_iter = 400
        # rendering
        self.is_render = 'False'
        self.max_render = 100
        self.imshow_dt = 1000
        self.seed(0)
        print('init ViewshedBasicEnv successfully!')

    def step(self, action):
        #assert self.action_space.contains(action
        state, reward, done = self.act_discrete(self.input_raster, self.shape_file, action)
        self.iteration = self.iteration + 1
        # state two channel
        return state, reward, done

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def reset(self):
        self.reset_shapefile(self.shape_file)
        self.state = np.zeros((self.im_height, self.im_width)) # self.state
        self.iteration = 0
        return self.state

    def render(self, mode='human'):
        # to show
        if self.is_render == 'True' and self.iteration < self.max_render :
            print('render --- ratio --- ', self.info)
            self.show_image(self.state, self.imshow_dt)

    def close(self):
        pass

    def show_image(self,show_array,dt):

        show_array = show_array * 100
        show_array = Image.fromarray(show_array, 'L')
        show_array = np.array(show_array)
        #show_array = cv2.resize(show_array, (800,800), interpolation = cv2.INTER_AREA)
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", show_array)
        cv2.waitKey(dt)
        cv2.destroyAllWindows()

    def reset_shapefile(self, shape_file):

        #print('Reset init camera locations')
        fieldlist=['AZIMUTH1','AZIMUTH2','OFFSETA','RADIUS2']
        tokens=['SHAPE@X','SHAPE@Y']
        with arcpy.da.UpdateCursor(shape_file,tokens+fieldlist) as cursor:
            delta = -1
            X = self.init_observer_dist
            for row in cursor:
                delta += 1
                #print('hei')
                row[0]= 150#self.init_x + (delta%4)*X
                row[1]= 150#self.init_y + (delta//4)*X*1.5
                row[2]= self.init_azimuth1
                row[3]= self.init_azimuth2
                x = row[0]
                y = row[1]
                p = str(x) + " " + str(y)
                p_value = arcpy.GetCellValue_management(self.input_raster, p)
                #print('p_value', p_value)
                if int(p_value[0]) > 1:
                    row[4] = int(p_value[0])
                else:
                    row[4]=  int(p_value[0]) #0
                row[5]= self.outer_radius
                cursor.updateRow(row)
        del cursor

    def act_discrete(self, input_raster, shape_file, action):

        # this function needs to do:
        # map the "action" to CELL value update in shapefile (actions x observers)
        # action [0 ... N] --- > action type x observerN
        # here assumption is that action will be 1xD array for all N cameras, and should be interpreted as which action to which observer

        # for 1 camera
        action_type = action%self.action_number
        observer_n = action//self.action_number + 1
        #print('action', action) # [0 ... 5]
        #print('action_type',action_type) # [0 ... 5]
        #print('observer_n',observer_n ) # [1 ... ]
        # update shapefile
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

        if ratio > self.ratio_threshhold:
            reward = self.reward_good_step
        else:
            reward = self.reward_bad_step

        if self.iteration > self.max_iter or reward == self.reward_good_step:
            done = 1
        else:
            done = 0

        return next_state, reward, done

    def update_shapefile_discrete(self, shape_file, action_type, observer_n):

        delta_x = self.delta_x
        delta_y = self.delta_y

        if action_type == 0:
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
        elif action_type == 1:
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
        elif action_type == 2:
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
        elif action_type == 3:
            # move in y - delta down
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
        else:
            raise ValueError('Invalid action for viewshed_env')

        #update observer height
        fields = ['OFFSETA']
        tokens=['SHAPE@X', 'SHAPE@Y']
        with arcpy.da.UpdateCursor(shape_file,tokens+fields) as cursor:
            s = 0
            for row in cursor:
                s = s + 1
                if s == observer_n:
                    #print('set h', row[0])
                    x = row[0]
                    y = row[1]
                    p = str(x) + " " + str(y)
                    p_value = arcpy.GetCellValue_management(self.input_raster, p)
                    if int(p_value[0]) > 1:
                        row[2] = int(p_value[0])
                    else:
                        row[2] = int(p_value[0])
                    cursor.updateRow(row)
        del cursor

    def create_viewshed(self, input_raster, shape_file):

        analysis_type_ = self.analysis_type
        analysis_method_ = self.analysis_method
        radius_is_3d_ = self.radius_is_3d
        observer_height_ = self.observer_height
        vertical_lower_angle_ = self.vertical_lower_angle
        vertical_upper_angle_ = self.vertical_upper_angle
        inner_radius_ = self.inner_radius
        outer_radius_ = self.outer_radius
        start_t = time.time()

        outViewshed2 = Viewshed2(in_raster=input_raster, in_observer_features= shape_file, out_agl_raster= "", analysis_type= analysis_type_,
                                 vertical_error= 0, out_observer_region_relationship_table= "", refractivity_coefficient= 0.13,
                                 surface_offset= 0, observer_offset = 0, observer_elevation = 0, inner_radius= inner_radius_,
                                 outer_radius= "RADIUS2", inner_radius_is_3d = radius_is_3d_, outer_radius_is_3d = radius_is_3d_,
                                 horizontal_start_angle = "AZIMUTH1", horizontal_end_angle= "AZIMUTH2", vertical_upper_angle = vertical_upper_angle_,
                                 vertical_lower_angle= vertical_lower_angle_, analysis_method=analysis_method_)

        print('elapsed for viewshed', time.time() - start_t)
        output_array = arcpy.RasterToNumPyArray(outViewshed2) # output array -> each cell how many observer can see that pixel
        output_array[output_array == 255] = 0
        visible_points = output_array > 0
        visible_area = visible_points.sum()
        return output_array, visible_area
