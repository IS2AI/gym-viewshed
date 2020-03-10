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

class ViewshedRandomEnv(gym.Env):
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
        # camera locations
        self.observer_locations = np.zeros((10,3)) 
        # viewshed params
        self.analysis_type = "FREQUENCY"
        self.analysis_method = "PERIMETER_SIGHTLINES"
        self.observer_height = 0
        self.vertical_lower_angle  = -90
        self.vertical_upper_angle = 90
        self.radius_is_3d = 'True'
        self.inner_radius = 0
        self.outer_radius = 1000
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
        print('init ViewshedRandomEnv successfully!')

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

    
    
    def generate_random_points(self):
        '''
        for camera 1:n
            find the closest point
            move to that point
        
        return updated_camera_points    
        '''
        camera_n = self.camera_number
        city_array = self.city_array 
        observer_locations = self.observer_locations
        
        for i in range(camera_n):
            curr_x = observer_locations[i,0]
            curr_y = observer_locations[i,1]
            curr_z = observer_locations[i,2]
            
            # find the closest point
            self.move_closest_point(city_array, i, curr_x, curr_y, curr_z)

    def move_closest_point(self, city_array, i, x, y, z):
        
        # given the point with the coor x,y,z find next positions
        # x,y,z
        is_done = 0
        count = 0
        move_step = 1
        
        while is_done==0:
            count = count + 1
            radius = 2 #count
            
            xy_coor = self.get_spiral(x,y,radius,move_step)
            h,w = xy_coor.shape
            
            for i in range(h):
            
                #next_x, next_y, next_z = find_closest_point(xy_coor)
               
                if height > 100:
                    observer_point.append(b[i])
                    
                    
            # find the next move ???
            city_array(x,y)
                     
            if count == self.camera_number:
                is_done = 1
                
        # next_x next_y next_z
        self.observer_locations[0,0] = next_x
        self.observer_locations[0,1] = next_y
        self.observer_locations[0,2] = next_z

    def get_spiral(self, x,y, radius, move_step):
        
        len_s = 2*radius+1
        xy_list = []
        
        temp_y = y-radius
        temp_x = x-radius
        
        # move right (bottom)
        x_coor = np.arange(temp_x, temp_x + len_s, move_step)
        y_coor = temp_y*np.ones(len(x_coor))
        xy_list.append([x_coor,y_coor])
        
        # move down (right)
        y_coor = np.arange(temp_y, temp_y + len_s, move_step)
        x_coor = (temp_x+len_s)*np.ones(len(x_coor))
        xy_list.append([x_coor,y_coor])    
        
        # move right (bottom)
        x_coor = np.arange(temp_x, temp_x + len_s, move_step)
        y_coor = (temp_y+len_s)*np.ones(len(x_coor))
        xy_list.append([x_coor,y_coor])

        # move down (left)
        y_coor = np.arange(temp_y, temp_y + len_s, move_step)
        x_coor = (temp_x)*np.ones(len(x_coor))
        xy_list.append([x_coor,y_coor])      
        
        # limit the xy_arr pairs [x,y] 
        xy_arr = np.asarray(xy_list)
        xy_arr = np.clip(xy_arr, 0, self.im_height)
                
        return xy_arr

    def act_discrete(self, input_raster, shape_file, action):

        # generate random points
        self.generate_random_points()
        # update shapefile
        self.update_shapefile_random(shape_file, self.observer_locations)
        # create the viewshed
        output_array, visible_area = self.create_viewshed(input_raster, shape_file)

        # next_state ?
        next_state = output_array
        ratio = visible_area/output_array.size

        # for rendering
        self.state = output_array
        self.info = ratio

        return next_state


    def update_shapefile_random(self, shape_file, observer_locations):
        '''
        For all observer points
        Update the shapefile
        '''
        
        #update observer height and x and y
        fields = ['OFFSETA']
        tokens=['SHAPE@X', 'SHAPE@Y']
        with arcpy.da.UpdateCursor(shape_file,tokens+fields) as cursor:
            s = -1
            for row in cursor:
                s = s + 1
                print('set xyz')
                row[0] = observer_locations[s,0]
                row[1] = observer_locations[s,1]
                row[2] = observer_locations[s,2]
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
