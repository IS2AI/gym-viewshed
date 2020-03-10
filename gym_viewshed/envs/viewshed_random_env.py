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
import math

env.scratchWorkspace = r"in_memory"
env.overwriteOutput = True
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

        #input Raster
        self.city_array = np.array((Image.open(r"../data/images/total_city7_nearest.png").convert('L')), dtype=np.uint8)  #.resize((900,600))
        #self.city_array = np.ones((800,800))
        #self.city_array[0:500,600:790] = 10
        self.im_height, self.im_width  = self.city_array.shape # reshape (width, height) [300,500] --> example: height = 500, width = 300
        self.input_raster = arcpy.NumPyArrayToRaster(self.city_array)
        # input shapefile
        self.shape_file = r"../data/input_shapefile/10/points_XYTableToPoint_second.shp"
        # camera locations
        self.a = 500*np.repeat(np.arange(10), 3)
        self.observer_locations = self.a.reshape(10,3)   #10*np.zeros((10,3))
        # viewshed params
        self.analysis_type = "FREQUENCY"
        self.analysis_method = "PERIMETER_SIGHTLINES"
        self.observer_height = 0
        self.vertical_lower_angle  = -90
        self.vertical_upper_angle = 90
        self.radius_is_3d = 'True'
        self.inner_radius = 0
        self.outer_radius = 750
        # init params
        self.init_x = 1000 #self.im_width/2
        self.init_y = 1000 #self.im_height/2
        self.init_observer_dist = 500 # how far init observer from each other
        self.init_azimuth1 = 0
        self.init_azimuth2 = 360
        # info extra about the env
        self.info = 0.0         # ratio
        self.iteration = 0
        self.state = np.zeros((self.im_height, self.im_width))
        # observer params
        self.camera_number = 10
        # search parameter
        self.radius = 1
        self.radius_delta = 1
        self.move_step = 1
        self.min_height = 20
        # rendering
        self.is_render = 'True'
        self.max_render = 100
        self.imshow_dt = 0
        self.seed(0)
        print('init ViewshedRandomEnv successfully!')

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def reset(self):
        self.reset_shapefile(self.shape_file)
        self.state = np.zeros((self.im_height, self.im_width))
        self.iteration = 0
        return self.state

    def reset_shapefile(self, shape_file):

        #print('Reset init camera locations')
        fieldlist=['AZIMUTH1','AZIMUTH2','OFFSETA','RADIUS2']
        tokens=['SHAPE@X','SHAPE@Y']
        with arcpy.da.UpdateCursor(shape_file,tokens+fieldlist) as cursor:
            delta = -1
            X = self.init_observer_dist
            for row in cursor:
                delta += 1
                row[0]= self.init_x + (delta%self.camera_number)*X
                row[1]= self.init_y + (delta//self.camera_number)*X*1.5
                row[2]= self.init_azimuth1
                row[3]= self.init_azimuth2
                x = row[0]
                y = row[1]
                p = str(x) + " " + str(y)
                p_value = arcpy.GetCellValue_management(self.input_raster, p)
                row[4] = int(p_value[0])
                row[5]= self.outer_radius
                cursor.updateRow(row)
        del cursor

    def render(self, mode='human'):
        # to show
        if self.is_render == 'True' and self.iteration < self.max_render :
            print('render --- ratio --- ', self.info)
            self.show_image(self.state, self.imshow_dt)

    def close(self):
        pass

    def show_image(self,show_array,dt):

        show_array = show_array * 20
        show_array = Image.fromarray(show_array, 'L')
        show_array = np.array(show_array)
        show_array = cv2.resize(show_array, (800,800), interpolation = cv2.INTER_AREA)
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", show_array)
        cv2.waitKey(dt & 0xFF)
        cv2.destroyAllWindows()


    ###
    ### STEP
    def step(self):

        self.iteration = self.iteration + 1
        # move all observer to closest point
        self.moveto_closest_point()
        # update shapefile
        self.update_shapefile_random(self.shape_file, self.observer_locations)
        # create the viewshed
        output_array, ratio = self.create_viewshed(self.input_raster, self.shape_file)

        # for rendering
        self.state = output_array
        self.info = ratio

    def moveto_closest_point(self):

        # given the point with the coor x,y,z find next positions
        is_done = 0
        n = -1
        move_step = self.move_step
        min_height = self.min_height

        while is_done==0:
            n = n + 1
            radius = self.radius
            is_found = 0

            y = self.observer_locations[n,0]
            x = self.observer_locations[n,1]
            z = self.observer_locations[n,2]

            while is_found == 0:
                yx_coor = self.get_spiral(y,x,radius,move_step)
                h,w = yx_coor.shape
                observer_points = []
                for i in range(h):
                    observer_height = self.city_array[yx_coor[i][0], yx_coor[i][1]]
                    if observer_height > min_height:
                        observer_distance = math.sqrt((y-yx_coor[i][0])**2 + (x-yx_coor[i][1])**2)
                        observer_points.append([yx_coor[i][0], yx_coor[i][1], observer_height, observer_distance])

                if len(observer_points) > 0:
                    is_found = 1
                    # sort by h and get first row
                    observer_points = sorted(observer_points, key=lambda l:l[3], reverse=False)
                    # or random
                    next_y = observer_points[0][0]
                    next_x = observer_points[0][1]
                    next_z = observer_points[0][2]

                radius = radius + self.radius_delta

            # next_x next_y next_z
            self.observer_locations[n,0] = next_y
            self.observer_locations[n,1] = next_x
            self.observer_locations[n,2] = next_z

            if n == self.camera_number-1:
                is_done = 1

    def get_spiral(self, y, x, radius, move_step):
        
        yx_list = []
        temp_yi = y-radius
        temp_xi = x-radius
        temp_yf = temp_yi + (2*radius+1)
        temp_xf = temp_xi + (2*radius+1) 

        # move right (up)
        x_coor = np.arange(temp_xi, temp_xf+1, move_step)  # +1 to compensate the np.arange below
        y_coor = temp_yi*np.ones(len(x_coor))
        for i in range(len(x_coor)):
            yx_list.append([y_coor[i],x_coor[i]])

        # move down (right)
        y_coor = np.arange(temp_yi, temp_yf+1, move_step)
        x_coor = temp_xf*np.ones(len(y_coor))
        for i in range(len(y_coor)):
            yx_list.append([y_coor[i],x_coor[i]])

        # move right (bottom)
        x_coor = np.arange(temp_xi, temp_xf+1, move_step)
        y_coor = temp_yf*np.ones(len(x_coor))
        for i in range(len(x_coor)):
            yx_list.append([y_coor[i],x_coor[i]])

        # move down (left)
        y_coor = np.arange(temp_yi, temp_yf+1, move_step)
        x_coor = temp_xi*np.ones(len(y_coor))
        for i in range(len(y_coor)):
            yx_list.append([y_coor[i],x_coor[i]])

        # limit the xy_arr pairs [x,y]
        yx_arr = np.asarray(yx_list, dtype=np.int16)
        yx_arr = np.clip(yx_arr, 0, self.im_height)

        return yx_arr

    def update_shapefile_random(self, shape_file, observer_loc):
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
                row[0] = observer_loc[s,1] #observer_locations[s,0]
                row[1] = self.im_height - observer_loc[s,0] - 1 #observer_locations[s,1]
                row[2] = observer_loc[s,2] 
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
                                 surface_offset= 0, observer_offset = 0, observer_elevation = "OFFSETA", inner_radius= inner_radius_,
                                 outer_radius= "RADIUS2", inner_radius_is_3d = radius_is_3d_, outer_radius_is_3d = radius_is_3d_,
                                 horizontal_start_angle = "AZIMUTH1", horizontal_end_angle= "AZIMUTH2", vertical_upper_angle = vertical_upper_angle_,
                                 vertical_lower_angle= vertical_lower_angle_, analysis_method=analysis_method_)

        print('elapsed for viewshed', time.time() - start_t)
        output_array = arcpy.RasterToNumPyArray(outViewshed2) # output array -> each cell how many observer can see that pixel
        output_array[output_array == 255] = 0
        visible_points = output_array > 0
        visible_area = visible_points.sum()
        ratio = visible_area/output_array.size

        return output_array, ratio
