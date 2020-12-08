"""
Created on Mon Nov 23 2020
@author: Daulet Baimukashev
"""

import numpy as np
from PIL import Image
import cv2
import time
import copy
import arcpy
from arcpy import env
from arcpy.sa import Viewshed2

#from arcpy.da import *

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math

#env.scratchWorkspace = r"in_memory"
#   print('ClearWorkspaceCache_management: ', arcpy.ClearWorkspaceCache_management())
arcpy.ClearWorkspaceCache_management()

env.scratchWorkspace = r"in_memory"
#env.workspace = r"../data/space/"
#env.workspace = r"C:/Users/Akmaral/Desktop/coverage/test4/shape_file_gen/"

env.overwriteOutput = True
env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 18N")
env.geographicTransformations = "Arc_1950_To_WGS_1984_5; PSAD_1956_To_WGS_1984_6"
#env.parallelProcessingFactor = "200%"
env.processorType = "GPU"
env.gpuID = "0"
env.compression = "LZ77" #"LZ77" #"JPEG" # LZW
env.tileSize = "128 128"
env.pyramid = "PYRAMIDS -1 CUBIC LZ77 NO_SKIP"

# arcpy.Delete_management("in_memory")

class ViewshedCoverageEnv(gym.Env):
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
        0   Pan +5 deg
        1   Pan -5 deg
        2   Tilt +5 deg
        3   Tilt -5 deg
        4   Zoom +5 factor
        5   Zoom -5 factor

    Reward:
        Reward 1 for game over

    Starting State:
        Init image of the city

    Episode termination:
        Episode > 100

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # import image of city
        self.city_array = np.array((Image.open(r"../data/images/RasterAstanaCroppedZero.png")), dtype=np.uint16) #.resize((900,600))
        # self.city_array = self.city_array/100
        print('+++ ', np.max(np.max(self.city_array)), np.min(np.min(self.city_array)))

        self.city_array = self.city_array/100 - 285             # convert to meter
        print('Original Image: ', type(self.city_array), self.city_array.shape)

        # crop the image with center at camera
        self.camera_location = (3073, 11684, 350)   # x,y,z coordinate  #  (11685, 7074, 350) - RasterAstana.png
        # self.camera_location = (3073, 11684, 350)   # x,y,z coordinate  #  (11685, 7074, 350) - RasterAstana.png

        self.coverage_radius = 2000                 # .. km square from the center
        self.city_array = self.city_array[self.camera_location[1]-self.coverage_radius:self.camera_location[1]+self.coverage_radius,
                                        self.camera_location[0]-self.coverage_radius:self.camera_location[0]+self.coverage_radius]


        # resize the image
        # self.city_array =  self.city_array[2500:3500, 2500:3500]#np.resize(self.city_array, (1000,1000))

        # self.city_array_res =  self.city_array[0:1000, 0:1000]

        self.im_height, self.im_width  = self.city_array.shape # reshape (width, height) [300,500] --> example: height = 500, width = 300

        print('Cropped Image: ', type(self.city_array), self.city_array.shape)
        print('Range Image: ', np.min(self.city_array), np.max(self.city_array))

        # input raster
        self.input_raster = arcpy.NumPyArrayToRaster(self.city_array)
        # input shapefile
        self.shape_file = r"../data/input_shapefile/1/points_XYTableToPoint_second.shp"

        # CAMERA params
        self.camera_number = 1
        self.camera_location_cropped = (int(self.coverage_radius), int(self.coverage_radius), self.camera_location[2]-285)
        print('Camera Loc: ', self.camera_location_cropped)

        #
        self.max_distance_min_zoom = 100       # at min zoom - 20mm - the max distance 50
        self.max_distance_max_zoom = 4000     # at min zoom - 800mm - the max distance 2000

        # PTZ
        self.pan_pos = 0
        self.tilt_pos = -45
        self.zoom_pos = 20               # 0 - 20mm (min), 1 - 800 mm (max)

        self.delta_pan  = 5                # deg
        self.delta_tilt = 3                 # deg
        self.delta_zoom =  1.25              # 1.25x times

        self.horizon_fov = 21               # 21           # Field of View deg
        self.vertical_fov =  11.8           # 11.8        # Field of View deg
        self.zoom_distance = self.max_distance_min_zoom


        # VIEWSHED params
        self.init_x = self.camera_location_cropped[0]           # self.im_width/2 #310
        self.init_y = self.camera_location_cropped[1]           # self.im_height/2 #80
        self.observer_height = self.camera_location_cropped[2] + 5  # height

        self.analysis_type = "FREQUENCY"
        self.analysis_method = "PERIMETER_SIGHTLINES"
        self.azimuth1 = self.pan_pos - self.horizon_fov/2
        self.azimuth2 = self.pan_pos + self.horizon_fov/2
        self.vertical_lower_angle  = self.tilt_pos - self.vertical_fov/2
        self.vertical_upper_angle = self.tilt_pos + self.vertical_fov/2
        self.radius_is_3d = 'True'
        self.inner_radius = 0
        self.outer_radius = self.zoom_distance

        # GYM env params
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_width,self.im_height, 1), dtype = np.uint8)
        self.action_space = spaces.Discrete(6)  # 6 different actions
        self.state = np.zeros((self.im_height, self.im_width)) # self.city_Array

        # render
        self.max_render = 100
        self.is_render = 'True'
        self.iteration = 0
        self.info = 0
        self.info_x = 0.0
        self.info_y = 0.0
        self.seed(0)

        # reward
        self.ratio_threshhold = 0.02
        self.reward_good_step = 1
        self.reward_bad_step = -0.05
        self.max_iter = 200

        # input
        self.input_total_coverage = np.asarray(Image.open(r"../data/images/RasterTotalCoverage4.png"))
        #self.input_total_coverage = np.asarray(Image.open(r"../data/images/RasterTotalCoverage4Resized.png"))

        self.rad_matrix, self.angle_matrix = self.create_cartesian()

    def step(self, action):

        #assert self.action_space.contains(action)

        # this function needs to do:
        # map the "action" to CELL value update in shapefile (actions x observers)
        # action [0 ... N] --- > action type x observerN
        # here assumption is that action will be 1xD array for all N cameras, and should be interpreted as which action to which observer

        # for 1 camera
        action_type = action # %cameraN
        observer_n = self.camera_number #action//actionN + 1

        #print('action', action) # [0 ... 5]
        #print('action_type',action_type) # [0 ... 5]
        #print('observerN',observerN ) # [1 ... ]

        self.update_shapefile_discrete(self.shape_file, action_type, observer_n)
        # create the viewshed
        output_array, visible_area = self.create_viewshed(self.input_raster, self.shape_file)
        output_array2, visible_area2 = self.get_coverage_fast()
        self.testing_im = output_array2

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

        crossed_map = np.multiply(self.input_total_coverage,(output_array))
        crossed_points = (crossed_map > 0).astype(int)
        crossed_area = crossed_points.sum()

        reward = crossed_area

        # if ratio > self.ratio_threshhold:
        #     reward = self.reward_good_step + ratio*5
        # else:
        #     reward = self.reward_bad_step + ratio*5

        if self.iteration > self.max_iter:
            done = 1
        else:
            done = 0


        self.iteration = self.iteration + 1

        self.input_total_coverage = np.multiply(self.input_total_coverage,(1-output_array))

        next_state = np.stack((self.input_total_coverage, next_state), axis = 0)

        return next_state, reward, done

    def seed(self, seed = None):
        self.np_random , seed = seeding.np_random()
        return [seed]

    def reset(self):
        print('Env reset ...')
        self.reset_shapefile(self.shape_file)
        self.state = np.zeros((self.im_height, self.im_width)) # self.state
        self.iteration = 0
        next_state = np.stack((self.input_total_coverage, self.state), axis = 0)

        return next_state

    def render(self, mode='human'):

        mode = 0  # 0 -  black/white ; 1 -  rgb

        if mode == 1:
            city_gray = np.array(self.city_array, dtype=np.uint8)
            show_array = np.stack((city_gray,)*3, axis=-1)
            show_array[:,:,2] = self.state*255
            show_array = cv2.resize(show_array, (1000,1000), interpolation = cv2.INTER_AREA)
        else:
            show_array = np.array(self.state*255, dtype='uint8')
            show_array = cv2.resize(show_array, (1000,1000), interpolation = cv2.INTER_AREA)

        # if mode == 1:
        #     city_gray1 = np.array(self.city_array, dtype=np.uint8)
        #     show_array1 = np.stack((city_gray1,)*3, axis= -1)
        #     show_array1[:,:,2] = self.testing_im*255
        #     show_array1 = cv2.resize(show_array1, (1000,1000), interpolation = cv2.INTER_AREA)
        # else:
        #     show_array1 = self.testing_im
        #     print('******    ', np.max(np.max(self.testing_im)))
        #     show_array1 = cv2.resize(show_array1, (800,800), interpolation = cv2.INTER_AREA)


        # if self.is_render == 'True' and self.iteration < self.max_render :
            # print('render --- ratio --- ', self.info)

        # cv2.startWindowThread()
        # cv2.namedWindow("preview")
        # cv2.imshow("preview", show_array)
        # #cv2.imshow("GET COVERAGE", show_array1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        try:
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            cv2.imshow("preview", show_array)
            cv2.namedWindow("COVERAGE")

            #show_array1 = cv2.resize(self.input_total_coverage, (1000,1000), interpolation = cv2.INTER_AREA)
            #cv2.imshow("COVERAGE", show_array1)

            array = np.array(self.testing_im*255, dtype='uint8')
            show_array1 = cv2.resize(array, (1000,1000), interpolation = cv2.INTER_AREA)
            cv2.imshow("COVERAGE", show_array1)

            #cv2.imshow("COVERAGE", show_array1)

            cv2.waitKey(100)
            #if cv2.waitKey(1)& 0xFF == ord('q'):
            #    quit()

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            # quit()


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
                row[2]= self.azimuth1
                row[3]= self.azimuth2
                cursor.updateRow(row)
        del cursor

    def update_shapefile_discrete(self, shape_file, action_type, observer_n):

        # Type: Discrete
        # Num Action
        # 0   Pan +5 deg
        # 1   Pan -5 deg
        # 2   Tilt +5 deg
        # 3   Tilt -5 deg
        # 4   Zoom +5 factor
        # 5   Zoom -5 factor

        if action_type == 0:    # rotate + delta
            print('... pan right')
            # update camera/ptz setting
            self.pan_pos += self.delta_pan
            if self.pan_pos >= 360:
                self.pan_pos -= 360

        elif action_type == 1:    # rotate - delta deg
            print('... pan left')
            # update camera/ptz setting
            self.pan_pos -= self.delta_pan
            if self.pan_pos < 0:
                self.pan_pos += 360

        elif action_type == 2:    # tilt + deg
            print('... tilt up')
            # update camera/ptz setting
            self.tilt_pos += self.delta_tilt
            if self.tilt_pos > 20:
                self.tilt_pos = 20

        elif action_type == 3:    # tilt - deg
            print('... tilt down')
            # update camera/ptz setting
            self.tilt_pos -= self.delta_tilt
            if self.tilt_pos < -45:
                self.tilt_pos = -45


        elif action_type == 4:    # zoom + in
            print('... zoom in')
            # update camera/ptz setting
            self.zoom_pos *= self.delta_zoom

            self.horizon_fov /= self.delta_zoom
            self.vertical_fov /= self.delta_zoom
            self.zoom_distance *= self.delta_zoom

            # boundaries
            if self.zoom_pos > 800:
                self.zoom_pos = 800

            if self.horizon_fov < 0.5:
                self.horizon_fov = 0.5

            if self.vertical_fov < 0.3:
                self.vertical_fov = 0.3

            if self.zoom_distance > self.max_distance_max_zoom:
                self.zoom_distance = self.max_distance_max_zoom

        elif action_type == 5:    # zoom - out
            print('... zoom out')

            # update camera/ptz setting
            self.zoom_pos /= self.delta_zoom

            self.horizon_fov *= self.delta_zoom
            self.vertical_fov *= self.delta_zoom
            self.zoom_distance /= self.delta_zoom

            # boundaries
            if self.zoom_pos < 20:
                self.zoom_pos = 20

            if self.horizon_fov > 21:
                self.horizon_fov = 21

            if self.vertical_fov > 11.8:
                self.vertical_fov = 11.8

            if self.zoom_distance < self.max_distance_min_zoom:
                self.zoom_distance = self.max_distance_min_zoom
        else:
            pass
            print('No action done ..')


    def create_viewshed(self, input_raster, shape_file):

        # UPDATE viewshed params
        self.azimuth1 = self.pan_pos - self.horizon_fov/2
        if self.azimuth1 < 0:
            self.azimuth1 += 360
        self.azimuth2 = self.pan_pos + self.horizon_fov/2

        # second
        # self.azimuth2 = self.pan_pos - self.horizon_fov/2
        # self.azimuth2 = 90 - self.azimuth2
        # if self.azimuth2 < 0:
        #     self.azimuth2 += 360
        # self.azimuth1 = self.azimuth2 - self.horizon_fov

        # temp_angle = self.pan_pos
        # temp_angle = 90 - temp_angle
        # if temp_angle < 0:
        #     temp_angle += 360
        #
        # self.azimuth1 = temp_angle - self.horizon_fov/2
        # #self.azimuth1 = 90 - self.azimuth1
        # if self.azimuth1 < 0:
        #     self.azimuth1 += 360
        # self.azimuth2 = temp_angle + self.horizon_fov/2

        self.vertical_lower_angle  = self.tilt_pos - self.vertical_fov/2
        self.vertical_upper_angle = self.tilt_pos + self.vertical_fov/2
        self.outer_radius = self.zoom_distance


        # print('Elapsed time for viewshed: ', time.time() - start_t)
        print('1 - camera   : pan_pos {},  tilt_pos {} , zoom_pos {}, horizon_fov {}, vertical_fov {}, zoom_distance {}'.format(
                                    self.pan_pos, self.tilt_pos, self.zoom_pos, self.horizon_fov, self.vertical_fov, self.zoom_distance))
        print('2 - viewshed : azimuth1 {},  azimuth2 {} , vertical_lower_angle {}, vertical_upper_angle {}, outer_radius {}'.format(
                   self.azimuth1, self.azimuth2, self.vertical_lower_angle, self.vertical_upper_angle, self.outer_radius))

        start_t = time.time()

        #self.azimuth1 = 315 #int(input("s1 "))
        #self.azimuth2 = 45  #int(input("s2 "))
        # self.vertical_lower_angle = -90
        # self.vertical_upper_angle = 90

        outViewshed2 = Viewshed2(in_raster=self.input_raster, in_observer_features= self.shape_file, out_agl_raster= "", analysis_type= self.analysis_type,
                                 vertical_error= 0, out_observer_region_relationship_table= "", refractivity_coefficient= 0.13,
                                 surface_offset= 0, observer_offset = 0, observer_elevation = self.observer_height, inner_radius= self.inner_radius,
                                 outer_radius= self.outer_radius, inner_radius_is_3d = self.radius_is_3d, outer_radius_is_3d = self.radius_is_3d,
                                 horizontal_start_angle= self.azimuth1, horizontal_end_angle= self.azimuth2, vertical_upper_angle = self.vertical_upper_angle,
                                 vertical_lower_angle= self.vertical_lower_angle, analysis_method=self.analysis_method)

        # # # manual
        # outViewshed2 = Viewshed2(in_raster=self.input_raster, in_observer_features= self.shape_file, out_agl_raster= "", analysis_type= self.analysis_type,
        #                          vertical_error= 0, out_observer_region_relationship_table= "", refractivity_coefficient= 0.13,
        #                          surface_offset= 0, observer_offset = 0, observer_elevation = 70, inner_radius= 0,
        #                          outer_radius= 200, inner_radius_is_3d = self.radius_is_3d, outer_radius_is_3d = self.radius_is_3d,
        #                          horizontal_start_angle= 0, horizontal_end_angle= 360, vertical_upper_angle = 25.9,
        #                          vertical_lower_angle= -56, analysis_method=self.analysis_method)

        #print('--------------- finished -----------------')

        print('Elapsed time for viewshed: ', time.time() - start_t)

        # extract the array
        output_array = arcpy.RasterToNumPyArray(outViewshed2) # output array -> each cell how many observer can see that pixel

        # not visible cells will have value of zero
        output_array[output_array == 255] = 0

        visible_points = output_array > 0
        visible_area = visible_points.sum()

        print('visible_points ', visible_area)

        # save
        # im = Image.fromarray(output_array*255)
        # im.save("../data/images/RasterTotalCoverage4.png")

        return output_array, visible_area

    #
    # def get_coverage(self):
    #     start_t = time.time()
    #     output_array = np.zeros((self.im_height, self.im_width))
    #
    #     temp_angle = self.pan_pos
    #     # temp_angle = 450-temp_angle
    #     # if temp_angle >= 360:
    #     #     temp_angle -= 360
    #
    #
    #     # temp_angle = 90-self.pan_pos
    #     # if temp_angle < -180:
    #     #     temp_angle = 90 + (temp_angle + 180)
    #     #
    #     # print('test: ', temp_angle, self.pan_pos)
    #
    #
    #     # #self.azimuth1 = temp_angle - self.horizon_fov/2
    #     #self.azimuth1 = 90 - self.azimuth1
    #     #if self.azimuth1 < 0:
    #     #    self.azimuth1 += 360
    #     #self.azimuth2 = temp_angle + self.horizon_fov/2
    #
    #     horizon_start = temp_angle - self.horizon_fov/2
    #     horizon_end = temp_angle + self.horizon_fov/2
    #     if horizon_start < 0:
    #         horizon_start += 360
    #
    #     # if horizon_start <= -180:
    #     #     horizon_end = 180 + (horizon_start + 180)
    #     #
    #     # if horizon_end > 180:
    #     #     horizon_start = -180 + (horizon_end - 180)
    #
    #     vertical_start =  self.tilt_pos - self.vertical_fov/2
    #     vertical_end = self.tilt_pos + self.vertical_fov/2
    #
    #     if vertical_start < 0 and vertical_end < 0:
    #
    #         radius_inner = self.observer_height*math.tan(math.radians(90+vertical_start))
    #         radius_outer = self.observer_height*math.tan(math.radians(90+vertical_end))
    #         if radius_outer > self.zoom_distance:
    #             radius_outer = self.zoom_distance
    #
    #         # print('rad ---> ', radius_inner, radius_outer)
    #         # print('hor ---> ', horizon_start, horizon_end)
    #
    #         for i in range(1500, 2500):
    #             for j in range(1500, 2500):
    #
    #                 point_rad = math.sqrt((self.coverage_radius-i)**2 + (self.coverage_radius-j)**2)
    #                 #if i == self.coverage_radius:
    #                 #    point_angle = 0
    #                 #else:
    #                     #point_angle = 90-math.degrees(math.atan((self.coverage_radius-j)/(self.coverage_radius-i)))
    #                     # point_angle = math.degrees(math.atan2((self.coverage_radius-j),(i-self.coverage_radius)))
    #                     # if point_angle < 0:
    #                     #     point_angle += 360
    #
    #                 point_angle = math.degrees(math.atan2((self.coverage_radius-i),(j-self.coverage_radius)))
    #                 point_angle *= -1
    #                 point_angle += 90
    #
    #                 #point_angle += 90
    #                 #if point_angle > 360:
    #                 #    point_angle -= 360
    #
    #                 if point_angle < 0:
    #                     point_angle += 360
    #
    #                 inside_rad = radius_inner < point_rad < radius_outer
    #
    #                 # case 1
    #
    #                 if horizon_start < horizon_end:
    #                     output_array[i,j] = (horizon_start < point_angle and point_angle < horizon_end) and inside_rad
    #                 else:
    #                     output_array[i,j] = (horizon_start < point_angle or point_angle < horizon_end) and inside_rad
    #
    #                 #output_array[i,j] = point_angle > horizon_start #
    #                 # output_array[i,j] = (radius_inner < point_rad < radius_outer) and (horizon_start < point_angle < horizon_end)
    #
    #         #point_rad = np.sqrt((self.city_array-self.coverage_radius)**2 + (self.city_array-self.coverage_radius)**2)
    #         #point_angle = -np.degrees(np.arctan((self.coverage_radius - self.city_array)/((self.coverage_radius - self.city_array).transpose())))
    #
    #
    #         #output_array = point_rad > 2000
    #
    #
    #         #output_array = (radius_inner < point_rad).astype(int) * (radius_outer > point_rad).astype(int) * (horizon_start < point_angle).astype(int) * (point_angle < horizon_end).astype(int)
    #         print('Elapsed time for coverage: ', time.time() - start_t)
    #
    #         output_array = output_array.astype(int)
    #         print('*** ', type(output_array), output_array.shape)
    #
    #         visible_points = (output_array > 0).astype(int)
    #         visible_area = 0 # visible_points.sum()
    #
    #     else:
    #         visible_area = 0
    #
    #     return output_array, visible_area


    def create_cartesian(self):

        rad_matrix = np.zeros((self.im_height, self.im_width))
        angle_matrix = np.zeros((self.im_height, self.im_width))

        for i in range(self.im_height):
            for j in range(self.im_width):

                point_rad  = math.sqrt((self.coverage_radius-i)**2 + (self.coverage_radius-j)**2)

                point_angle = math.degrees(math.atan2((self.coverage_radius-i),(j-self.coverage_radius)))
                point_angle *= -1
                point_angle += 90
                if point_angle < 0:
                    point_angle += 360


                rad_matrix[i,j] = point_rad
                angle_matrix[i,j] = point_angle

        return rad_matrix, angle_matrix

    def get_coverage_fast(self):
        start_t = time.time()
        output_array = np.zeros((self.im_height, self.im_width))

        temp_angle = self.pan_pos
        # temp_angle = 450-temp_angle
        # if temp_angle >= 360:
        #     temp_angle -= 360


        # temp_angle = 90-self.pan_pos
        # if temp_angle < -180:
        #     temp_angle = 90 + (temp_angle + 180)
        #
        # print('test: ', temp_angle, self.pan_pos)


        # #self.azimuth1 = temp_angle - self.horizon_fov/2
        #self.azimuth1 = 90 - self.azimuth1
        #if self.azimuth1 < 0:
        #    self.azimuth1 += 360
        #self.azimuth2 = temp_angle + self.horizon_fov/2

        horizon_start = temp_angle - self.horizon_fov/2
        horizon_end = temp_angle + self.horizon_fov/2
        if horizon_start < 0:
            horizon_start += 360
        if horizon_end >= 360:
            horizon_end -= 360
            
        # if horizon_start <= -180:
        #     horizon_end = 180 + (horizon_start + 180)
        #
        # if horizon_end > 180:
        #     horizon_start = -180 + (horizon_end - 180)

        vertical_start =  self.tilt_pos - self.vertical_fov/2
        vertical_end = self.tilt_pos + self.vertical_fov/2

        if vertical_start < 0 and vertical_end < 0:

            radius_inner = self.observer_height*math.tan(math.radians(90+vertical_start))
            radius_outer = self.observer_height*math.tan(math.radians(90+vertical_end))
            if radius_outer > self.zoom_distance:
                radius_outer = self.zoom_distance

            # matrix
            rad_matrix, angle_matrix = self.rad_matrix, self.angle_matrix

            #inside_rad = radius_inner < rad_matrix and rad_matrix < radius_outer
            inside_rad = np.multiply( np.greater_equal(rad_matrix, radius_inner), np.greater_equal(radius_outer, rad_matrix))

            # if horizon_start < horizon_end:
            #     inside_angle = (horizon_start < point_angle and point_angle < horizon_end)
            # else:
            #     inside_angle = (horizon_start < point_angle or  point_angle < horizon_end)

            #
            if horizon_start < horizon_end:
                inside_angle = np.multiply(np.greater_equal(angle_matrix, horizon_start), np.greater_equal(horizon_end, angle_matrix))
            else:
                inside_angle = np.add(np.greater_equal(angle_matrix, horizon_start), np.greater_equal(horizon_end, angle_matrix))


            inside_sector = np.multiply(inside_rad, inside_angle)

            print('Here --- ', inside_rad.shape, inside_angle.shape, inside_sector.shape)
            print('2 - coverage : horizon_start {},  horizon_end {} , vertical_start {}, vertical_end {}, radius_inner{}, outer_radius {}'.format(
                   horizon_start, horizon_end, vertical_start, vertical_end, radius_inner, radius_outer))

            output_array = inside_sector
            print('Elapsed time for coverage: ', time.time() - start_t)

            output_array = output_array.astype(int)
            print('*** ', type(output_array), output_array.shape)

            visible_points = (output_array > 0).astype(int)
            visible_area = 0 # visible_points.sum()

        else:
            print('Tilt Angle is larger than zero !!!')
            visible_area = 0

        return output_array, visible_area
