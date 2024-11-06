0#!/usr/bin/env python3


################################################################################
import sys
sys.path.append('../')
from datetime import timedelta
import random
from datetime import datetime
from os import path
import os.path
import cv2
import pyds
import numpy as np
from   common.FPS import GETFPS
from   common.bus_call import bus_call
from   common.is_aarch_64 import is_aarch64
import platform
import math
import time
from ctypes import *
from gi.repository import GLib
from gi.repository import GObject, Gst
import configparser
import gi
import cupy as cp
import ctypes
import sys
import os
from database_entry import insert_data


os.environ['CUDA_MODULE_LOADING'] = '1'
gi.require_version('Gst', '1.0')
track_id_to_color = {}
# Initialize or load the class_colors dictionary
class_colors = {}
fps_streams = {}
frame_count = {}
saved_count = {}
global PGIE_CLASS_ID_VEHICLE
PGIE_CLASS_ID_VEHICLE = 1
global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 0

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 1
PGIE_CLASS_ID_PERSON = 0
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["person","vehicle"]
sgie_classes_str = ["bigbus", "car", "heavytruck", "lighttruck",
                    "microbus", "midtruck", "minibus", "threewheeler", "utility", "motorbike"]
MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4
SGIE_UNIQUE_ID = 5

global src_names
src_names = []
global cap_time
cap_time = []
global qTime
qTime = []
global dateS
dateS = []


obj_name = None
obj_name_original = None
osd_enable = True
save_image = True
sgie_enable = True
classification_threshold = 0.51
gamma_correction = False
# Define the parameters for exposure, brightness, contrast, and gamma
enable_brightness = True
enable_contrast = True
enable_saturation = True
enable_exposure = True
enable_gamma = True

exposure = 1.9  # Adjust as needed
brightness = 70  # Adjust as needed
contrast = 0.9 # Adjust as needed

# analytics_cfg_path = "/home/sigmind/analytics_tool_new_2/analytics_tool_sabbir/resources/cfg/config_nvdsanalytics_htpa.txt"
# pgie_cfg_path = "./deepstream_yolo4/config_infer_primary_yoloV4.txt"


transfer_learn_path = "/home/sigmind/deepstream_sdk_v6.3.0_x86_64/opt/nvidia/deepstream/deepstream-6.3/sources/apps/deepstream_python_apps/apps/deepstream-imagedata-multistream/transfer"
# nvanlytics_src_pad_buffer_probe  will extract metadata received on nvtiler sink pad
# and update params for drawing rectangle, object information etc.


def nvanalytics_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0
            
        }
        #print("#"*50)
        while l_obj:
            try:
                # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                # The casting is done by pyds.NvDsObjectMeta.cast()
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            #obj_counter[obj_meta.class_id] += 1
            l_user_meta = obj_meta.obj_user_meta_list
            # Extract object level meta data from NvDsAnalyticsObjInfo
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(
                            user_meta.user_meta_data)
                        #if user_meta_data.lcStatus:
                        #print("Object {0} line crossing status: {1}".format(obj_meta.object_id, user_meta_data.lcStatus))
                except StopIteration:
                    break

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Get meta data from NvDsAnalyticsFrameMeta
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(
                        user_meta.user_meta_data)
                
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def class_id_to_ctg_id(class_id):
    if class_id == 0:
        ctg_id = 4
    elif class_id == 9:
        ctg_id = 5
    elif class_id == 10:
        ctg_id = 8
    elif class_id == 8:
        ctg_id = 7
    elif class_id == 4:
        ctg_id = 9
    elif class_id == 6:
        ctg_id = 10
    elif class_id == 1:
        ctg_id = 11
    elif class_id == 3:
        ctg_id = 12
    elif class_id == 5:
        ctg_id = 13
    elif class_id == 2:
        ctg_id = 14
    elif class_id == 7:
        ctg_id = 3
    else:
        ctg_id = 15

    return ctg_id

def convertURSTP_to_bdvehiclenet(obj_name_original):
    if obj_name_original == "1":
        return "0"
    elif obj_name_original == "2":
        return "12"
    elif obj_name_original == "3":
        return "13"
    elif obj_name_original == "4":
        return "14"
    elif obj_name_original == "5":
        return "14"
    elif obj_name_original == "6":
        return "15"
    elif obj_name_original == "7":
        return "11"
    elif obj_name_original == "8":
        return "9"
    elif obj_name_original == "9":
        return "10"
    elif obj_name_original == "10":
        return "8"
    elif obj_name_original == "11":
        return "8"
    elif obj_name_original == "12":
        return "6"
    elif obj_name_original == "13":
        return "7"
    elif obj_name_original == "14":
        return "3"
    elif obj_name_original == "15":
        return "5"
    elif obj_name_original == "16":
        return "4"
    elif obj_name_original == "17":
        return "4"
    elif obj_name_original == "18":
        return "5"
    elif obj_name_original == "19":
        return "5"
    elif obj_name_original == "20":
        return "3"
    elif obj_name_original == "21":
        return "2"
    elif obj_name_original == "22":
        return "1"
    elif obj_name_original == "23":
        return "1"
    elif obj_name_original == "24":
        return "1"
    elif obj_name_original == "25":
        return "16"
    else:
        return None
    
def get_time_interval(time_interval_code):
    hour = int(time_interval_code[:2])
    quarter = int(time_interval_code[2])

    if quarter == 1:
        start_time = f"{hour:02d}:00"
    elif quarter == 2:
        start_time = f"{hour:02d}:15"
    elif quarter == 3:
        start_time = f"{hour:02d}:30"
    elif quarter == 4:
        start_time = f"{hour:02d}:45"

    return f"{start_time}"


def convert_gst_timestampp_to_string(timestamp, time_to_add):
    # Convert the timestamp to seconds and nanoseconds
    seconds = timestamp // Gst.SECOND
    nanoseconds = timestamp % Gst.SECOND

    # Convert the timestamp to the desired format
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    milliseconds = nanoseconds // Gst.MSECOND

    # Format the timestamp as a string
    timestamp_str = "{:02d}:{:02d}:{:02d}:{:03d}".format(
        hours % 24, minutes, seconds, milliseconds)

    return timestamp_str


def convert_gst_timestamp_to_string(timestamp, time_to_add):
    # Convert the timestamp to seconds and nanoseconds
    seconds = timestamp // Gst.SECOND
    nanoseconds = timestamp % Gst.SECOND

    # Convert the timestamp to the desired format
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    milliseconds = nanoseconds // Gst.MSECOND

    # Format the timestamp as a string
    timestamp_str = "{:02d}:{:02d}:{:02d}:{:03d}".format(
        hours % 24, minutes, seconds, milliseconds)

    # Parse the time_to_add string
    time_parts = time_to_add.split(":")
    time_hours = int(time_parts[0])
    time_minutes = int(time_parts[1])

    # Add the time_to_add to the timestamp
    hours = (hours + time_hours) % 24
    minutes = (minutes + time_minutes) % 60

    # Format the final timestamp with the added time
    final_timestamp = "{:02d}:{:02d}:{:02d}:{:03d}".format(
        hours, minutes, seconds, milliseconds)

    return final_timestamp


# Function to generate unique colors for each class ID and update the colors dictionary



def generate_class_colors(num_classes, colors):
    random.seed(42)  # For reproducibility, change the seed as needed
    print(f"numm:{num_classes}")
    for class_id in range(0, num_classes + 1):
        # Generate a random color for each class ID if it doesn't already exist
        if class_id not in colors:
            # r, g, b = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
            # Generate random red, green, and blue values.
            red = random.uniform(0.0, 1.0)
            green = random.uniform(0.0, 1.0)
            blue = random.uniform(0.0, 1.0)

            # Make sure the color is not pink or red.
            while red < 0.75 and green < 0.75:
                red = random.uniform(0.5, 1.0)
                green = random.uniform(0.5, 1.0)
            bg_color = pyds.NvOSD_ColorParams()
            bg_color.red = red
            bg_color.green = green
            bg_color.blue = blue
            bg_color.alpha = 0.38  # Set alpha as needed
            colors[class_id] = bg_color
            print(f" For {class_id}, color is {bg_color}")

# Generate unique colors for each class ID and update the class_colors dictionary
generate_class_colors(23, class_colors)



def tiler_sink_pad_buffer_probe(pad, info, u_data):
    got_visual = False
    frame_number = 0
    save_image = False
    
    num_rects = 0
    # Get the current date
    current_date = datetime.now().date()
    current_date_str = current_date.strftime("%Y-%m-%d")
    frames_folder = "/mnt/sde1/data/images"
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
        print(f"frames_folder '{frames_folder}' created.")
    annotations_folder = "/mnt/sde1/data/labels"
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
        print(f"annotations_folder '{annotations_folder}' created.")
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    timestampp = gst_buffer.pts

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_lines = 2
        
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        class_zero_objects = []
        other_objects = []


        
        
        # save_image = True
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0
        }
        annotation_data = []

        obj_counter = 0

        new_width = 640
        new_height = 480
       
        # Create dummy owner object to keep memory for the image array alive
        owner = None
        # Getting Image data using nvbufsurface
        # the input should be address of buffer and batch_id
        # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
            hash(gst_buffer), frame_meta.batch_id)
        # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object, ctypes.c_char_p]
        # Get pointer to buffer and create UnownedMemory object from the gpu buffer
        c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
        # Create MemoryPointer object from unownedmem, at index 0
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        # Create cupy array to access the image data. This array is in GPU buffer
        n_frame_gpu = cp.ndarray(
            shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
        # Initialize cuda.stream object for stream synchronization
        # Use null stream to prevent other cuda applications from making illegal memory access of buffer
        stream = cp.cuda.stream.Stream(null=True)

        
        # convert the array into cv2 default color format
        frame_copy = cv2.cvtColor(n_frame_gpu.get(), cv2.COLOR_RGBA2BGRA)
        h, w, _ = frame_copy.shape
        width_ratio = new_width / w
        height_ratio = new_height / h
        
        frame_resized = cv2.resize(frame_copy, (new_width, new_height))


        obj_name_original = None
        bdvehiclenet_classid = None
        detections = []
      

###########
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                obj_meta.rect_params.has_bg_color = 1
                bg_color = pyds.NvOSD_ColorParams()
          
                obj_track_id = obj_meta.object_id
                alpha = 0.5

                
               

           

                obj_meta.rect_params.bg_color.alpha = 0.5

                
                obj_meta.text_params.font_params.font_size = 12
                obj_meta.text_params.set_bg_clr = 1

        
                obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.4)

                obj_meta.text_params.font_params.font_name = "Arial Bold"

                obj_meta.text_params.display_text = f"ID: {obj_meta.object_id}"
                
       
            except StopIteration:
                break
            if obj_meta.class_id == 78:  # pedestrian
                rect_params_P = obj_meta.rect_params
                width = int(rect_params_P.width)
                height = int(rect_params_P.height)
                # Calculate the aspect ratio
                aspect_ratio = height / width
                if aspect_ratio > 2.4: # Its Human
                    obj_name_original = "1"
                
                obj_meta.rect_params.has_bg_color = 1
                bg_color = pyds.NvOSD_ColorParams()
                


                bg_color.red = 0.0
                bg_color.green = 0.0
                bg_color.blue = 1.0
                bg_color.alpha = 0.2
                obj_meta.rect_params.bg_color = bg_color

       

                # # Map the top_value to the font size range
                # variable_font_size = 5 + (5 - 25) * (top_value / 1080)

                 
                obj_meta.text_params.font_params.font_size = 10
                obj_meta.text_params.set_bg_clr = 1



            else:

           

                #Extract data from sgie
                l = obj_meta.classifier_meta_list
                # print("Classifier meta list = ", l)
                while l is not None:
                    try:
                        classifier_meta = pyds.NvDsClassifierMeta.cast(l.data)
                        # print("Classifier meta = ", classifier_meta)
                        n = classifier_meta.label_info_list
                        while n is not None:
                            try:
                                labelInfo = pyds.NvDsLabelInfo.cast(n.data)
                               
                                # Check the conditions individually and store the results in variables
                                # confidence_condition = obj_meta.confidence > 0.1
                                # result_prob_condition = labelInfo.result_prob > 0.6

                                # # Set save_image to True if both conditions are met, otherwise set it to False
                                # save_image = confidence_condition and result_prob_condition
                                # print(f"Label Info = {labelInfo.result_prob}")
                                if(labelInfo.result_prob >= 0.51):
                                    obj_name_original = labelInfo.result_label
                                    obj_class_id = labelInfo.result_class_id

                                    obj_meta.rect_params.has_bg_color = 1
                                    bg_color = pyds.NvOSD_ColorParams()
                                    

                                    obj_bg_color =  class_colors[obj_class_id]

                                   
                                    # Set bg_color components
                                    bg_color.red = obj_bg_color.red
                                    bg_color.green = obj_bg_color.green
                                    bg_color.blue = obj_bg_color.blue
                                    bg_color.alpha = obj_bg_color.alpha

                                    obj_meta.rect_params.bg_color = bg_color
                                    obj_meta.rect_params.border_color.alpha = 0.4
                                    top_value = obj_meta.rect_params.top  # Assuming it ranges from 0 to 1080

                                    # Map the top_value to the font size range
                                    variable_font_size = round(8 + (21 - 8) * (top_value / 1080))

                                    
                                    obj_meta.text_params.font_params.font_size = int(variable_font_size)

                                    if int(obj_meta.rect_params.top) < 100:
                                        label_pos = 100
                                        obj_meta.text_params.y_offset = int(obj_meta.rect_params.top)
                                    else:
                                        obj_meta.text_params.y_offset = int(obj_meta.rect_params.top - 50)

                                    


                                    obj_meta.text_params.font_params.font_color.set(0.0, 0.0, 0.0, 1) 
                                    
                                    obj_meta.text_params.text_bg_clr.set(bg_color.red, bg_color.green, bg_color.blue, 0.4)


                                    
                           
                                    if obj_meta.class_id == 0:
                                        obj_meta.text_params.display_text = f"ID:{obj_meta.object_id} Person"

                                    else:
                                        obj_meta.text_params.display_text = f"ID:{obj_meta.object_id} {convertURSTP_to_bdvehiclenet(obj_name_original)}"
                                      
                                        



                               

                            except StopIteration:
                                break
                            try:
                                n = n.next
                            except StopIteration:
                                break

                    except StopIteration:
                        break

                    try:
                        l = l.next
                    except StopIteration:
                        break
                   
            if obj_name_original is not None:
                img_width = frame_meta.source_frame_width
                img_height = frame_meta.source_frame_height

                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                width = obj_meta.rect_params.width
                height = obj_meta.rect_params.height
                                


          


            if obj_name_original is None:
                obj_meta.rect_params.has_bg_color = 1
                bg_color = pyds.NvOSD_ColorParams()
                bg_color.red = 1.0
                bg_color.green = 0.0
                bg_color.blue = 0.0
                bg_color.alpha = 0.2
                obj_meta.rect_params.bg_color = bg_color
                obj_meta.text_params.font_params.font_size = 10
                obj_meta.text_params.set_bg_clr = 1
                obj_name_original = "26"
            if save_image:        
                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                right = left + obj_meta.rect_params.width
                bottom = top + obj_meta.rect_params.height
                print("Resize to ratio")
                left, top = left * width_ratio, top * height_ratio
                right, bottom = right * width_ratio, bottom * height_ratio
               

              
                

                if obj_name_original:
                    if bdvehiclenet_classid is not None:
                        annotation_data.append("{} 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(bdvehiclenet_classid, left, top, right, bottom))
                       
############
    ############

            l_user_meta = obj_meta.obj_user_meta_list
            # Extract object level meta data from NvDsAnalyticsObjInfo
        
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(
                            user_meta.user_meta_data)
                        if user_meta_data.lcStatus:
                           timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                           source_id = frame_meta.source_id
                           crossing_type = user_meta_data.lcStatus[0]
                           if obj_name_original is not None:
                               obj_name = obj_name_original
                               print(f"object_name from urstp:{obj_name}")
                               video_timestamp_str = convert_gst_timestamp_to_string(
                                   timestampp, get_time_interval(qTime[source_id]))
                               write_kitti_output(n_frame_gpu, obj_meta, timestamp, src_names, source_id,
                                                  crossing_type, cap_time, dateS, qTime, obj_name, video_timestamp_str)

                except StopIteration:
                    break

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break


            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        if save_image and annotation_data:
            
            random_number = format(random.randint(0, 9999999999), '010')
            src_id = frame_meta.source_id
            if not os.path.exists(frames_folder):
                os.makedirs(frames_folder)
                print(f"frames_folder '{frames_folder}' created.")
            annotations_folder = "/media/sigmind/URSTP_HDD1414/aatrainingdata/detector/labels"
            if not os.path.exists(annotations_folder):
                os.makedirs(annotations_folder)
                print(f"annotations_folder '{annotations_folder}' created.")

            img_path = "{}/frame_{}_{}_{}_{}_{}_{}_{}.jpg".format(frames_folder,src_names[src_id],cap_time[src_id],qTime[src_id],obj_meta.object_id, obj_name_original, bdvehiclenet_classid,random_number)
            txt_path = "{}/frame_{}_{}_{}_{}_{}_{}_{}.txt".format(annotations_folder,src_names[src_id],cap_time[src_id],qTime[src_id],obj_meta.object_id, obj_name_original, bdvehiclenet_classid,random_number)
           

        saved_count["stream_{}".format(frame_meta.pad_index)] += 1


        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def write_kitti_output(image, obj_meta, timestamp,src_names,src_id,crossing_type,cap_time, dateS, qTime,obj_name_RHD,video_timestamp):

    #cap_time Location number, qTime video number#
    ##############################################
    obj_name = obj_name_original
    image_name = "{}_{}_{}_{}_{}_{}_{}.jpg".format(video_timestamp,src_names[src_id],cap_time[src_id],qTime[src_id],obj_meta.object_id, obj_name_RHD, crossing_type)

    crop_image_file_path = "./kitti_data"+"/classification/images/"  + src_names[src_id] + "/" + "Cam" + str(src_id) + "/" + crossing_type  + "/" + qTime[src_id] + "/" + obj_name_RHD + "/" + image_name
    if not os.path.exists(os.path.dirname(crop_image_file_path)):
        os.makedirs(os.path.dirname(crop_image_file_path))


    cropped_object_image = crop_bounding_boxes_cupy(image, obj_meta, obj_meta.confidence)
    cropped_object_image_RGB = cv2.cvtColor(
        cropped_object_image.get(), cv2.COLOR_BGR2RGB)
    cv2.imwrite(crop_image_file_path, cropped_object_image_RGB)
    print(f"captime,src_names,src_id{cap_time,src_names,src_id}")
    insert_data(video_timestamp,qTime,crossing_type,obj_meta.object_id, obj_meta.confidence, obj_name, crop_image_file_path)

def crop_bounding_boxes_cupy(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    cropped_image = image[top:top+height, left:left+width, :]
    return cropped_image


def crop_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    cropped_image = image[top:top+height, left:left+width]
    return cropped_image


def draw_bounding_boxes(image, source_id, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_class = pgie_classes_str[obj_meta.class_id]
    cropped_image = image[top:top+height, left:left+width]
    image = cv2.rectangle(image, (left, top), (left + width,
                          top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name + ',C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255, 0), 2)
    return image


def nightvision_gamma(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Here, the batch has only one source
    l_frame = batch_meta.frame_meta_list
    if not l_frame:
        return

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        source_id = frame_meta.source_id

        if qTime[source_id] in night_time:
            #print("Night Correction Enabled")
            gamma_correction = False
            
        else:
            gamma_correction = False

        if gamma_correction:
        
            # Create dummy owner object to keep memory for the image array alive
            owner = None
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
            data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
                hash(gst_buffer), frame_meta.batch_id)
            # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                ctypes.py_object, ctypes.c_char_p]
            # Get pointer to buffer and create UnownedMemory object from the gpu buffer
            c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
            unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
            # Create MemoryPointer object from unownedmem, at index 0
            memptr = cp.cuda.MemoryPointer(unownedmem, 0)
            # Create cupy array to access the image data. This array is in GPU buffer
            n_frame_gpu = cp.ndarray(
                shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
            # Initialize cuda.stream object for stream synchronization
            # Use null stream to prevent other cuda applications from making illegal memory access of buffer
            stream = cp.cuda.stream.Stream(null=True)

            # Apply gamma correction to the n_frame_gpu
            # gamma = 10  # You can adjust this value to control the level of correction
            # Modify the red channel to add blue tint to image
            with stream:
                n_frame_gpu_scaled = n_frame_gpu / 255.0

                # Apply individual adjustments if enabled
                if enable_exposure:
                    n_frame_gpu_scaled = n_frame_gpu_scaled ** exposure

                if enable_brightness:
                    n_frame_gpu_scaled = (n_frame_gpu_scaled * contrast) + brightness / 255.0

                if enable_contrast:
                    n_frame_gpu_scaled = n_frame_gpu_scaled * contrast

                # Apply gamma correction if enabled
                if enable_gamma:
                    corrected_n_frame_gpu = cp.power(n_frame_gpu_scaled, 1 / gamma)
                else:
                    corrected_n_frame_gpu = n_frame_gpu_scaled

                # Clip and scale the corrected image
                corrected_n_frame_gpu = cp.clip(corrected_n_frame_gpu, 0, 1)
                corrected_n_frame_gpu = (corrected_n_frame_gpu * 255.0).astype(cp.uint8)

                cp.copyto(n_frame_gpu, corrected_n_frame_gpu)

            stream.synchronize()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK



def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

#    if "source" in name:
#        Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def normalize_bbox(left, top, width, height, image_width, image_height):
    x_center = (left + width / 2) / image_width
    y_center = (top + height / 2) / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height
    return x_center, y_center, width_normalized, height_normalized


def main(args):
    # Check input arguments
    if len(args) < 5:
        sys.stderr.write(
            "usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    for i in range(0, len(args) - 4):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    number_sources = len(args) - 4

    global cap_time, qTime, src_names,dateS, folder_name
 
    GObject.threads_init()
    Gst.init(None)

    uploaded_analytics_path= args[-2]
    uploaded_detector_path =args[-1]
    print(f"anal: {uploaded_analytics_path}")
    print(f"dect: {uploaded_detector_path}")
    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i+1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")

        # Create nvvidconv3 \n
        nvvidconv3 = Gst.ElementFactory.make(
            "nvvideoconvert", "convertor3_" + str(i))
        if not nvvidconv3:
            sys.stderr.write(" Unable to create nvvidconv3 \n")

        # Create filter3 \n
        caps3 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter3 = Gst.ElementFactory.make("capsfilter", "filter3_" + str(i))
        if not filter3:
            sys.stderr.write(" Unable to get the caps filter3 \n")
        filter3.set_property("caps", caps3)
        pipeline.add(source_bin)
        # Add nvvidconv3 and filter3 to the pipeline
        pipeline.add(nvvidconv3)
        pipeline.add(filter3)
        # Link source_bin -> nvvidconv3 -> filter3
        source_bin.link(nvvidconv3)
        nvvidconv3.link(filter3)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")

        # Get the srcpad of filter3 instead of source_bin
        srcpad = filter3.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvtracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    sgie = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to make sgie \n")

    print("Creating nvdsanalytics \n ")
    analytics_cfg_path= uploaded_analytics_path
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write(" Unable to create nvanalytics \n")
    nvanalytics.set_property("config-file", analytics_cfg_path)

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    
    nvof = Gst.ElementFactory.make("nvof", "nvopticalflow")
    if not nvof:
        sys.stderr.write("Unable to create optical flow \n")
    print("Creating nv optical flow visualisation element \n")
    nvofvisual = Gst.ElementFactory.make("nvofvisual", "nvopticalflowvisual")
    if not nvofvisual:
        sys.stderr.write("Unable to create flow visualisation element")
    print("Creating queue \n ")
    of_queue = Gst.ElementFactory.make("queue", "q_after_of")
    if not of_queue:
        sys.stderr.write("Unable to create queue \n")
    print("Creating queue \n")
    ofvisual_queue = Gst.ElementFactory.make("queue", "q_after_ofvisual")
    if not ofvisual_queue:
        sys.stderr.write("Unable to create queue \n")



    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if (is_aarch64()):
        print("Creating transform \n ")
        transform = Gst.ElementFactory.make(
            "nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 0)

    pgie_cfg_path = uploaded_detector_path
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 40000)
    pgie.set_property('config-file-path', pgie_cfg_path)
    pgie.set_property('interval', 0)

    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    pgie.set_property("interval", 3)
    if sgie_enable:
        sgie.set_property('config-file-path',
                          "./FAN_URSTP/config_infer_secondary_vehicletypenet.txt")
        sgie.set_property("batch-size", number_sources+1)

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    # tiler.set_property("rows", tiler_rows)
    # tiler.set_property("columns", tiler_columns)
    tiler.set_property("rows", 2)
    tiler.set_property("columns", 2)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    # mem_type = int(pyds.NVBUF_MEM_CUDA_DEVICE)
    streammux.set_property("nvbuf-memory-type",
                           int(pyds.NVBUF_MEM_CUDA_DEVICE))
    nvvidconv.set_property("nvbuf-memory-type", mem_type)
    nvvidconv1.set_property("nvbuf-memory-type", mem_type)
    tiler.set_property("nvbuf-memory-type", mem_type)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dsnvanalytics_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process',
                                 tracker_enable_batch_process)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame',
                                 tracker_enable_past_frame)

    # Prev pipeline
    # print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    if sgie_enable:
        pipeline.add(sgie)
    pipeline.add(nvanalytics)
    pipeline.add(nvof)
    pipeline.add(of_queue)
    pipeline.add(tiler)
    # pipeline.add(nvvidconv)
    # pipeline.add(filter1)
    # pipeline.add(nvvidconv1)
    if osd_enable:
        pipeline.add(nvosd)

    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    # pgie.link(sgie)
    pgie.link(tracker)
    if sgie_enable:
        tracker.link(sgie)
        sgie.link(nvanalytics)
    else:
        tracker.link(nvanalytics)
    nvanalytics.link(tiler)
    # nvvidconv1.link(filter1)
    # filter1.link(tiler)
    # tiler.link(nvvidconv)
    if osd_enable:
        tiler.link(nvosd)
        nvosd.link(sink)
        if is_aarch64():
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)

    else:
        tiler.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_srcpad = pgie.get_static_pad("src")
    if not pgie_srcpad:
        sys.stderr.write("Unable to get pgie src pad \n")
    # else:
    #     pgie_srcpad.add_probe(Gst.PadProbeType.BUFFER, nightvision_gamma, None)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    nvanalytics_src_pad = nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        nvanalytics_src_pad.add_probe(
            Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)
            cap_time.append(source.split("/")[-3])
            dateS.append(source.split("/")[-2])
            qTime.append(source.split("/")[-1][:-4])
            src_names.append(source.split("/")[-4])
            print("src name = {}, cap_time = {}, qTime = {}, dateS = {}".format(
                src_names, cap_time, qTime, dateS))


    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    #obj_name = "mawa"
    sys.exit(main(sys.argv))

