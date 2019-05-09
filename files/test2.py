import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
import cv2
import pandas as pd
from pandas import Series, DataFrame
import match
cap = cv2.VideoCapture(0)
 
sys.path.append("..")
 
from utils import label_map_util
 
from utils import visualization_utils as vis_util
 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
 
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
i=0
over_write=0
class_data1=Series([])
class_data5=Series([])
class_data4=Series([])
class_data3=Series([])
class_data2=Series([])
class_data6=Series([])
cod_data1=Series([])
cod_data5=Series([])
cod_data4=Series([])
cod_data3=Series([])
cod_data2=Series([])
cod_data6=Series([])
arr=Series([])
tcod_data=Series([])
tcl_data=Series([])
def crop(image_path, coords, saved_location):
  """
  @param image_path: The path to the image to edit
  @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
  @param saved_location: Path to save the cropped image
  """
  image_obj = Image.open(image_path)
  cropped_image = image_obj.crop(coords)
  cropped_image.save(saved_location)
  cropped_image.show()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
        ret, image_np = cap.read()
        im = Image.fromarray(image_np)
        im.save("tmp/{}org.jpg".format(over_write))
        img = cv2.imread("tmp/"+str(over_write)+"org.jpg")
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        codarr,classarr,image_np=vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        if(i<60):
          if(i==0):
            cod_data1=codarr
            class_data1=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          if(i==10):
            cod_data2=codarr
            class_data2=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          if(i==20):
            cod_data3=codarr
            class_data3=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          if(i==30):
            cod_data4=codarr
            class_data4=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          if(i==40):
            cod_data5=codarr
            class_data5=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          if(i==50):
            cod_data6=codarr
            class_data6=classarr
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
        elif i%10 == 0:
          if over_write==0 :
            tcod_data=cod_data1
            tcl_data=class_data1
            cod_data1=codarr
            class_data1=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          elif over_write==1 :
            tcod_data=cod_data2
            tcl_data=class_data2
            cod_data2=codarr
            class_data2=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          elif over_write==2 :
            tcod_data=cod_data3
            tcl_data=class_data3
            cod_data3=codarr
            class_data3=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          elif over_write==3 :
            tcod_data=cod_data4
            tcl_data=class_data4
            cod_data4=codarr
            class_data4=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          elif over_write==4 :
            tcod_data=cod_data5
            tcl_data=class_data5
            cod_data5=codarr
            class_data5=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
          elif over_write==5 :
            tcod_data=cod_data6
            tcl_data=class_data6
            cod_data6=codarr
            class_data6=classarr
            for itr, val in enumerate(tcl_data):
              exists=os.path.isfile("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              if(exists):
                os.remove("tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")
              os.rename("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(itr)+"new.jpg")             
            for itr, val in enumerate(classarr):
              crop_img = img[int(codarr.iloc[itr]['ymin']*480):int(codarr.iloc[itr]['ymax']*480), int(codarr.iloc[itr]['xmin']*640):int(codarr.iloc[itr]['xmax']*640)]
              cv2.imwrite("tmp/"+str(over_write)+"_"+str(itr)+"new.jpg", crop_img)
        
        print("Prev")
        if(i%10==0):
          print(over_write)
        if(i%10==0 and i>=60):
          print(over_write)
          for ii,j in enumerate(tcl_data):
            for k,l in enumerate(classarr):
              if(l==j):
                if(float(codarr.iloc[k]['ymax'])/float(tcod_data.iloc[ii]['ymax'])>=0.9 and float(codarr.iloc[k]['ymax'])/float(tcod_data.iloc[ii]['ymax'])<=1.1 and 
                float(codarr.iloc[k]['xmax'])/float(tcod_data.iloc[ii]['xmax'])>=0.9 and float(codarr.iloc[k]['xmax'])/float(tcod_data.iloc[ii]['xmax'])<=1.1):
                  tresult=0
                  if(float(codarr.iloc[k]['ymin'])==0 and float(tcod_data.iloc[ii]['ymin'])==0):
                    tresult=1
                  else:
                    tvar=abs(float(codarr.iloc[k]['ymin'])-float(tcod_data.iloc[ii]['ymin']))/max(float(codarr.iloc[k]['ymin']),float(tcod_data.iloc[ii]['ymin']))
                    if(tvar<=.2):
                      tresult=1
                  if(tresult==1):
                    print(j)
                    print("tmp/"+str(over_write)+"_"+str(k)+"new.jpg")
                    error=match.mean_square_error("tmp/"+str(over_write)+"_"+str(k)+"new.jpg","tmp/"+str(over_write+6)+"_"+str(ii)+"new.jpg")
                    if(error<250):
                      img = cv2.imread("tmp/"+str(over_write+6)+"_"+str(ii)+"new.jpg",0)
                      if(j=="person"):
                        cv2.imwrite("tmp1/persons/"+str(i)+"lb_"+str(itr)+"new.jpg", img)
                      else:
                        cv2.imwrite("tmp1/objects/"+str(i)+"lb_"+str(itr)+"new.jpg", img)

        print("end")
        i=i+1
        if(i%10==0):
          over_write=(over_write+1)%6
        cv2.imshow('Object Detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break