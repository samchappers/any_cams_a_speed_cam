# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
from turtle import distance
import cv2
import os
import sys
from config.config import Configuration as config
import pickle
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='classifier score threshold')
    parser.add_argument("-c", "--calibration_file_path", required=False, default="",
                    help="Path to the calibration pkl file path.")
    args = parser.parse_args()
    config.load_config("./config.yml")

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
 #   pkl_file_path = args["calibration_file_path"]
    pkl_file_path = args.calibration_file_path
    if pkl_file_path == "":
        pkl_file_path = config.cfg["calibration"]["pkl_file_path"]
    with open(pkl_file_path, 'rb') as f:
        transformation_matrix, scale_factor = pickle.load(f)

    cap = cv2.VideoCapture(args.camera_idx)


    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        cv2_im, bird_view_points, camera_view_points = append_objs_find_points(cv2_im, inference_size, objs, labels, transformation_matrix)
        cv2_im = distance_calculation_label(cv2_im, bird_view_points, camera_view_points, scale_factor)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def distance_calculation_label(cv2_im, bird_view_points, camera_view_points, scale_factor):
    distances = np.array([[0 for i in range(len(bird_view_points))] for j in range(len(bird_view_points))])
    for i, _ in enumerate(bird_view_points):
        p1 = bird_view_points[i]
        for j, _ in enumerate(bird_view_points):
            p2 = bird_view_points[j]
            if i==j:
                distances[i][j] = 0
            elif all(bird_view_points[i]) == 0 or all(bird_view_points[j]) == 0: # if either point is an origin point set dist to zero
                distances[i][j] = 0
            else:
                dist = np.linalg.norm(p1 - p2)
                distances[i][j] = dist * scale_factor

    violation_distance_threshold = config.cfg["social_distancing"]["distance_threshold_ft"]
    violations = []
    if not len(distances)== 0:
        rows, columns = distances.shape
        for i in range(rows):
            for j in range(columns):
                if not distances[i][j] == 0 and distances[i][j] < violation_distance_threshold:
                    violations.append([i, j])
        if len(violations) > 0:
            cv2_im = plot_violations(cv2_im, camera_view_points, violations)

    return cv2_im


def plot_violations(cv2_im, camera_view_points, violations):
    for violation in violations:
        b1, b2 = violation
        p1, p2 = camera_view_points[b1], camera_view_points[b2]
        p1 = [int(x) for x in p1]; p2 = [int(x) for x in p2]
        line = np.array([p1,p2])
        cv2_im2 = cv2.line(cv2_im, (p1[0],p1[1]), (p2[0],p2[1]), (0, 0, 255), 10)
#        cv2_im2 = cv2.polylines(cv2_im, line, 1, (0, 255, 255), 6)
#        cv2_im2 = cv2.drawContours(cv2_im, line, -1 , (0,255,0),4)
#        cv2_im = cv2.line(cv2_im, p1, p2, (0, 255, 255), 2)
        print(violation)
        print(p1)
        print(p2)
    return cv2_im2

def append_objs_find_points(cv2_im, inference_size, objs, labels, transformation_matrix):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    bird_view_points = np.zeros([len(objs),2]) ; camera_view_points = np.zeros([len(objs),2])
    for i, obj in enumerate(objs):
        if labels.get(obj.id, obj.id) == 'person' or labels.get(obj.id, obj.id) == 'car':
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
            
            contactpoint = np.array([int(x0 + ((x1-x0)/2)), int(y0 + ((y1-y0)*4/5))]) #this gives the center point or point of contact with the floor

            points = np.append(contactpoint, 1)
            points = np.matmul(transformation_matrix, points) #transforms the contact point to birds eye view
            bird_view_points[i] = points[:2]
            camera_view_points[i] = contactpoint

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im, bird_view_points, camera_view_points

if __name__ == '__main__':
    main()
