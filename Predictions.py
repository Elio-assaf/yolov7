
## use the Label Studio API to make predictions on a set of images
## process the predictions using detect.py and send them to the API

import os
import sys
import json
import requests
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from torch import det
from detect import detect
import time
import subprocess
#from label_studio_converter import LabelStudioConverter
#from label_studio_converter import LabelStudioConverterError

# set up the command line arguments
parser = argparse.ArgumentParser(description='Make predictions on a set of images using the Label Studio API')
parser.add_argument('--api_url', type=str, default='http://localhost:8081/api', help='URL of the Label Studio API')
parser.add_argument('--input_dir', type=str, default='./../Data/images', help='directory containing the images to be predicted')
parser.add_argument('--output_dir', type=str, default='./..', help='directory to save the predictions')
parser.add_argument('--weights', type=str, default='./../weights/best.pt', help='path to the model weights')
parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--augment', action='store_true', help='augment images for increased accuracy')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save_crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--project_name', type=str, default='Predictions', help='name of the project in Label Studio')
parser.add_argument('--label_config', type=str, default='./label_config.xml', help='path to the label config file')
parser.add_argument('--label_config_url', type=str, default=None, help='URL of the label config file')
parser.add_argument('--label_config_json', type=str, default=None, help='path to the label config JSON file')
parser.add_argument('--label_config_json_url', type=str, default=None, help='URL of the label config JSON file')
parser.add_argument('--label_config_json_string', type=str, default=None, help='label config JSON string')
parser.add_argument('--label_config_json_string_url', type=str, default=None, help='URL of the label config JSON string')

args = parser.parse_args()

# check that the input directory exists
if not os.path.isdir(args.input_dir):
    print('ERROR: input directory does not exist')
    sys.exit()

# check that the output directory exists
if not os.path.isdir(args.output_dir):
    print('ERROR: output directory does not exist')
    sys.exit()

# check that the model exists
if not os.path.isfile(args.weights):
    print('ERROR: weights file does not exist')
    sys.exit()

# check that the label config file exists
if not os.path.isfile(args.label_config):
    print('ERROR: label config file does not exist')
    sys.exit()

# provide authentication credentials
TOKEN = 'dd3f146381240448ef590c41563d274931ab6c84'
# authenticate with the API
response = requests.get(args.api_url + '/projects', headers={'Authorization': 'Token ' + TOKEN})
# check that the authentication was successful
if response.status_code != 200:
    print('ERROR: authentication failed')
    sys.exit()
else:
    print('authentication successful')

# get the list of images to be predicted
image_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
print('found', len(image_files), 'images to be predicted')

# get the label config
label_config = None
if args.label_config_json_string is not None:
    label_config = args.label_config_json_string
elif args.label_config_json_string_url is not None:
    label_config = requests.get(args.label_config_json_string_url).text
elif args.label_config_json is not None:
    with open(args.label_config_json) as f:
        label_config = json.load(f)
elif args.label_config_json_url is not None:
    label_config = requests.get(args.label_config_json_url).json()
elif args.label_config_url is not None:
    label_config = requests.get(args.label_config_url).text
else:
    with open(args.label_config) as f:
        label_config = f.read()

# # create the Label Studio converter
# try:
#     converter = LabelStudioConverter(label_config)
# except LabelStudioConverterError as e:
#     print('ERROR: could not create the Label Studio converter')
#     print(e)
#     sys.exit()

# project name is Predictions by default
project_name = args.project_name
# get the project id from the project name
response = requests.get(args.api_url + '/projects', headers={'Authorization': 'Token ' + TOKEN})
if response.status_code != 200:
    print('ERROR: could not get the list of projects')
    sys.exit()
else:
    projects = response.json()
    project_id = None
    for project in projects['results']:
        if project['title'] == project_name:
            project_id = str(project['id'])
            break
    if project_id is None:
        print('ERROR: could not find project', project_name)
        sys.exit()


# get the list of tasks
response = requests.get(args.api_url + '/projects/' + project_id + '/tasks/', headers={'Authorization': 'Token ' + TOKEN})
# print the number of tasks returned by the API
if response.status_code != 200:
    tasks = response.json()
    if tasks['detail'] == 'Not found.':
        print('No tasks were found')
    else:
        print('ERROR: could not get the list of tasks')
        sys.exit()
else: # code 200
    tasks = response.json()
    print('found', len(tasks), 'tasks in project: ', project_name)
    # list the ids in the project
    task_ids = []
    for task in tasks:
        task_ids.append(task['id'])
    print('task ids:', task_ids)


# get the total annotations of the project : project["total_annotations_number"] by using GET /api/projects/{project_id}
project = requests.get(args.api_url + '/projects/' + project_id, headers={'Authorization': 'Token ' + TOKEN}).json()
# print the number of annotations in the project
print('found', project['total_annotations_number'], 'annotations in project: ', project_name)
time.sleep(1)
# "total_predictions_number"
print('found', project['total_predictions_number'], 'predictions in project: ', project_name)

time.sleep(1)

###############################################################################################################################################3
# The images are locally processed and predicted then imported to the project

# predict the image using the YOLO detect.py script
# python detect.py --weights args.weights --img 640 --conf 0.25 --source args.input_dir --save-txt --project args.output_dir
# call the detect.py script with its flags
labels_dir = args.output_dir + '/Data/labels'
images_dir = args.input_dir

# if the labels directory is empty, try the os command
if not os.listdir(labels_dir):
    try:
        os.system('python detect.py --weights ' + args.weights + ' --img 640 --conf 0.25 --source ' 
        + args.input_dir + ' --save-txt --project ' + args.output_dir + ' --name Data' + ' --exist-ok')
    except Exception as e:
        print('ERROR: could not predict the images')
        print(e)
        sys.exit()

# The labels and images are imported to the project


# get the list of labels to be imported
label_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f)) and f.lower().endswith(('.txt'))]
label_files.sort()
print('found', len(label_files), 'labels to be imported')

# get the list of images to be imported
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
print('found', len(image_files), 'images to be imported')


# import the labeled images to the project: POST /api/projects/{project_id}/import
# Payload:
# example_prediction_task.json.
# [{
#   "data": {
#     "image": "/static/samples/sample.jpg" 
#   },

#   "predictions": [{
#     "model_version": "one",
#     "score": 0.5,
#     "result": [
#       {
#         "id": "result1",
#         "type": "rectanglelabels",        
#         "from_name": "label", "to_name": "image",
#         "original_width": 600, "original_height": 403,
#         "image_rotation": 0,
#         "value": {
#           "rotation": 0,          
#           "x": 4.98, "y": 12.82,
#           "width": 32.52, "height": 44.91,
#           "rectanglelabels": ["Airplane"]
#         }
#       },
#       {
#         "id": "result2",
#         "type": "rectanglelabels",        
#         "from_name": "label", "to_name": "image",
#         "original_width": 600, "original_height": 403,
#         "image_rotation": 0,
#         "value": {
#           "rotation": 0,          
#           "x": 75.47, "y": 82.33,
#           "width": 5.74, "height": 7.40,
#           "rectanglelabels": ["Car"]
#         }
#       },
#       {
#         "id": "result3",
#         "type": "choices",
#         "from_name": "choice", "to_name": "image",
#         "value": {
#           "choices": ["Airbus"]
#       }
#     }]
#   }]
# }]

# create import storage for the images
# POST /api/storages/localfiles
# Payload:
# {
#   "path": "string",
#   "regex_filter": "string",
#   "use_blob_urls": true,
#   "title": "string",
#   "description": "string",
#   "last_sync": "2019-08-24T14:15:22Z",
#   "last_sync_count": 0,
#   "project": 0
# }

# create the import storage
local_dir = os.path.dirname(os.path.realpath(__file__))
# print('local_dir', local_dir)
# response = requests.post(args.api_url + '/storages/localfiles', 
# headers={'Authorization': 'Token ' + TOKEN},
# json={
#     "path": local_dir + '/' + args.input_dir,
#     "regex_filter": None,
#     "use_blob_urls": False,
#     "title": "YOLOv7 predictions",
#     "description": "YOLOv7 predictions",
#     "last_sync": None,
#     "last_sync_count": None,
#     "project": project_id
# })
# print(response.status_code)
# print(response.json())

# sys.exit()
#--------------------------------------------------------------------------------------------------------------
'''
Note, but didn^t work
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/elio


WHAT DID WORK IS:

Go to images directory and start a simple http server in a new terminal
python3 -m http.server 1212

Just make sure it is still active when you export the images
'''

Classes = {'1': 'Special Forces', '2': 'Terrorist', '0': 'Civilian', '3': 'Weapon'} 
for image_file in image_files:
    # get the image id from the image file name
    image_id = image_file.split('.')[0]
    # get the label file name from the image file name
    label_file = image_id + '.txt'
    # get the image url
    #image_url = 'file://' + local_dir + '/' + images_dir + '/' + image_file
    #image_url = 'file:///home/elio/PycharmProjects/YOLO_Predictions_Label_Studio/images/image1.jpg'
    image_url = 'http://0.0.0.0:1212/' + image_file
    #image_url = 'https://www.planet-wissen.de/natur/voegel/pinguine/intropinguinfederkleidgjpg100~_v-gseagaleriexl.jpg'
    # get the label file path
    label_file_path = labels_dir + '/' + label_file

    # get the pixel width
    image_width = Image.open(images_dir + '/' + image_file).size[0]
    image_width = 1600
    print(image_width)
    # get the pixel height
    image_height = Image.open(images_dir + '/' + image_file).size[1]
    image_height = 900
    print(image_height)

    # get the labels
    labels = []
    with open(label_file_path) as f:
        for line in f:
            # get the label
            label = line.split()
            # get the label name
            label_name = label[0]
            # get the label coordinates
            x = float(label[1]) * 100
            y = float(label[2]) * 100
            width = float(label[3]) * 100
            height = float(label[4]) * 100
            rotation = 0.0
            # convert the label to the label studio format
            x = x - width / 2
            y = y - height / 2

            # create the label
            label =  {
                "id": "result" + str(len(labels) + 1),
                "from_name": "label", 
                "to_name": "image",
                "type": "rectanglelabels",
                # "original_width": image_width,
                # "original_height": image_height,
                # "image_rotation": 0,
                "value": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rotation": rotation,
                            "rectanglelabels": [
                                                Classes[label_name]
                                                ]
                        }
                }
            
            
            # add the label to the list of labels
            labels.append(label)

    # create the payload
    payload = {
        "data": {
            "image": image_url
        },
        "predictions": [
            {   
                "model_version": "one", ## TRAINED MODEL Reference ID
                "score": 0.5,
                "result": labels
                                
            }
        ]
    }


    try:
        response = requests.post(args.api_url + '/projects/' + project_id + '/import', headers={'Authorization': 'Token ' + TOKEN} , json=payload)
        print(response.status_code)
        #print(response.text)
        # check if the response status_code is 201 for success
        if response.status_code == 201:
            pass
        else:
            print('ERROR: could not import image: ', image_file)
            print('details:' ,response['detail'])
            print(response['validation_errors']['non_field_errors'])
            if not response['exc_info'] == None:
                print('exc_info : \n',response['exc_info'])
            sys.exit()
        response = response.json()
        print('imported image', image_id, 'to project', project_name)
        # print(response)
    except:
        print('ERROR: could not import image', image_id, 'to project', project_name)
        print(response.json())
        sys.exit()
    # wait for 1 second


sys.exit()




for image_file in image_files:
    # get the image id from the image file name
    image_id = image_file.split('.')[0]
    # get the label file name from the image file name
    label_file = image_id + '.txt'
    # get the image url
    image_url = images_dir + '/' + image_file
    # get the label file path
    label_file_path = labels_dir + '/' + label_file
    # get the labels
    labels = []
    with open(label_file_path) as f:
        for line in f:
            # get the label
            label = line.split()
            # get the label name
            label_name = label[0]
            # get the label coordinates
            label_x = float(label[1])
            label_y = float(label[2])
            label_width = float(label[3])
            label_height = float(label[4])
            # convert the coordinates to the Label Studio format
            x = label_x - label_width / 2
            y = label_y - label_height / 2
            width = label_width
            height = label_height
            rotation = 0.0
            # create the label
            label = {
                "from_name": "box",
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [
                        {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rotation": rotation
                        }
                    ]
                }
            }
            # add the label to the list of labels
            labels.append(label)

    # create the payload
    payload = {
        "files": [
            {
                "image": image_url,
                "predictions": [
                    {
                        "result": labels
                    }
                ]
            }
        ]
    }
    # import the image to the project
    try:
        response = requests.post(args.api_url + '/projects/' + project_id + '/import', headers={'Authorization': 'Token ' + TOKEN}, json=payload)
        response = response.json()
        # check if the response status_code is 201 for success
        if response['status_code'] == 201:
            pass
        else:
            print('ERROR: could not import image: ', image_file)
            print('details:' ,response['detail'])
            print(response['validation_errors']['non_field_errors'])
            if not response['exc_info'] == None:
                print('exc_info : \n',response['exc_info'])
            sys.exit()

        print('imported image', image_id, 'to project', project_name)
        # print(response)
    except:
        print('ERROR: could not import image', image_id, 'to project', project_name)
        print(response.json())
        sys.exit()
    # wait for 1 second
    time.sleep(1)


sys.exit()


























# loop over the images to be predicted
for image_file in image_files:

    # get the image path
    image_path = os.path.join(args.input_dir, image_file)
    print('predicting: ', image_path)

    # get the image
    image = Image.open(image_path)

    # predict the image using the YOLO detect.py script
    # python detect.py --weights opt.weights --img 640 --conf 0.25 --source opt.input_dir --save-txt --project opt.output_dir

    print('predicting image', image_path)

    
    # get the image name
    image_name = os.path.splitext(image_file)[0]

    # get the image width and height
    width, height = image.size

    # create the Label Studio task
    task = {
        'data': {
            'image': image_path
        },
        'completions': [
            {
                'result': [
                    {
                        'from_name': 'img',
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': []
                        }
                    }
                ],
                'ground_truth': False,
                'skipped': False,
                'was_cancelled': False,
                'created_at': '',
                'id': 0,
                'lead_time': 0,
                'updated_at': ''
            }
        ],
        'predictions': [
            {
                'result': [
                    {
                        'from_name': 'img',
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': []
                        }
                    }
                ],
                'ground_truth': False,
                'skipped': False,
                'was_cancelled': False,
                'created_at': '',
                'id': 0,
                'lead_time': 0,
                'updated_at': ''
            }
        ],
        'ground_truth': False,
        'id': 0,
        'predictions': [],
        'project': args.project_name,
        'created_at': '',
        'updated_at': ''

    }



    # loop over the results
    for result in results:

        # get the bounding box
        x1, y1, x2, y2 = result[0]

        # get the label
        label = result[1]

        # get the confidence
        confidence = result[2]

        # convert the bounding box to Label Studio format
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        # add the bounding box to the Label Studio task
        task['completions'][0]['result'][0]['value']['rectanglelabels'].append({
            'height': h,
            'width': w,
            'x': x,
            'y': y,
            'rectanglelabels': [
                label
            ]
        })

        # add the bounding box to the Label Studio task
        task['predictions'][0]['result'][0]['value']['rectanglelabels'].append({
            'height': h,
            'width': w,
            'x': x,
            'y': y,
            'rectanglelabels': [
                label
            ]
        })


    # create the Label Studio task
    print('creating Label Studio task for image', image_path)
    task = requests.post(args.api_url + '/tasks/', json=task, headers={'Authorization': 'Token ' + TOKEN}).json()

    # get the task ID
    task_id = task['id']

    # get the task URL
    task_url = args.api_url + '/tasks/' + str(task_id) + '/'
    print('task URL:', task_url)

    # get the task
    task = requests.get(task_url).json()

    # get the task completions
    completions = task['completions']

    # get the task predictions
    predictions = task['predictions']

    # get the task result
    result = completions[0]['result'][0]['value']['rectanglelabels']

    # get the task result
    prediction = predictions[0]['result'][0]['value']['rectanglelabels']

    
