
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
import shutil

# set up the command line arguments
parser = argparse.ArgumentParser(description='Make predictions on a set of images using the Label Studio API')
parser.add_argument('--api_url', type=str, default='http://localhost:8080/api', help='URL of the Label Studio API')
parser.add_argument('--input_dir', type=str, default='./../Data/images', help='directory containing the images to be predicted')
parser.add_argument('--url_dir', type=str, default='./../Data/url', help='URL of the input images')
parser.add_argument('--url_image_dir', type=str, default='./../Data/images', help='directory to save the predictions')
parser.add_argument('--output_dir', type=str, default='./..', help='directory to save the predictions')
parser.add_argument('--database_predicted_dir', type=str, default='./../Data/Database_Predicted', help='directory to save the predictions')
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
parser.add_argument('--project_name', type=str, default='Demo', help='name of the project in Label Studio')
parser.add_argument('--label_config', type=str, default='./label_config.xml', help='path to the label config file')
parser.add_argument('--label_config_url', type=str, default=None, help='URL of the label config file')
parser.add_argument('--label_config_json', type=str, default=None, help='path to the label config JSON file')

args = parser.parse_args()

# provide API authentication credentials
TOKEN = 'dd3f146381240448ef590c41563d274931ab6c84'

# authenticate with the API
response = requests.get(args.api_url + '/projects', headers={'Authorization': 'Token ' + TOKEN})
# check that the authentication was successful
if response.status_code != 200:
    print('ERROR: authentication with API failed')
    sys.exit()
else: print('authentication check was successful')

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

# get the list of images to be predicted
image_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
print('found', len(image_files), 'images to be predicted in file:', args.input_dir)

# convert an image url to a saved image in image_path
def url_to_image(url):
    # download the image, 
    response = requests.get(url)
    # if response is 200 (OK), then convert the image to a numpy array
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    # otherwise, return not possible and print url
    else:
        print('not possible to download image from url:', url)
        return None

# open args.url_dir and go inside every text file that does not start with 'Processed' and then read the urls into a list
url_files = [f for f in os.listdir(args.url_dir) if os.path.isfile(os.path.join(args.url_dir, f)) and f.lower().endswith(('.txt')) and not f.startswith('Processed')]
url_files.sort()
print('found', len(url_files), 'text files to be predicted in file:', args.url_dir)

time_string = time.strftime("%Y%m%d-%H%M%S")
# if there are text file in the url directory, then download the images from the urls
if len(url_files) > 0:

    # create a list of urls
    url_list = []
    for url_file in url_files:
        with open(os.path.join(args.url_dir, url_file), 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                url_list.append(line)
    # remove '\n' from the end of each url
    url_list = [url.rstrip('\n') for url in url_list]

    # create a list of images
    image_list = []
    for url in url_list:
        if url_to_image(url) is not None:
            image_list.append(url_to_image(url))
            print('downloaded and converted image', url)

    # save the list of images inside url_image_dir
    for i in range(len(image_list)):
        # check the format of the image
        if image_list[i].format == 'JPEG':
        # save as url_[Data:Time]_imageX.jpg
            image_list[i].save(os.path.join(args.url_image_dir, 'url_' + time_string + '_image' + str(i) + '.jpg'))
        elif image_list[i].format == 'PNG':
        # save as url_[Data:Time]_imageX.png
            image_list[i].save(os.path.join(args.url_image_dir, 'url_' + time_string + '_image' + str(i) + '.png'))
        # properly save WEBP format 
        elif image_list[i].format == 'WEBP':
            image_list[i].save(os.path.join(args.url_image_dir, 'url_' + time_string + '_image' + str(i) + '.png'), format='PNG')
        else:
            print('image format not supported:', image_list[i].format, ' FOR IMAGE: ', image_list[i])



# get the project id from the project name
response = requests.get(args.api_url + '/projects', headers={'Authorization': 'Token ' + TOKEN})
if response.status_code != 200:
    print('ERROR: could not get the list of projects')
    sys.exit()
else:
    projects = response.json()
    project_id = None
    for project in projects['results']:
        if project['title'] == args.project_name:
            project_id = str(project['id'])
            break
    if project_id is None:
        print('ERROR: could not find project', args.project_name)
        sys.exit()


# get the list of tasks
response = requests.get(args.api_url + '/projects/' + project_id + '/tasks/', headers={'Authorization': 'Token ' + TOKEN})
# print the number of tasks returned by the API
if response.status_code != 200:
    tasks = response.json()
    if tasks['detail'] == 'Not found.':
        print('No pending tasks were found in the project')
    else:
        print('ERROR: could not get the list of tasks')
        sys.exit()
else: # code 200
    tasks = response.json()
    print('found', len(tasks), 'tasks in project: ', args.project_name)
    # list the ids in the project
    task_ids = []
    for task in tasks:
        task_ids.append(task['id'])
    print('task ids:', task_ids)


# get the total annotations of the project : project["total_annotations_number"] by using GET /api/projects/{project_id}
project = requests.get(args.api_url + '/projects/' + project_id, headers={'Authorization': 'Token ' + TOKEN}).json()
# print the number of annotations in the project
print('found', project['total_annotations_number'], 'annotations and', project['total_predictions_number'], 'predictions in project:',  args.project_name)

###############################################################################################################################################3
# The images are locally processed and predicted then imported to the project

# predict the image using the YOLO detect.py script
# python detect.py --weights args.weights --img 640 --conf 0.25 --source args.input_dir --save-txt --project args.output_dir
# call the detect.py script with its flags
old_labels_dir = args.output_dir + '/Data/labels'
old_images_dir = args.input_dir

# create a new folder in the database with the time string
Data_folder = args.database_predicted_dir + '/' + time_string
os.mkdir(Data_folder)

# if the labels directory is empty, try the os command
if not os.listdir(old_labels_dir):
    try:
        os.system('python detect.py --weights ' + args.weights + ' --img 640 --conf 0.25 --source ' 
        + args.input_dir + ' --save-txt --project ' + args.output_dir + ' --name Data' + ' --exist-ok')
        print('Predictions saved in', args.output_dir)
        # create a predictions directory
        predictions_dir = args.output_dir + '/Data/predictions'
        os.mkdir(predictions_dir)
        # put all the images.jpg deom /Data to /Data/predictions
        for image in os.listdir(args.output_dir + '/Data'):
            if image.endswith('.jpg') or image.endswith('.png'):
                shutil.move(args.output_dir + '/Data/' + image, predictions_dir)
        new_predictions_dir = Data_folder + '/predictions'
        print('moving', len(os.listdir(predictions_dir)), 'predictions to the database')
        shutil.move(predictions_dir, new_predictions_dir)
        time.sleep(0.2)

    except Exception as e:
        print('ERROR: could not predict the images')
        print(e)
        sys.exit()
else: print('As the label directory is not empty, the images were not predicted')

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
print('-----------------------------------------------------------------------------------------------------------------')
print('Will now move DATA to Database and free folders')

# cut url folder content and paste in Database_Predicted/url
# move the images and labels to the database


# create images and labels inside
images_dir = Data_folder + '/images'
labels_dir = Data_folder + '/labels'
os.mkdir(images_dir)
os.mkdir(labels_dir)

print('moving', len(os.listdir(old_labels_dir)), 'labels to the database')
for file in os.listdir(old_labels_dir):
    shutil.move(old_labels_dir + '/' + file, labels_dir + '/' + file)

print('moving', len(os.listdir(old_images_dir)), 'images to the database')
for file in os.listdir(old_images_dir):
    shutil.move( old_images_dir + '/' + file, images_dir + '/' + file)

time.sleep(0.2)


print('making a record of processed url files')
url_files = [f for f in os.listdir(args.url_dir) if os.path.isfile(os.path.join(args.url_dir, f)) and f.lower().endswith(('.txt')) and not f.startswith('Processed')]
for file in url_files:
    # save it with time stamp: file_[time stamp].txt
    shutil.copy(args.url_dir + '/' + file, args.url_dir + '/' + 'Processed.' + file.split('.')[0] + '_' + time_string + '.txt')
print('-----------------------------------------------------------------------------------------------------------------')

# The labels and images are imported to the project
# get the list of labels to be imported
print('labels dir is: ', labels_dir)
label_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f)) and f.lower().endswith(('.txt'))]
label_files.sort()

print('images dir is: ', images_dir)
# get the list of images to be imported
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

print('found', len(label_files), 'labels and', len(image_files), 'images to be imported')

## wait until enter is pressed to confitm the upload to the server
print('-----------------------------------------------------------------------------------------------------------------')
print('type c to confirm the upload to the server')
print('-----------------------------------------------------------------------------------------------------------------')
print('type t to terminate the script')
print('-----------------------------------------------------------------------------------------------------------------')
while True:
    # ask the user to type c to continue
    i = input()
    if i == 'c':
        print('Starting the upload to the server')
        break
    elif i == 't':
        print('Terminating the script')
        sys.exit()
    else:
        print('waiting for operator to type c')
    

# create the import storage
local_dir = os.path.dirname(os.path.realpath(__file__))

Classes = {'1': 'Special Forces', '2': 'Terrorist', '0': 'Civilian', '3': 'Weapon'} 
for image_file in image_files:
    # get the image id from the image file name
    image_id = image_file.split('.')[0]
    # get the label file name from the image file name
    label_file = image_id + '.txt'
    # get the image url
    #image_url = 'file://' + local_dir + '/' + images_dir + '/' + image_file
    image_url = 'http://0.0.0.0:1212/' + time_string + '/images/' + image_file
    # get the label file path
    label_file_path = labels_dir + '/' + label_file
    # get the pixel width
    image_width = Image.open(images_dir + '/' + image_file).size[0]
    # get the pixel height
    image_height = Image.open(images_dir + '/' + image_file).size[1]
    # get the labels
    labels = []
    num = 0
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
                "id": image_id + '_' + str(num),
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
            num = num+1

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
        #print(response.status_code)
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
        print('imported', image_id, 'to the project')
        # print(response)
    except:
        print('ERROR: could not import image', image_id, 'to project', args.project_name)
        print(response.json())
        sys.exit()
