# Management of DATASET, pre-labelling and exporting to Label Studio:

**Author: Elio Assaf**




# Installation on New Linux or Windows Device 

In order to install, you need to have a virtual environment with yolov7 running first.
+ clone the repo of yolov7 classification, then install its dependencies. This is explained as well in the mask-pose project, where you also have to first install the wheels for torch and torchvision, then clone yolo and install the commented requirements.txt.
```cmd
python -m venv venv

# enter the venv then:

pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

# comment out torch and torchvision in requirements.txt

pip3 install -r requirements.txt

```

> Now create new folders next to yolov7 for the prediction data, and weights, then create some sub-folders inside the data folder.
```cmd
#step out of from yolov7 repo:
cd ..
mkdir DATA
mkdir weights
cd DATA
mkdir url
mkdir Database_Predicted
```

> Next, Paste `Predictions.py` in the yolov7 directory, and `README-ELIO.md` as well outside of the yolov7 directory.

---

---
# Instructions for Exporting Pre-Labelled Predicted Data to Label Studio using the API.
**Author: Elio Assaf**

---

## Label Studio Setup:

+ Start Label Studio in a new terminal connected to the python virtual environment by executing `label-studio start`

+ Register to label studio by going to http://localhost:8080/
+ Create a new project with `Create`.
    + In `Project Name`, name the project `Predictions`, and go to 
    + In `Labelling Setup` choose `Object Detection with Bounding Boxes`. 
    Next you will be asked for the template for labelling. You can press ` Code` and paste the code below to have 4 classes.
    ```json
    <View>
    <Image name="image" value="$image"/>
    <RectangleLabels name="label" toName="image">
        
        
    <Label value="Special Forces" background="#68df75"/><Label value="Terrorist" background="#D4380D"/><Label value="Civilian" background="#759bf5"/><Label value="Weapon" background="#c15add"/></RectangleLabels>
    </View>
    ```
---
## Virtual Environment Setup
First, run the following commands:
```cmd
# Go to DATA/Database_predicted
cd /DATA/Database_predicted/

# Start a localhost server on port 1212 with the following command:
python3 -m http.server 1212
```
> ### How to store Images to be predicted?
> + The new images to be labelled are now accepted in the form of ***url*** links written in a text file each in a new line.
> + The user can have as many text files containing the urls, and they need to be stored in ***DATA/url/*** . 
### Prediction Steps of Predictions.py:
+ The code first downloads all url links into images stored in `DATA/images`. This location is temporary. 
+ Next, The yolov7 model is called with the weights stored in `weights/best.pt`. When a better model is trained, these weights should be also updated. The prediction is then made and stored in `DATA/labels/`.
+ The user is then asked if they want to upload the predictions to Label Studio. 
    + If the user choses to do so, the labels and images folders will be uploaded in the wright format and stored as a BACKUP in `DATA/Database_predicted/` with the appropriate time stamp. The text file containing the url will also be renaimed to `Processed.name?_time_stamp.txt` in order not to be processed again in the next run.
    + If the user opts to stop the process, the process is re-initialized and nothing is stored or exported.

## Run the Prediction:
> Running the predictions can only be done after everything has been setup.

> As a check list, make sure of the following:
+ Label studio is started in a new Terminal
+ A project inside Label studio has been created
+ A second local host server is listening on port 1212 and initialised from the right directory.
+ The images have been stored in url format inside a .txt file and in the right directory.

> If all steps are checked, you can continue by running the code

### To run the prediction, execute the following commands:

```
cd yolov7
python Predictions.py --api-token XXXXXXXXXXXXXX
```
### Important Flags:

` --api-url ` # The localhost port specification for Label Studio. Default is `http://localhost:8080/api`

` --api-token ` # The api token. Can be found in the settings of Label Studio.

` --project-name` # The name of the project in the Label Studio interface. Default is `Predictions`. This project should exist in the Label Studio Interface along with the Labels of the classes.
