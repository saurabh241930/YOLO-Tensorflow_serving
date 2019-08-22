# YOLO-Tensorflow_serving_Flask
In this project ,I created end-to-en object detection pipeline which takes image or video as a input and returns class names &amp; object bounding box location and deployed using TF-serving & Flask
<img src="https://i.imgur.com/NmLCJnH.png" border=0>

## Process / Steps


I converted weights file of keras darkflow tiny_yolo model into frozen pb format for tf serving and hosted in tf-serving docker and created external flask api backend which is using tf-serving REST api ,this backend takes base64 format of image and outputs a JSON output.
(I’ve already converted weights & cfg file into using flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb )

## Setup : 

First host a  frozen model inside tf-serving docker

**`docker run -p 8501:8501 -v /home/sp/Documents/submission/built_graph/:/models/darkflow -e MODEL_NAME=darkflow -t tensorflow/serving`**

Install all required libraries

**`pip3 install -r requirements.txt`**

Then start Flask server

**`S:~/Documents/submission$ python3 app.py`**

Start sending the image as a client in base 64 format

**`S:~/Documents/submission$ python base64_request.py -i uploaded.jpg`**

## Screenshot

Tensorflow Serving Docker
<img src="https://i.imgur.com/knGYEXA.png" border=0>

Starting flask server

<img src="https://i.imgur.com/oli4ac3.png" border=0>
<img src="https://i.imgur.com/yf8B3b1.png" border=0>

Frontend Client 

<img src="https://i.imgur.com/Bi5BVT1.png" border=0>






