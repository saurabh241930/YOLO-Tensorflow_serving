import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify,render_template
from keras.preprocessing import image
import numpy as np
from darkflow.net.build import TFNet
from flask_cors import CORS

app = Flask(__name__)

options = {"model": 'cfg/tiny-yolo.cfg', "threshold": 0.3}
tfnet = TFNet(options)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/')
def render_static():
    return render_template('index.html')





@app.route('/yolo/predict/', methods=['POST'])
def yolo_detection():



    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
                                            target_size=(416, 416))) / 255.

    img = img.astype('float16')

    payload = {
    "signature_name" : "predict",
    "instances": [{'input': img.tolist()}]
    }


    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/darkflow:predict', json=payload)

    json_response = json.loads(str(r.text))

    net_out = np.squeeze(np.array(json_response['predictions'], dtype='float32'))
    boxes = tfnet.framework.findboxes(net_out)
    h, w, _ = img.shape
    threshold = tfnet.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = tfnet.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })

        for prediction in boxesInfo:
            print(prediction)

    return jsonify(str(boxesInfo))      



if __name__ == "__main__":
    app.run(host='0.0.0.0')
