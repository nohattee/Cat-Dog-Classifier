from flask import Flask, jsonify, make_response, send_from_directory, request
import os
from os.path import exists, join
import imageio
import gc
from model import ImageClassification

import torch
import torch.nn.functional as F
from torchvision import transforms, models

from constants import CONSTANTS

app = Flask(__name__, static_folder='build')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4919, 0.4615, 0.4179], 
                             std=[0.2557, 0.2470, 0.2503])
    ])
    image = preprocess(image)
    return image

def predict_(namemodel, img):
    model_dict = {}
    model = torch.load("server/checkpoints/" + namemodel + ".pth")
    model.eval()
    img = imageio.imread(img)
    img = process_image(img).unsqueeze(0)
    result = model(img)
    return F.softmax(result, dim=1)

@app.route('/predict/<modelname>', methods=["POST"])
def predict(modelname):
    img = request.files['img']
    result = predict_(modelname, img)
    gc.collect()
    return jsonify(result.detach().numpy().tolist())

# Catching all routes
# This route is used to serve all the routes in the frontend application after deployment.
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    file_to_serve = path if path and exists(join(app.static_folder, path)) else 'index.html'
    return send_from_directory(app.static_folder, file_to_serve)

# Error Handler
@app.errorhandler(404)
def page_not_found(error):
    json_response = jsonify({'error': 'Page not found'})
    return make_response(json_response, CONSTANTS['HTTP_STATUS']['404_NOT_FOUND'])

if __name__ == '__main__':
    app.run(debug=True, port=CONSTANTS['PORT'])