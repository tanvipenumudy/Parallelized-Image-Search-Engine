#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import multiprocessing
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pickle
import time

app = Flask(__name__)

fe = FeatureExtractor()
features = pickle.load(open('features.pkl','rb'))
print("features.pkl loaded")
img_paths = pickle.load(open('img_paths.pkl','rb'))
print("img_paths.pkl loaded")
features = np.array(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        t = time.time()
        file = request.files['query_img']

        img = Image.open(file.stream)  
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:50]  
        scores = [(dists[id], img_paths[id]) for id in ids]
        print("Time Taken (Server 1):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=5000)
