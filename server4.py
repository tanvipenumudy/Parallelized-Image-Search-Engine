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
        
        def worker1(features):
            return np.linalg.norm(features-query, axis=1)
        def worker2(features):
            return np.linalg.norm(features-query, axis=1)
        def worker3(features):
            return np.linalg.norm(features-query, axis=1)
        def worker4(features):
            return np.linalg.norm(features-query, axis=1)

        if __name__ == "__main__": 
            p1 = multiprocessing.Process(target=worker1)
            p2 = multiprocessing.Process(target=worker2)
            p3 = multiprocessing.Process(target=worker3)
            p4 = multiprocessing.Process(target=worker4)

            p1.start()
            p2.start()
            p3.start()
            p4.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()

            p1.join() 
            p2.join() 
            p3.join() 
            p4.join() 

            n = int(50000/4)
            d1 = worker1(features[:n])
            d2 = worker2(features[n:2*n])
            d3 = worker3(features[2*n:3*n])
            d4 = worker4(features[3*n:])

        dists = np.concatenate((d1,d2,d3,d4))
        ids = np.argsort(dists)[:50]  
        scores = [(dists[id], img_paths[id]) for id in ids]
        print("Time Taken (Server 4):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=8000)
