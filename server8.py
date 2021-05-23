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
        def worker5(features):
            return np.linalg.norm(features-query, axis=1)
        def worker6(features):
            return np.linalg.norm(features-query, axis=1)
        def worker7(features):
            return np.linalg.norm(features-query, axis=1)
        def worker8(features):
            return np.linalg.norm(features-query, axis=1)

        if __name__ == "__main__": 
            p1 = multiprocessing.Process(target=worker1)
            p2 = multiprocessing.Process(target=worker2)
            p3 = multiprocessing.Process(target=worker3)
            p4 = multiprocessing.Process(target=worker4)
            p5 = multiprocessing.Process(target=worker5)
            p6 = multiprocessing.Process(target=worker6)
            p7 = multiprocessing.Process(target=worker7)
            p8 = multiprocessing.Process(target=worker8)

            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            p8.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()
            p7.join()
            p8.join()

            p1.join() 
            p2.join() 
            p3.join() 
            p4.join() 
            p5.join() 
            p6.join() 
            p7.join() 
            p8.join() 

            n = int(50000/8)
            d1 = worker1(features[:n])
            d2 = worker2(features[n:2*n])
            d3 = worker3(features[2*n:3*n])
            d4 = worker4(features[3*n:4*n])
            d5 = worker5(features[4*n:5*n])
            d6 = worker6(features[5*n:6*n])
            d7 = worker7(features[6*n:7*n])
            d8 = worker8(features[7*n:])

        dists = np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8))
        ids = np.argsort(dists)[:50] 
        scores = [(dists[id], img_paths[id]) for id in ids]
        print("Time Taken (Server 8):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=8080)
