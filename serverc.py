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
group_feat = pickle.load(open('group_feat.pkl','rb'))
print("group_feat.pkl loaded")
groups = pickle.load(open('Groups.pkl','rb'))
print("Groups.pkl loaded")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        t = time.time()
        file = request.files['query_img']
        
        img = Image.open(file.stream)  
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        query = fe.extract(img)
        arr = []
        for i in range(1000):
            group_feat[i] = np.array(group_feat[i])
            arr.append(np.mean(np.linalg.norm(group_feat[i][:3]-query, axis=1)))
        ids = np.argsort(np.array(arr))[:4]  

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

            d1 = worker1(group_feat[ids[0]])
            d2 = worker2(group_feat[ids[1]])
            d3 = worker3(group_feat[ids[2]])
            d4 = worker4(group_feat[ids[3]])

        dists = np.concatenate((d1,d2,d3,d4))
        img_paths = groups[ids[0]]+groups[ids[1]]+groups[ids[2]]+groups[ids[3]]
        if(dists.shape[0]>50):
            ids = np.argsort(dists)[:50]  
        else:
            ids = np.argsort(dists)
        scores = [(dists[id], img_paths[id]) for id in ids]
        print("Time Taken (Server C4):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=5050)
