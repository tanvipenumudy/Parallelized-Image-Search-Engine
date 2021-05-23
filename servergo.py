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
group_feat3 = pickle.load(open('group_feat3.pkl','rb'))
print("group_feat3.pkl loaded")
graphBFS_feat = pickle.load(open('graphBFS_feat.pkl','rb'))
print("graphBFS_feat.pkl loaded")
graphBFS_paths = pickle.load(open('graphBFS_paths.pkl','rb'))
print("graphBFS_paths.pkl loaded")

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
            group_feat3[i] = np.array(group_feat3[i])
            graphBFS_feat[i] = np.array(graphBFS_feat[i])
            arr.append(np.mean(np.linalg.norm(group_feat3[i]-query, axis=1)))
        ids = np.argsort(np.array(arr))[:1]  

        def worker1(features):
            return np.linalg.norm(features-query, axis=1)
        def worker2(features):
            return np.linalg.norm(features-query, axis=1)

        if __name__ == "__main__": 
            p1 = multiprocessing.Process(target=worker1)
            p2 = multiprocessing.Process(target=worker2)

            p1.start()
            p2.start()

            p1.join()
            p2.join()

            p1.join() 
            p2.join()

            ll = len(graphBFS_feat[ids[0]])//2
            d1 = worker1(graphBFS_feat[ids[0]][:ll])
            d2 = worker2(graphBFS_feat[ids[0]][ll:])

        dists = np.concatenate((d1,d2))
        #dists = np.array(list(set(dists.tolist())))
        img_paths = graphBFS_paths[ids[0]]
        #img_paths = list(set(img_paths))
        if(dists.shape[0]>50):
            ids = np.argsort(dists)[:50]  
        else:
            ids = np.argsort(dists)
        scores = [(dists[id], img_paths[id]) for id in ids]
        print("Time Taken (Server G2):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=8080)
