#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import multiprocessing
from PIL import Image
from feature_extractor import FeatureExtractor
from graphBFS import traversal
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pickle
import time

app = Flask(__name__)

fe = FeatureExtractor()
firsts = pickle.load(open('firsts.pkl','rb'))
print("firsts.pkl loaded")
group_feat3 = pickle.load(open('group_feat3.pkl','rb'))
print("group_feat3.pkl loaded")
features = pickle.load(open('features.pkl','rb'))
print("features.pkl loaded")
img_paths = pickle.load(open('img_paths.pkl','rb'))
print("img_paths.pkl loaded")

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
            arr.append(np.mean(np.linalg.norm(group_feat3[i]-query, axis=1)))

        idx = np.argsort(np.array(arr))[:1]  
        source = firsts[idx[0]]
        dist1, dist2 = traversal(source)

        def worker1(dist):
            graphBFS_feat = []
            graphBFS_paths = []
            for i in dist:
                graphBFS_feat.append(features[i-1])
                graphBFS_paths.append(img_paths[i-1])
            return np.linalg.norm(np.array(graphBFS_feat)-query, axis=1), graphBFS_paths
        def worker2(dist):
            graphBFS_feat = []
            graphBFS_paths = []
            for i in dist:
                graphBFS_feat.append(features[i-1])
                graphBFS_paths.append(img_paths[i-1])
            return np.linalg.norm(np.array(graphBFS_feat)-query, axis=1), graphBFS_paths

        if __name__ == "__main__": 
            p1 = multiprocessing.Process(target=worker1)
            p2 = multiprocessing.Process(target=worker2)

            p1.start()
            p2.start()

            p1.join()
            p2.join()

            p1.join() 
            p2.join()

            d1, g1 = worker1(dist1)
            d2, g2 = worker2(dist2)

        dists = np.concatenate((d1,d2))
        #dists = np.array(list(set(dists.tolist())))
        paths = g1+g2
        #paths = list(set(paths))
        if(dists.shape[0]>50):
            ids = np.argsort(dists)[:50]  
        else:
            ids = np.argsort(dists)
        scores = [(dists[id], paths[id]) for id in ids]
        print("Time Taken (Server G2):",time.time()-t)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(port=8080)
