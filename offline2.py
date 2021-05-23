# ~5:45 hrs+ (Parallel - 2 Cores)
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import multiprocessing
import numpy as np

fe = FeatureExtractor()
img_paths = pickle.load(open('img_paths.pkl','rb'))
print("img_paths.pkl loaded")

def worker1(img_paths):
    for img_path in img_paths:
        print(img_path)  
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy") 
        np.save(feature_path, feature)
def worker2(img_paths):
    for img_path in img_paths:
        print(img_path)  
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy") 
        np.save(feature_path, feature)

if __name__ == "__main__": 
    p1 = multiprocessing.Process(target=worker1)
    p2 = multiprocessing.Process(target=worker2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    p1.join() 
    p2.join() 

    n = int(50000/2)
    worker1(img_paths[:n])
    worker2(img_paths[n:2*n])