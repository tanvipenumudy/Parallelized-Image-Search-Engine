# ~6 hrs+ (Serial)
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.JPEG")):
        print(img_path)  
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy") 
        np.save(feature_path, feature)
