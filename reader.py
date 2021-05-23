# ~1:15 hrs (Serial)
from pathlib import Path
import pickle
import numpy as np

features = []
img_paths = []
c=0
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".JPEG"))
    c+=1
    print(c)

pickle.dump(features,open('features.pkl','wb'))
pickle.dump(img_paths,open('img_paths.pkl','wb'))

