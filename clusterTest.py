from pathlib import Path
from PIL import Image
import pickle
import numpy as np
#from feature_extractor import FeatureExtractor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from keras.preprocessing.image import load_img 
import pickle

#data = {}
#c=0
#for i in sorted(Path("./static/sample").glob("*.png")):
#    fe = FeatureExtractor()
#    data[i] = fe.extract(img=Image.open(i))
#    c+=1
#    print(c)

#pickle.dump(data,open('sample.pkl','wb'))

#filenames = np.array(list(data.keys()))
#feat = np.array(list(data.values()))

features = pickle.load(open('features.pkl','rb'))
print("features.pkl loaded")

"""img_paths = pickle.load(open('img_paths.pkl','rb'))
print("img_paths.pkl loaded")"""

"""print(len(features), len(img_paths))"""

#features = features[:100]
#img_paths = img_paths[:100]

"""pca = PCA(n_components=100, random_state=123)
pca.fit(features)
x = pca.transform(features)

pickle.dump(x,open('PCA.pkl','wb'))
print("PCA Done!")"""

"""x = pickle.load(open('PCA.pkl','rb'))
print("PCA.pkl loaded!")

kmeans = KMeans(n_clusters=1000, random_state=123).fit(x)
print("KMeans Done!")

groups = {}
for file, cluster in zip(img_paths, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

pickle.dump(groups,open('Groups.pkl','wb'))
print("Grouping Done!")

print(len(groups))"""

groups = pickle.load(open('Groups.pkl','rb'))
print("Groups.pkl Loaded!")

#print(groups[0])

"""def view_cluster(cluster):
    plt.figure(figsize=(25,25))
    files = groups[cluster]
    if(len(files)>30):
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

view_cluster(0)"""

group_feat = {}
for i in range(1000):
    group_feat[i] = []
    for j in groups[i]:
        group_feat[i].append(features[int((str(j.stem)[-5:]).lstrip('0'))-1])

pickle.dump(group_feat,open('group_feat.pkl','wb'))
print("Features Appended!")









