import pickle
features = pickle.load(open('features.pkl','rb'))
group_feat = pickle.load(open('group_feat.pkl','rb'))
print("features.pkl loaded")
img_paths = pickle.load(open('img_paths.pkl','rb'))
first = pickle.load(open('first.pkl','rb'))
print("img_paths.pkl loaded")

graphBFS_feat = {}
graphBFS_paths = {}

count = 0

for i in range(1000):
	graphBFS_feat[i] = []
	graphBFS_paths[i] = []
	for j in first[i]:
		graphBFS_feat[i].append(features[j-1])
		graphBFS_paths[i].append(img_paths[j-1])
	count+=1
	print(count)

pickle.dump(graphBFS_feat,open('graphBFS_feat.pkl','wb'))
print("graphBFS_feat done!")
pickle.dump(graphBFS_paths,open('graphBFS_paths.pkl','wb'))
print("graphBFS_paths done!")

for i in range(1000):
	group_feat[i] = group_feat[i][:3]

pickle.dump(group_feat,open('group_feat3.pkl','wb'))
