import pickle
import numpy as np

groups = pickle.load(open('Groups.pkl','rb'))
group_feat = pickle.load(open('group_feat.pkl','rb'))
names = pickle.load(open('names.pkl','rb'))
comb = pickle.load(open('comb.pkl','rb'))
arr0 = comb[0]
arr = comb[1]
res = []

print("Done Initialization!")

"""
a = group_feat[arr0[0]][names[arr0[0]].index(133)]
b = group_feat[arr[0]][names[arr[0]].index(792)]

a = np.array(a)
b = np.array(b)

print(np.linalg.norm(a-b))

"""
c = 0
for k in range(len(arr)):
	dist = np.array([np.linalg.norm(np.array(group_feat[arr0[k]][names[arr0[k]].index(j)])-np.array(group_feat[arr[k]][names[arr[k]].index(i)])) for j in names[arr0[k]] for i in names[arr[k]]])
	source = np.array([(i,j) for j in names[arr0[k]] for i in names[arr[k]]])
	ids = np.argsort(dist)[:5]
	res.append(source[ids[0]].tolist())
	res.append(source[ids[1]].tolist())
	res.append(source[ids[2]].tolist())
	res.append(source[ids[3]].tolist())
	res.append(source[ids[4]].tolist())
	c+=1
	print(c)

pickle.dump(res,open('res.pkl','wb'))
print("Result Appended!")
print(len(res))