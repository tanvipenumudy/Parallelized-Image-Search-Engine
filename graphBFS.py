import networkx as nx
import pickle
import queue

graph = pickle.load(open('graph50knx.pkl','rb'))
print("graph50knx.pkl loaded")
nodes = pickle.load(open('nodes.pkl','rb'))
print("nodes.pkl loaded")

def traversal(source):
    N = len(nodes)
    visited = [False]*N
    distance = [None]*N

    que = queue.Queue()
    que.put(source)
    distance[source] = 0
    visited[source] = True

    while(not que.empty()):
        temp = que.get()
        data = graph.neighbors(temp)
        for i in data:
            if(visited[i] == False):
                que.put(i)
                visited[i] = True
                distance[i] = distance[temp]+1
                
    return [i for i, x in enumerate(distance) if x==1], [i for i, x in enumerate(distance) if x==2]


