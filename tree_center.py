
import numpy as np
from scipy.spatial.distance import cosine

import json

cosine_count = 0

count = 0
f = open("all_questions.jsonl.old",mode="r",encoding="utf-8")
lines = f.readlines()
all_questions = []
for line in lines:
    all_questions.append(line.rstrip())
    # if count > 1000:
    #     break
    # count += 1

count = 0
f = open("all_embeddings.jsonl.old",mode="r",encoding="utf-8")
lines = f.readlines()
all_embeddings = []
for line in lines:
    all_embeddings.append(json.loads(line.rstrip()))
    # if count > 1000:
    #     break
    # count += 1

question2embedding = {}
for i,q in enumerate(all_questions):
    question2embedding[q] = all_embeddings[i]


f = open("bert_code/run/datasets/quora/test_all.tsv",mode="r",encoding="utf-8")

all_question_with_gold_label = set()
true_pair = {}
true_pair2 = {}
count = 0
for line in f:
    label,q1,q2,_ = line.rstrip().split("\t")
    if label=="1":
        true_pair[q1.rstrip()]= q2.rstrip()
        # true_pair[q2.rstrip()] = q1.rstrip()
        true_pair2[q2.rstrip()]= q1.rstrip()
        # true_pair2[q1.rstrip()] = q2.rstrip()
        all_question_with_gold_label.add(q1.rstrip())
        all_question_with_gold_label.add(q2.rstrip())
    count+=1


def findTopNindex(array,N):
    # return np.argsort(array)[::-1][:N] # 从大到小
    return np.argsort(array)[:N] # 从小到大

def cos_distance(vector1, vector2):
    # if vector1.any()==0:
    #     vector1 += 0.00001
    global cosine_count
    cosine_count+=1
    return cosine(vector1,vector2)

    # return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class Node():
    def __init__(self,vector,level=-1,doc_id=-1):
        self.vector = vector
        self.level = level
        self.child_list = []
        self.doc_id = doc_id

    def append_child(self, node):
        self.child_list.append(node)

    def insert_child(self, node, index):
        self.child_list.insert(index,node)

    def get_child_list(self):
        return self.child_list


class Tree():
    def __init__(self, root_node):
        self.root_node = root_node

    def breadth_travel(self):
        if self.root_node == None:
            return
        queue = [self.root_node]
        last_level = -1
        while len(queue)>0:
            cur_point = queue.pop(0)

            if last_level!=cur_point.level:
                print()
            last_level = cur_point.level

            print(len(cur_point.get_child_list()), end=' ')

            for child in cur_point.get_child_list():
              if len(child.get_child_list())>0:
                queue.append(child)

    def beam_search(self, query_vector, top_k=2):
        child_list = self.root_node.get_child_list()
        score_list = []
        for child in child_list:
            score_list.append(cos_distance(child.vector, query_vector))
        indexes = findTopNindex(score_list, top_k)

        topk_child = np.array(child_list)[indexes]

        topk_child_merge = []
        topk_score_merge = []
        for child in topk_child:
            child_list = child.get_child_list()
            score_list = []
            for child in child_list:
                score_list.append(cos_distance(child.vector, query_vector))
            indexes = findTopNindex(score_list, top_k)
            topk_child = np.array(child_list)[indexes]
            topk_score = np.array(score_list)[indexes]
            topk_child_merge.extend(topk_child)
            topk_score_merge.extend(topk_score)

        indexes = findTopNindex(topk_score_merge, top_k)
        topk_child = np.array(topk_child_merge)[indexes]
        topk_child_merge = []
        topk_score_merge = []
        for child in topk_child:
            child_list = child.get_child_list()
            score_list = []
            for child in child_list:
                score_list.append(cos_distance(child.vector, query_vector))
            indexes = findTopNindex(score_list, top_k)
            topk_child = np.array(child_list)[indexes]
            topk_score = np.array(score_list)[indexes]
            topk_child_merge.extend(topk_child)
            topk_score_merge.extend(topk_score)

        indexes = findTopNindex(topk_score_merge, top_k)
        topk_child = np.array(topk_child_merge)[indexes]
        topk_child_merge = []
        topk_score_merge = []
        for child in topk_child:
            child_list = child.get_child_list()
            score_list = []
            for child in child_list:
                score_list.append(cos_distance(child.vector, query_vector))
            indexes = findTopNindex(score_list, top_k)
            topk_child = np.array(child_list)[indexes]
            topk_score = np.array(score_list)[indexes]
            topk_child_merge.extend(topk_child)
            topk_score_merge.extend(topk_score)

        indexes = findTopNindex(topk_score_merge, top_k)
        topk_child = np.array(topk_child_merge)[indexes]

        return [child.doc_id for child in topk_child]


    def search(self, query_vector):
        top_k = 1
        child_list = self.root_node.get_child_list()
        score_list = []
        for child in child_list:
            score_list.append(cos_distance(child.vector,query_vector))
        indexes = findTopNindex(score_list,top_k)

        topk_child = np.array(child_list)[indexes]
        score_list = []
        child_list = topk_child[0].get_child_list()
        for child in child_list:
            score_list.append(cos_distance(child.vector, query_vector))
        indexes = findTopNindex(score_list, top_k)

        topk_child = np.array(child_list)[indexes]
        score_list = []
        child_list = topk_child[0].get_child_list()
        for child in child_list:
            score_list.append(cos_distance(child.vector, query_vector))
        indexes = findTopNindex(score_list, top_k)

        topk_child = np.array(child_list)[indexes]
        score_list = []
        child_list = topk_child[0].get_child_list()
        for child in child_list:
            score_list.append(cos_distance(child.vector, query_vector))
        indexes = findTopNindex(score_list, top_k)

        return [child.doc_id for child in np.array(child_list)[indexes]]

import re
def build_tokenizer():
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    return lambda doc: token_pattern.findall(doc)


# all_sentence_vector = []
# print(build_tokenizer()("The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."))
# for i in range(1000):
#     sentence_vector = np.random.random([300])
#     all_sentence_vector.append(sentence_vector)



print("begin kmeans")
cluster_num = 10
from sklearn.cluster import KMeans
X = np.array(all_embeddings)
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
class_list = kmeans.labels_
centers = kmeans.cluster_centers_

class vector_and_docid():
    def __init__(self,vector,doc_id):
        self.vector = vector
        self.doc_id = doc_id

group3 = []
for i in range(cluster_num):
    group3.append([])

for i in range(len(all_embeddings)):
    for j in range(cluster_num):
        if class_list[i]==j:
            group3[j].append(vector_and_docid(all_embeddings[i],i))

class_list3 = []
centers_list3 = []
for i in range(cluster_num):
    tmp_matrix = []
    for vec_and_docid in group3[i]:
        tmp_matrix.append(vec_and_docid.vector)
    X = np.array(tmp_matrix)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
    centers_list3.append(kmeans.cluster_centers_)
    class_list3.append(kmeans.labels_)

group9 = []
for i in range(cluster_num*cluster_num):
    group9.append([]) # 中间node，按index顺序放

for i in range(cluster_num): # 3group分成9
   for j in range(len(group3[i])):
       for k in range(cluster_num):
          if class_list3[i][j]==k:
              group9[i*cluster_num+k].append(group3[i][j])

class_list9 = []
centers_list9 = []
for i in range(cluster_num*cluster_num):
    tmp_matrix = []
    for vec_and_docid in group9[i]:
        tmp_matrix.append(vec_and_docid.vector)
    X = np.array(tmp_matrix)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
    class_list9.append(kmeans.labels_)
    centers_list9.append(kmeans.cluster_centers_)


group27 = []
for i in range(cluster_num*cluster_num*cluster_num):
    group27.append([]) # leaf，按index顺序放


for i in range(cluster_num*cluster_num): # 3group分成9
   for j in range(len(group9[i])):
       for k in range(cluster_num):
          if class_list9[i][j]==k:
              group27[i*cluster_num+k].append(group9[i][j])

# for i,j in enumerate(group27):
#     print(len(j))


# sample leaf to be mid node
print()
root_node = Node(np.zeros([300]),level=0,doc_id=-1)


for i in range(cluster_num):
    node3 = Node(centers[i],level=1,doc_id=-1)
    for j in range(cluster_num):
        node9 = Node(centers_list3[i][j],level=2,doc_id=-1)
        for k in range(cluster_num):
            node27 = Node(centers_list9[i * cluster_num + j][k],level=3,doc_id=-1)
            for vec_and_docid in group27[i * cluster_num * cluster_num + j * cluster_num + k]:
                node27.append_child(Node(vec_and_docid.vector,level=4,doc_id=vec_and_docid.doc_id))
            node9.append_child(node27)
        node3.append_child(node9)
    root_node.append_child(node3)

tree = Tree(root_node)
tree.breadth_travel()
print()
print("end kmeans")

query_vector = all_embeddings[0]
f3 = open("eval_mytree_center_10_final.tsv",mode="w",encoding="utf-8")

import time
start_time = time.time()
# doc_ids = tree.search(query_vector)
print()
count = 0
cosine_count100 = 0
for question in question2embedding:
    if question in all_question_with_gold_label:
        emb = question2embedding[question]
        doc_ids = tree.beam_search(emb,top_k=20)
        cosine_count100 += cosine_count
        cosine_count = 0
        if count%100==0:
            print(time.time()-start_time)
            print(count)
            print("cosine_count ", cosine_count100/100)
            cosine_count100 = 0
        count+=1
        for idx in doc_ids:
            if (question != all_questions[idx]):

                    f3.write(question)
                    f3.write("\t")
                    f3.write(all_questions[idx])
                    f3.write("\t")

                    if all_questions[idx] in true_pair and true_pair[all_questions[idx]] == question or \
                            question in true_pair and true_pair[question] == all_questions[idx] or \
                            all_questions[idx] in true_pair2 and true_pair2[all_questions[idx]] == question or \
                            question in true_pair2 and true_pair2[question] == all_questions[idx]:
                        f3.write("1\n")
                    else:
                        f3.write("0\n")



