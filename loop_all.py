

f = open("bert_code/run/datasets/quora/test_all.tsv",mode="r",encoding="utf-8")

all_question_with_gold_label = set()
true_pair = {}
true_pair2 = {}
count = 0
for line in f:
    label,q1,q2,_ = line.rstrip().split("\t")
    if label=="1":
        true_pair[q1] = q2
        true_pair[q2] = q1
        true_pair2[q2] = q1
        true_pair2[q1] = q2
        all_question_with_gold_label.add(q1.rstrip())
        all_question_with_gold_label.add(q2.rstrip())
    count+=1

import json
count = 0
f = open("all_questions_final.jsonl",mode="r",encoding="utf-8")
lines = f.readlines()
all_questions = []
for line in lines:
    all_questions.append(line.rstrip())
    # if count > 1000:
    #     break
    # count += 1

count = 0
f = open("all_embeddings_final.jsonl",mode="r",encoding="utf-8")
lines = f.readlines()
all_embeddings = []
for line in lines:
    all_embeddings.append(json.loads(line.rstrip()))
    # if count > 1000:
    #     break
    # count += 1

count = 0
from scipy import spatial
import numpy as np
# X = np.array(all_embeddings)
# tree = spatial.KDTree(data=X)
import time
f3 = open("eval_loop_final.tsv",mode="w",encoding="utf-8")
closest_n=20
start_time = time.time()


import scipy.spatial
for question, question_embedding in zip(all_questions, all_embeddings):

  if question in all_question_with_gold_label:

      distances = scipy.spatial.distance.cdist([question_embedding], all_embeddings, "cosine")[0]

      results = zip(range(len(distances)), distances)
      results = sorted(results, key=lambda x: x[1])
      if count % 100 == 0:
          print(time.time() - start_time)
          print(count)

      for idx, distance in results[0:closest_n]:
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
      count += 1
    # distance  = tree.query(question_embedding,k=closest_n)
    # tmp = sorted(distance, key=lambda x: x[1])
    # if count%100==0:
    #     print(time.time()-start_time)
    #     print(count)
    # for idx,score in zip(tmp[1],tmp[0]):
    #     idx = int(idx)
    #     if (question != all_questions[idx]):
    #
    #             f3.write(question)
    #             f3.write("\t")
    #             f3.write(all_questions[idx])
    #             f3.write("\t")
    #
    #             if all_questions[idx] in true_pair and true_pair[all_questions[idx]] == question or \
    #                         question in true_pair and true_pair[question] == all_questions[idx] or \
    #                         all_questions[idx] in true_pair2 and true_pair2[all_questions[idx]] == question or \
    #                         question in true_pair2 and true_pair2[question] == all_questions[idx]:
    #                 f3.write("1\n")
    #             else:
    #                 f3.write("0\n")

