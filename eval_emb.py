
import eval_step.util_function as util

f = open("eval_loop_final.tsv",mode="r",encoding="utf-8")


#[0.53318017 0.45739494 0.53811161 0.58476755 0.53510438] 10 tree
#[0.35372922 0.30016313 0.35422035 0.38251991 0.35257775] string
#[0.64218761 0.56315515 0.6390096  0.68873115 0.63567972] bm25
#[0.60947804 0.52316423 0.61459753 0.66771929 0.61104056] loop
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
lines = f.readlines()
last_left = ""
gt_doc_ids = set()
pred_doc_ids = []
batch_index = 0
total_result = [0,0,0,0,0]
total_count = 0
top1_count = 0
for i,line in enumerate(lines):
    left,right,label = line.split("\t")
    if i%19==0 and i>0:
        if len(gt_doc_ids)==0:
            gt_doc_ids.add(21)
        pred_doc_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        result = util.metrics(
            gt=gt_doc_ids, pred=pred_doc_ids, metrics_map=METRICS_MAP)
        total_result += result
        total_count +=1

        batch_index = 0
        gt_doc_ids = set()
        pred_doc_ids = []

        batch_index += 1
        if label.rstrip() == "1":
            gt_doc_ids.add(batch_index)
            top1_count += 1
        pred_doc_ids.append(batch_index)
    else:
        batch_index += 1
        if label.rstrip()=="1":
            gt_doc_ids.add(batch_index)
        pred_doc_ids.append(batch_index)


    last_left = left
print(top1_count)
print(total_result/total_count)



