import pickle
from tqdm import tqdm
f=open("/home/tingchen_fu/ConvDR/dataset/cast_shared/car_id_to_idx.pickle",'rb')
car2id=pickle.load(f)
f.close()

trec_file='/home/tingchen_fu/ConvDR/temp.trec'
fin=open(trec_file)
fout=open(trec_file+'_','w')
for line in tqdm(fin.readlines()):
    qid,_,docid,rank,score,__=line.strip().split('\t')
    if docid.startswith('MARCO_'):
        docid=docid[6:]
    else:
        assert docid in car2id.keys(), "docid {} not in car2id".format(docid)
        docid=car2id[docid]
    fout.write(qid+'\t'+_+'\t'+str(docid)+'\t'+str(rank)+'\t'+str(score)+'\t'+str(__)+'\n')

fin.close()
fout.close()

import os
os.system('mv '+trec_file+'_  '+trec_file)