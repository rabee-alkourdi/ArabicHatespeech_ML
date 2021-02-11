# -*- coding: utf-8 -*-

import pandas as pd
import json
import codecs
import stanfordnlp

'''
nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos", lang="ar")
sen = nlp("لعب الطفل في الشارع وسقط على الأرض")
sen.sentences[0].
#sen.sentences[0].
'''
nlp = stanfordnlp.Pipeline(lang="en")
data = pd.read_excel('../feature datasets/en/cleaned_data_with_stopwords.xlsx', encoding = 'ISO-8859-1', index_col=0)

#sen1 = nlp("انت حيو لعم عراق شرف طهر انو ذكر اسم اسم شباب عراق مو متل عني ازا انت شفت اخت طالع متل شرميط كيد مارح يهم لان لوطي تحي لاهل عراق بحبك كتير ختك سوري")
#sen = nlp("لعب الطفل في الشارع وسقط على الأرض")
#wrds = sen.sentences[0].words
#for w in wrds:
#    print(w.text + ": " + w.upos)
#s = sen.sentences[0].dependencies_string()
#l = sen.sentences[0].dependencies
#print(l[1].dependency_relation)
'''
# Dependencies
new_dict = dict()

for index, row in data.iterrows()
    comment = str(row['text'])
    idx = str(row['index'])

    new_dict[idx] = nlp(comment).sentences[0].dependencies_string().split ("\n")

with codecs.open("../feature datasets/dependency_dict_n.json", 'w', encoding='utf-8') as f:
    json.dump(new_dict, f, ensure_ascii=False)
 
'''

POS_LIST = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ', 'SYM','VERB','X']
REL_LIST_AR = ['acl', 'advcl', 'advmod', 'advmod:emph', 'amod', 'appos', 
'aux', 'aux:pass', 'case', 'cc', 'ccomp', 'conj', 
'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'discourse', 
'dislocated', 'fixed', 'flat:foreign', 'iobj', 'mark', 'nmod', 
'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:arg', 
'orphan', 'parataxis', 'punct', 'root', 'xcomp']

REL_LIST_EN = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 
'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'compound', 'compound:prt',
'conj', 'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'det:predet',
'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'flat:foreign', 'goeswith', 
'iobj', 'list', 'mark', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod', 
'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:npmod', 'obl:tmod',
'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

pos = pd.DataFrame(index=data.index, columns=POS_LIST)

for type in POS_LIST:
    pos[type] = 0
   
for index, row in data.iterrows():
    comment = str(row['text'])
    
    for wrd in nlp(comment).sentences[0].words:
        pos.loc[index,str(wrd.upos)] += 1

pos = pos.add_prefix('pos:')
pos.to_excel('../feature datasets/en/pos_features.xlsx', index_label="index")
