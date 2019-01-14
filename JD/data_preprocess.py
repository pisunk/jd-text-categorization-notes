
# coding: utf-8

# # 对JD商品评价进行分类

# In[2]:


import numpy as np
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# In[3]:


#每句话最大长度
MAX_LEN=64


# In[11]:


#数据清理
def clean(sent):
    punctuation_remove = u'[、：，？！。；……（）『』《》【】～!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sent = re.sub(r'ldquo', "", sent)
    sent = re.sub(r'hellip', "", sent)
    sent = re.sub(r'rdquo', "", sent)
    sent = re.sub(r'yen', "", sent)
    sent = re.sub(r'⑦', "7", sent)
    sent = re.sub(r'(， ){2,}', "", sent)
    sent = re.sub(r'(！ ){2,}', "", sent)  # delete too many！，？，。等
    sent = re.sub(r'(？ ){2,}', "", sent)
    sent = re.sub(r'(。 ){2,}', "", sent)
    sent = re.sub(r'\n', "", sent)
    # sent=sent.split()
    # sent_no_pun=[]
    # for word in sent:
    #     if(word!=('，'or'。'or'？'or'！'or'：'or'；'or'（'or'）'or'『'or'』'or'《'or'》'or'【'or'】'or'～'or'!'or'\"'or'\''or'?'or','or'.')):
    #         sent_no_pun.append(word)
    # s=' '.join(sent_no_pun)
    sent = re.sub(punctuation_remove, "", sent)  # delete punctuations
    #若长度大于64，则截取前64个长度
    if(len(sent.split())>MAX_LEN):
        s=' '.join(sent.split()[:MAX_LEN])
    else:
        s = ' '.join(sent.split())  # delete additional space

    return s


# In[12]:


good_data=open('D:/opt/JD_nlp/data/good_cut_jieba.txt','r',encoding='utf-8').readlines()


# In[13]:


good_data


# In[14]:


good_data=[clean(line) for line in good_data]


# In[15]:


good_data


# In[27]:


# word_to_inx={}
# len(word_to_inx)


# In[28]:


# word_to_inx['s'] = len(word_to_inx)


# In[29]:


# word_to_inx


# In[30]:


# len(word_to_inx)


# In[32]:


#文本处理写成函数，对word先做labelencode，再对句子做labelencode，句子维度为MAX_LEN

#创建词典,用于word2index+index2word
word_to_inx={}
inx_to_word={}
#为数据生成label
def get_data():
    good_data=open('D:/opt/JD_nlp/data/good_cut_jieba.txt','r',encoding='utf-8').readlines()
    good_data=[clean(line) for line in good_data]
    good_data_label=[0 for i in range(len(good_data))]

    bad_data = open('D:/opt/JD_nlp/data/bad_cut_jieba.txt', 'r', encoding='utf-8').readlines()
    bad_data = [clean(line) for line in bad_data]
    bad_data_label = [1 for i in range(len(bad_data))]

    mid_data = open('D:/opt/JD_nlp/data/mid_cut_jieba.txt', 'r', encoding='utf-8').readlines()
    mid_data = [clean(line) for line in mid_data]
    mid_data_label = [2 for i in range(len(mid_data))]

    #total feature+label
    data=good_data+bad_data+mid_data
    data_label=good_data_label+bad_data_label+mid_data_label
    # print(data[0:5])
    # print(data_label[0:5])

    #创建一个word的set
    vocab=[word for s in data for word in s.split()]
    vocab=set(vocab)
    #print(vocab)

    for word in vocab:
        inx_to_word[len(word_to_inx)]=word
        word_to_inx[word]=len(word_to_inx)

    data_id=[]

    for sent in data:
        s_id=[]
        for word in sent.split():
            s_id.append(word_to_inx[word])
        #sent_labelincode 把sent补充为MAX_LEN
        s_id=s_id+[0]*(MAX_LEN-len(s_id))
        data_id.append(s_id)
        
    # 求出句子的最大长度
    # max_len=0
    # for s in data_id:
    #     if len(s)>max_len:
    #         max_len=len(s)
    # print(max_len)
    # 318
    # print(data_id[5:10])
    # print(len(data_id),' ',len(data_label))
    return data_id,data_label,word_to_inx,inx_to_word


# In[39]:


#获取两个字典
def get_dic():
    _,_,word_to_inx,inx_to_word=get_data()
    return word_to_inx,inx_to_word


# In[43]:


#将数据转化为tensor
def tensorFromData():
    #获取sent_labelencode和label
    data_id,data_lable,_,_=get_data()
    data_id_train, data_id_test, data_label_train, data_label_test = train_test_split(data_id, data_lable,
                                                                                      test_size=0.2,
                                                                                      random_state=233)
    data_id_train=torch.LongTensor(data_id_train)
    data_id_test=torch.LongTensor(data_id_test)
    data_label_train=torch.LongTensor(data_label_train)
    data_label_test=torch.LongTensor(data_label_test)
    return data_id_train,data_id_test,data_label_train,data_label_test
    # return data_id,data_lable

'''
pytroch创建自己的Dataset和Dataloader
https://blog.csdn.net/york1996/article/details/84141034
'''
class TextDataSet(Dataset):
    def __init__(self,inputs,outputs):
        self.inputs,self.outputs=inputs,outputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        return self.inputs[index],self.outputs[index]

