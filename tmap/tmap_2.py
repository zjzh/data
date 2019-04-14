# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
import os,copy
import sys,json
import string
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
sys.path.append("..")
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import preprocess_sen_new,load_json,split_Uppercase,preprocess_sen_new_tmap,extract_method_api_sig,extract_var_api_sig
import numpy as np
import nltk
import scipy.spatial as sp 
from produce_map_set.statics_info import map_Statics,write_comparison_data,write_api_characteristic
def get_all_class_path(dir_parh):
 
    path_list=[]
    for s_file_name in os.listdir(dir_parh) :
        if os.path.isdir(dir_parh+s_file_name):
            continue
        path_list.append(dir_parh+s_file_name)
    
    return  path_list

'''
@author: ztw

取一段话的前n句，用于description字段的处理
'''
def first_n_sen(des, n):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(des)
    count = min(n, len(sentences))
    return_string = " ".join(sentences[:count])
    return return_string
'''
对若干文档，TF-IDF计算
corpus是list，每一个元素是一个文档
'''
def Tfidf(corpus) :
#    print("corpus",corpus)
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    
    word = vectorizer.get_feature_names() #所有文本的关键字
    weight = tfidf.toarray()              #对应的tfidf矩阵
#    print("word: ",len(word))
#    print("corpus length: ",len(corpus))
#    print("weight: ",weight,weight.shape)
    return weight,word 
'''
提取每个 api的描述信息
'''
def extrac_class_des(path,is_class_name_des=True,is_method_des=True,is_vars=False):
#    print("path",path)
    api_sig_list=[]
    cla_apis_list=[]
    json_data = load_json(path)
    
    if is_class_name_des:
            pack_name=" ".join(json_data["package_name"].split("."))
            class_name=split_Uppercase(json_data["class_name"])
            class_des=first_n_sen(json_data["class_description"], 5)
    if is_method_des:#对每个方法提取相应的单元
        method_list=json_data["Methods"]
        for i,method in enumerate(method_list):
            api_sig=extract_method_api_sig(method)  
            api_sig_list.append(api_sig)
            me_des=first_n_sen(method["method_description"], 2)+" "+split_Uppercase(method["method_name"])
            cla_apis_list.append(pack_name+" "+class_name+" "+class_des+" "+me_des)
    if is_vars:
        vars_list=json_data["Vars"]
        if vars_list!=None:
            for i,var in enumerate(vars_list):
                api_sig=extract_var_api_sig(var,json_data["class_name"])
                api_sig_list.append(api_sig)
                me_des=""
                if len(var['var_description']) > 0:
                    me_des =var['var_description'][0]
                cla_apis_list.append(pack_name+" "+class_name+" "+class_des+" "+me_des)
    return cla_apis_list,api_sig_list
'''
all_dst_api_list 每个元素是一个列表，java api对应到若干swift apis
all_dst_api_list=[]
for swift的每个api:
        确定所有出现的词word_list
        
for java 的每个api:
    确定top-k个词
    确定相应词的向量表示
    余弦相似度列表
    for 相应的swift api的信息：
            
        if java的top-k个词均出现在word_list：
            all_dst_api_list.append(swift_api_sig)
            计算cos 添加到余弦相似度列表
    排序cos
    获得排序后的api_sig_list列表
    all_dst_api_list追加
    
'''
def filter_words(x,word,swift_weight,i):
    if swift_weight[i][x]>1e-3:
        return word[x]
def extract_keywords(top_k,java_weight,swift_weight,word,j_api_sig_list,s_api_sig_list,all_swift_words):
#    index_sort=np.argsort(weight, axis=1)[::-1][:top_k]
#    print("java_weight.shape: ",java_weight.shape)
    all_dst_api_list=[]
    

    index_java_sort=copy.deepcopy(np.argsort(java_weight, axis=1)[:,::-1][:,:top_k])
#    j_weight_list=list(map(lambda x:java_weight[i][word.index(x)],swift_words))
#    print("i: ",index_java_sort)
#    index_swift_sort=copy.deepcopy(np.argsort(swift_weight, axis=1)[:,::-1][:,:top_k])

#        all_swift_weight.append(weight_list)
#        print("swift_word_list: ",word_list)
    flag=False
    for i,api in enumerate(index_java_sort):
        count=0
#        print("i: ",j_api_sig_list[i])
        word_list=list(map(lambda x:word[x],api))
        j_weight_list=list(map(lambda x:java_weight[i][x],api))
#        dict_java_api_words_tf_idf[j_api_sig_list[i]]=[word_list,weight_list]
#        print("word_list_1: ",word_list)
        cos_list=[]
        dst_api_sig_list=[]
        for index,swift_words in enumerate(all_swift_words):
            
            if  set(swift_words)>set(word_list):
                s_new_weight=list(map(lambda x:swift_weight[index][word.index(x)],word_list))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%重叠: ",set(swift_words),set(word_list))
                sim=1 - sp.distance.cdist([j_weight_list], [s_new_weight], 'cosine')
                cos_list.append(sim[0][0])
                dst_api_sig_list.append(s_api_sig_list[index])
                count=count+1
#                print("sim-1: ",sim)
                flag=True
#        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$count: ",count)
        
#        if flag:
#            break
         
        index_list=np.argsort(cos_list)[::-1]
        sort_dst_api_sig_list=[]
        for cos_index in index_list:
            sort_dst_api_sig_list.append(dst_api_sig_list[cos_index])
        all_dst_api_list.append(sort_dst_api_sig_list)
    return all_dst_api_list
#==============================================================================
#         for i,api in enumerate(index_swift_sort):
#             word_list=list(map(lambda x:word[x],api))
#             weight_list=list(map(lambda x:swift_weight[i][x],api))
#             dict_java_api_words_tf_idf[j_api_sig_list[i]]=[word_list,weight_list]
#==============================================================================
def make_acc(all_src_sig,all_dst_sig,base_sig_path,top_k_acc=10):

    
    map_statics=map_Statics("",[],[],[])
   
    map_statics
    acc_list=[]

        
    acc=0
    one_to_one_num=0
    base_sig=load_json(base_sig_path)  
#        print("base_sig: ",base_sig)
    yes_api_sig_list=[]
    no_api_sig_list=[]
    for i,src in enumerate(all_src_sig):
        if i<5:
            print("len_all_dst_sig: ",len(all_dst_sig[i]))
        if base_sig[src]:
            one_to_one_num=one_to_one_num+1
        if src in base_sig.keys() and base_sig[src] and (set(all_dst_sig[i][0:top_k_acc])&   set(base_sig[src])):                        
            acc=acc+1
            yes_api_sig_list.append(src)
        elif base_sig[src]:
            
                no_api_sig_list.append(src)
    map_statics.yes_api_sig_list.append(yes_api_sig_list)
    map_statics.no_api_sig_list.append(no_api_sig_list)
    
    if one_to_one_num!=0:
        print("one_to_one_num is zero",one_to_one_num)
        acc_list.append(float(acc)/one_to_one_num)
    else:
        print("one_to_one_num is zero")
        acc_list.append(0.0)
    map_statics.acc=acc_list
    
    return acc_list,map_statics    
    
if __name__ == "__main__" : 
    swift_dir_path="../ios_api_doc/newest/all_3_precess_new/"
    java_dir_path="../android_api_doc_new/json_3_class_type_precess_new/"
    base_sig_path="../data/map_baseline/"
    all_path_list=get_all_class_path(java_dir_path)
    java_class_num=len(all_path_list)
    swift_path_list=get_all_class_path(swift_dir_path)
    all_path_list.extend(swift_path_list)
    dict_java_api_words_tf_idf={}
    dict_swift_api_words_tf_idf={}
    #制作文档集
    all_api_sig_list=[]
    j_all_api_sig_list=[]
    api_j_num=0
    all_apis_corpus=[]#每个元素是一个类
    for i,class_info_path in  enumerate(all_path_list):

        if(i<java_class_num):
            cla_apis_list,api_sig_list = extrac_class_des(class_info_path, True, True,  False)
            api_j_num=api_j_num+len(api_sig_list)
            j_all_api_sig_list.extend(api_sig_list)
#            all_apis_corpus.extend(cla_apis_list)
        else:
            cla_apis_list,api_sig_list= extrac_class_des(class_info_path,True,True,True)
#            all_apis_corpus.extend(cla_apis_list)
        all_api_sig_list.extend(api_sig_list)
        for sen in cla_apis_list:
            sen=preprocess_sen_new_tmap(sen,True)
            snowball_stemmer = SnowballStemmer('english')
            input_str=word_tokenize(sen)
            sen=" ".join([snowball_stemmer.stem(value) for value in input_str])
            all_apis_corpus.append(sen)
    weight,word=Tfidf(all_apis_corpus)
    all_swift_words=[]
#    all_swift_weight=[]
    swift_weight=np.array(weight[api_j_num:])
    s_new_weight_list=[]
    for i,api in enumerate(all_api_sig_list[api_j_num:]):
        word_list=all_apis_corpus[i+api_j_num].split(" ")
        new_word_list=filter(lambda x: x in word, word_list)
        index_list=list(map(lambda x:word.index(x),new_word_list))
        s_new_weight_list.append(list(map(lambda x:swift_weight[i][x],index_list)))
#        word_list=list(map(lambda x:filter_words(x,word,swift_weight,i), [j for j in range(swift_weight.shape[1])]))
        print("len(word_list): ",len(word_list))
#        weight_list=list(map(lambda x: swift_weight[i][x],api))
        all_swift_words.append(word_list)
    map_java_list_name=["String","StringBuilder","File","FileInputStream","FileOutputStream","ArrayList","LinkedList","HashSet","HashMap"]
    top_k=5
    for i,value in enumerate(map_java_list_name):
        src_api_sig_list=[]
        dst_api_sig_list=[]
        map_java_to_swift=[]
        for index, api_sig in enumerate(j_all_api_sig_list):
            class_name=api_sig.split(" ")[0]
            if class_name==value:
                src_api_sig_list.append(api_sig)
                map_java_to_swift.append(weight[index])
        print("**************extract_keywords***************")
        dst_api_sig_list=extract_keywords(top_k,np.array(map_java_to_swift),swift_weight,word,src_api_sig_list,all_api_sig_list[api_j_num:],all_swift_words)
        acc_list,map_statics=make_acc(src_api_sig_list,dst_api_sig_list,base_sig_path+value+".json",10)
        print("class_name acc: ",value,acc_list)
#        break