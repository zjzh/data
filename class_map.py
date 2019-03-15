# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
import os,copy
import sys,json
import string
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
sys.path.append("..")
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import preprocess_sen_new,load_json,split_Uppercase
import numpy as np
def get_all_class_path(dir_parh):
    count_name_dict={}
    class_name_list={}
    class_count=0 
    path_list=[]
    for root,j,k in os.walk(dir_parh):
        root=root.replace('\\',"/")
        for k_each in k:
#            print("k_each: ",root,root[-1],k_each)
            k_each=k_each.replace('\\',"/")
            if len(k_each.split("."))>2:
                continue
            class_name_list[k_each.split(".")[0]]=class_count
            count_name_dict[class_count]=k_each.split(".")[0]
            class_count=class_count+1
            if root[-1]=="/":
                path_list.append(root+k_each)
#                print("path: ",root+k_each)
            else:
                path_list.append(root+"/"+k_each)
#                print("path: ",root+"/"+k_each)
    return  path_list,class_name_list,count_name_dict
'''
类信息：类名和类描述

每个方法API信息：方法描述，形参描述，返回值描述，形参名和方法名

is_signature  是否加入形参名，方法名
'''
def extrac_class_des(path,is_class_name_des=True,is_method_des=False,is_para_des=False,is_return_des=False,is_signature=False) :
#    print("path",path)
    
    json_data = load_json(path)
    class_all_des=""
    if is_class_name_des:
        class_name=split_Uppercase(json_data["class_name"])
        class_all_des=class_all_des+class_name+" "+json_data["class_description"]
    if is_method_des:#对每个方法提取相应的单元
        method_list=json_data["Methods"]
        for i,method in enumerate(method_list):
            class_all_des=class_all_des+" "+method["method_description"]
            if is_return_des:
#                print("len: ",method['return_value']['return_description'])
                if len(method['return_value']['return_description'])>0:
                    class_all_des=class_all_des+" "+method['return_value']['return_description'][0]
            if is_para_des:
                if len(method['params'])>0:
                    for i,para in enumerate(method['params']):
                        class_all_des=class_all_des+" "+para["param_description"]
                        if is_signature:
                            class_all_des=class_all_des+" "+" ".join(para["param_name"])
    return class_all_des
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
#    print("weight: ",weight,weight.shape)
    return weight,word
#    sFilePath = './tfidffile'
#    if not os.path.exists(sFilePath) : 
#        os.mkdir(sFilePath)
#
#    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
#    for i in range(len(weight)) :
#　　　　 print(u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------")
#        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
#        for j in range(len(word)) :
#            f.write(word[j]+"    "+str(weight[i][j])+"\n")
#        f.close()
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
'''
weight 每一行是一个文档   每一行的内容是每个词的tf-idf
index_sort 为每个类提取k个最大的tf-idf值
java_swift_words 存放重叠词的下标
java_swift_num 存放重叠的词的数目
java_swift_cos #存储向量的cos近似度

java_swift_map_index 相似度从大到小存储 每个java类对应的swift的类 下标
java_swift_map_words_index 按相似度从大到小存储 每个java类对应的swift的类的重叠词的下标

1.为每个文档提取k个关键词 index_sort(按tf-idf从大到小)
2.计算重叠的词的数目和重叠词的余弦值
这里的weight是java和swift的文档混合，java_num是分界点
for 每个java文档
    for 每个swift文档
        不同类的相交的关键词的数目
        不同类相交的关键词的向量列表
'''
def extract_keywords(top_k,java_weight,swift_weight):
#    index_sort=np.argsort(weight, axis=1)[::-1][:top_k]
#    print("java_weight.shape: ",java_weight.shape)
    swift_class_num=swift_weight.shape[0]
    java_class_num=java_weight.shape[0]
    index_java_sort=copy.deepcopy(np.argsort(java_weight, axis=1)[:,::-1][:,:top_k])
    index_swift_sort=copy.deepcopy(np.argsort(swift_weight, axis=1)[:,::-1][:,:top_k])
#    print("index_java_sort: ",index_java_sort.shape,index_swift_sort.shape)
    java_swift_num=np.zeros((java_class_num,swift_class_num))#存储重叠的词汇数目
    java_swift_cos=np.zeros((java_class_num,swift_class_num))#存储向量的cos近似度
    java_swift_words=[[] for i in range(len(java_weight))]#存储每个java类和swift类的重叠词汇的下标
    for i,java_class in enumerate(index_java_sort):
        for j,swift_class in enumerate(index_swift_sort):
            index_intersect=list(set(java_class)&set(swift_class))#存储相交的下标，即重叠的词汇
            if len(index_intersect)>0:
                java_swift_words[i].append(index_intersect)
                java_swift_num[i][j]=len(index_intersect)*-1
                java_vec=[java_weight[i][word_index] for word_index in index_intersect]
                swift_vec=[swift_weight[j][word_index] for word_index in index_intersect]
                java_swift_cos[i][j]=cos_sim(java_vec,swift_vec)*-1
            else:
                java_swift_words[i].append([])
                java_swift_num[i][j]=swift_class_num#这里设置成了最大值 ，实际应该是0
                java_swift_cos[i][j]=1#这里设置成了最大值1 ，实际应该是0
    java_swift_map_index=np.zeros((java_class_num,swift_class_num),dtype=int)#每一个元素存储swift的类下标
    java_swift_map_words_index=[]
    for row in range(java_swift_num.shape[0]):
        java_swift_map_index[row] = np.lexsort((java_swift_cos[row],java_swift_num[row]))
#        print("index: ",java_swift_words[row][0])
        java_swift_map_words_index.append( [java_swift_words[row][swift_index] for swift_index in java_swift_map_index[row]])
    return  index_java_sort,java_swift_map_index,java_swift_map_words_index
'''
1. 生成tf-idf的数据集，所有类文件路径
2. 训练tf-idf矩阵
3. 生成java到swift的类映射的排序
    确定每个类需要提取的关键词的数目
    按照关键词的重叠数目
    之后按照重叠关键词的余弦相似度
4. 对java类 提取前k个最相近的类写入文件
    映射的swift的类名+“------------”+关键词的数目 空格分开
'''
if __name__ == "__main__" : 
#    (allfile,path) = getFilelist(sys.argv)
    
    all_path_list=[]
    java_class_name_list={}#存放tf-idf的每一行下标对应的类名  下标索引 
    swift_class_name_list={}#存放tf-idf的每一行下标对应的类名  下标索引
    java_count_name_dict={}
    swift_count_name_dict={}
    java_dir_path="../android_api_doc_new/json_2/"#test
    swift_dir_path="../ios_api_doc/newest/all/"
    all_path_list,java_class_name_list,java_count_name_dict=get_all_class_path(java_dir_path)
    java_class_num=len(all_path_list)
    swift_path_list,swift_class_name_list,swift_count_name_dict=get_all_class_path(swift_dir_path)
    all_path_list.extend(swift_path_list)
    
    #制作文档集
    all_class_corpus=[]#每个元素是一个类
    for i,class_info_path in  enumerate(all_path_list):
        #def extrac_class_des(path,is_class_name_des=True,is_method_des=False,is_para_des=False,is_return_des=False,is_signature=False) :
        sen= extrac_class_des(class_info_path,True,True,True,True,True)
#        print("sen: ",sen)
        sen=preprocess_sen_new(sen,True)
#        print("sen: ",sen)
        lancaster_stemmer = LancasterStemmer()  
        input_str=word_tokenize(sen)
        sen=" ".join([lancaster_stemmer.stem(value) for value in input_str])
#        print("sen: ",sen)
#        if i>2:
#            break
        all_class_corpus.append(sen)
#    all_class_corpus=["Swift makes it easy to create arrays in your code using an array literal: simply surround a comma-separated list of values with square brackets. Without any other information, Swift creates an array that includes the specified values, automatically inferring the array’s Element type."
#                      ,"Arrays are one of the most commonly used data types in an app. You use arrays to organize your app’s data. Specifically, you use the Array type to hold elements of a single type, the array’s Element type. An array can store any kind of elements—from integers to strings to classes."]
##    for class_1 in all_class_corpus:
##        print("class: ",class_1)
    weight,word=Tfidf(all_class_corpus)
    
    top_k=10
#    map_java_list_name=["String"]
    map_java_list_name=["String","CharSequence","StringBuffer","StringBuilder","File","FileInputStream","FileOutputStream","ArrayList","LinkedList","Hashtable","HashSet"]
    map_java_to_swift=[]
    for i,value in enumerate(map_java_list_name):
        map_java_to_swift.append(list(weight[java_class_name_list[value]]))
    java_key_words_index,java_swift_map_index,java_swift_map_words_index=extract_keywords(top_k,np.array(map_java_to_swift),np.array(weight[java_class_num:]))
    
    
    dir_class_map_path="../data/class_map/"
    like_class_num=20
    for i,file_name in enumerate(map_java_list_name):
        print("write file : ",file_name)
        with open(dir_class_map_path+file_name+".txt", 'w',encoding='utf-8', errors='ignore') as f_w: 
            java_key_words=[]
            for index in java_key_words_index[i]:
                java_key_words.append(word[index])
            f_w.write(" ".join(java_key_words)+"\n")
            for j in range(like_class_num):  
                swift_index=java_swift_map_index[i][j]
                key_words_list=[]
                for key_word_index in java_swift_map_words_index[i][j]:
#                    print("java_swift_map_words_index[i][j]: ",java_swift_map_words_index[i][j],key_word_index)
                    key_words_list.append(word[key_word_index])
                f_w.write(swift_count_name_dict[swift_index]+": "+" ".join(key_words_list)+"\n")
#            f_w.close()
#    print("all_path_list: ",all_path_list)
