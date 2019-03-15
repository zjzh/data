# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:20:31 2018

@author: dell
"""
import os,json,re,copy
import numpy as np
#import gensim
#from api_map import get_method_signature
import pandas as pd
#def mak_dic_word():
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords   
'''
注意这里提取method的条件应该和def extract_method_list(path):一致’
使得返回的list长度一致
'''
def extract_method_api_sig(method):
    class_name=method["class_name"]
    return_type=method['return_value']['return_type']
    method_name=method["method_name"]
#    if method_name=="character":
#        print("return_type: ",return_type)
    param_list=get_par_type_name_list(method)
    api_sig=Api_sig(class_name,return_type, method_name, param_list)
    return api_sig 
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
def pre_sen(sen):
    sentences=re.sub("CF|NS|AU|CAF|QC|CG|UI|CA|AV|\\)|\\(","",sen)
    result=re.split(" |_|-|—|:",sentences)
#    print("result: ",result)
    result=map(split_Uppercase,result)#将词以大写字母开始后加小写字母直至遇到大写字母,全是大写则不分开
   
    result=map(delete_num,result)
    input_str_old=copy.deepcopy(list(result))
    result=" ".join(input_str_old)
    sentences=result.lower()
    table = str.maketrans("", "", string.punctuation)
    result = sentences.translate(table)
#        print("sentence punctuation: ",result)
#        result=result.strip()#去除空格
    input_str=word_tokenize(result)
    lemmatizer=WordNetLemmatizer()
    input_str=map(lambda x:lemmatizer.lemmatize(x),input_str)
    list1=copy.deepcopy(list(input_str))
#    print(list1)
    return " ".join(list1)
'''
将一些单词进行大小写转换，去掉数字，标点符号，停用词
'''
def preprocess_sen_new(sen,if_stop_words=False):
    sentences=re.sub("CF|NS|AU|CAF|QC|CG|UI|CA|AV|\\)|\\(","",sen)
    result=re.split(" |_|-|—|:",sentences)
    
    result=map(split_Uppercase,result)#将词以大写字母开始后加小写字母直至遇到大写字母,全是大写则不分开
   
    result=map(delete_num,result)
    input_str_old=copy.deepcopy(list(result))
    result=" ".join(input_str_old)
    
    sentences=result.lower()
    table = str.maketrans("", "", string.punctuation)
    result = sentences.translate(table)
#    print("result: ",result)
    if if_stop_words:
        stopworddic = set(stopwords.words('english'))
        stopworddic.add("return")
        result = [i for i in result.split(" ") if i not in stopworddic ]
        return " ".join(result)
    return result
def split_Uppercase(x):  
    #这里假设以小写开始，之后全是大写开始拆分  stringAbdBcd
    x=x.replace(".","")
    upp=re.findall('[A-Z][^A-Z]+',x)
    
    if len(upp)==0:
        return x.lower()
#    print("split_Uppercase: ",x.replace("".join(upp),"")+" "+" ".join(upp))
    return (x.replace("".join(upp),"")+" "+" ".join(upp)).strip().lower()
#将字符串按大小写拆分为若干字符
def split_words(json_str):
    sentence=re.sub("CF|NS|AU|CAF|QC|CG|UI|CA|AV|\\)|\\(","",json_str)
    result_list=copy.deepcopy(split_Uppercase(sentence).split(" "))
    return result_list
def delete_num(x):

    if re.search("\d",x)==None:
        return x
    else:
        return ""
def prepro(sentences):
    sentences=sentences.strip()
    sentences=re.sub("’\w|’|\\([\s\S]*?\\)|>|<"," ",sentences)#去除掉 ‘ 去除掉括号里面的内容 主要是因为方法名的影响
#    sentences=re.sub('“([^“]*)”',"",sentences)
    sentences=re.sub("CF|NS|AU|CAF|QC|CG|UI|CA|AV|\\)|\\(","",sentences)
    result=re.split(" |_|-|—|:|\n",sentences)
#    print("result: ",result)
    result=map(split_Uppercase,result)#将词以大写字母开始后加小写字母直至遇到大写字母,全是大写则不分开
   
    result=map(delete_num,result)
#    
    input_str_old=copy.deepcopy(list(result))
#    input_str_old=map(lambda x: x+" ",input_str_old)
#    print("pre**************: "," ".join(input_str_old).split(" "))
    return " ".join(input_str_old).split(" ")
def get_class_datas(dir_path):
    data_list=[]
    for file_name in os.listdir(dir_path):
        if not os.path.isdir(dir_path+file_name):
            with open(dir_path+file_name, 'r') as f:
                data = json.load(f)
                data_list.append(data)
    return data_list
def get_par_type_name_list(method):
    result_list=[]
    for para in method["params"]:
        result_list.append((para['param_type'],para['param_name']))
    return result_list
def get_method_params_name_str(method):
    param_names=""
    for para in method["params"]:
        for name in para['param_name']:
            param_names=param_names+" "+name
     
    return param_names
def get_method_signature(method):
#    sentence_descrip=""
#    sentence_descrip=sentence_descrip+method['method_description']+"\nreturn:"+''.join(method["return_value"]['return_description'])+"\nparams:"
    sen_signature=""
    sen_signature=sen_signature+method['method_name']+" ( "
#    param_name_list=[]
    for para in method["params"]:
        sen_signature=sen_signature+para['param_type']+','+str(para['param_name'])+" "
#        param_name_list.append(para['param_name'])
#        sentence_descrip=sentence_descrip+para['param_description']+"\n"
#    if 'return_type' in method.keys():
    sen_signature=sen_signature+" ) "+str(method["return_value"]['return_type'])
    return sen_signature
def get_method_description(method):
    sentence_descrip=""
    meth_des="NO METHOD DESCRIPTION"
    if method['method_description']!="":
        meth_des=method['method_description'].split(".")[0]
    sentence_descrip=sentence_descrip+meth_des+"\n"+''.join(method["return_value"]['return_description'])+"\n"
    for para in method["params"]:
        sentence_descrip=sentence_descrip+para['param_description']+"\n"
    return sentence_descrip
def get_method_description_read(method):
    sentence_descrip=""
    meth_des="NO METHOD DESCRIPTION"
    if method['method_description']!="":
        meth_des=method['method_description'].split(".")[0]
    sentence_descrip=sentence_descrip+meth_des+"\nRETURN:"+''.join(method["return_value"]['return_description'])
    for para in method["params"]:
        sentence_descrip=sentence_descrip+"\nPARAM: "+str(para['param_description'])
    return sentence_descrip
def get_method_list(path):
    json_data=load_json(path)
    method_list=json_data["Methods"]
    return method_list
'''
这里是连续写若干个sheet表到xls文件
column_list  列名
data_list [[sheet1],[sheet2]]  sheet1: 每一行的数据[colomn1,col2,col3]
sheet_name_list  sheet名
如果transpose 则list的每个元素是写入到xlxs的行数据
否则 list的每个元素是写入到xlsx的列数据
'''

def write_xls(dir_path,file_name,column_list,data_list,sheet_name_list,transpose=True):
    print("file_name: ",column_list,len(data_list))
    writer = pd.ExcelWriter(dir_path+file_name)
    for i in range(len(data_list)):
        if transpose:
            each_sheet_data=np.array(data_list[i]).T
        else:
            each_sheet_data=np.array(data_list[i])
        print("shape: ",each_sheet_data.shape)
        data_dict={}
        for j in range(each_sheet_data.shape[0]):
            data_dict.setdefault(column_list[j],each_sheet_data[j])
        df1 = pd.DataFrame(data=data_dict)
        df1.to_excel(writer,sheet_name_list[i][:31],index=False,columns=column_list)
    writer.close()
def dump_json(json_file,data):
    with open(json_file,"w") as f:
            json.dump(data,f) 
    
def load_json(json_file):
    fileObject = open(json_file, 'r')
    data=json.load(fileObject)
    fileObject.close()
    return data
def append_one_colum(dir_path,file_name,col_list,col_data,sheet_name):
    
    df = pd.read_excel(dir_path+file_name)
    old_column=df.columns.tolist()
    new_col=[]
    for col in old_column:
        new_col.append(col)
    new_col.extend(col_list)
    print(new_col)
    writer = pd.ExcelWriter(dir_path+file_name)
    data={}
    for i,col in enumerate(col_list):
       data.setdefault(col_list[i],col_data[i]) 
    df2=pd.concat([df, pd.DataFrame(columns=col_list,data=data)], axis=1)
    df2.to_excel(writer,sheet_name,index=False,columns=new_col)
    writer.close()
