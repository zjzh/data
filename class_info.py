# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:54:26 2018

@author: dell
"""

class Param:
        param_type=""
        param_name=""
        param_description=""
        def __init__(self,param_type,param_name,param_description):
            self.param_type=param_type
            self.param_name=param_name
            self.param_description=param_description
class Return_value:
        return_type=[]
        return_name=[]
        return_description=""
        def __init__(self,return_type=[],return_description=[],return_name=""):
            self.return_type=return_type
            self.return_description=return_description
            self.return_name=return_name
'''
添加了方法声明项，主要用来解析swift的数据类型问题
'''
class Method:
    is_static=False
    class_name=""
    method_name=""
    method_description=""
    params=[]
    return_value=Return_value()
    def __init__(self,class_name,method_name,method_description,params,return_value,is_static=False,method_declar=""):
        self.class_name =class_name
        self.method_name=method_name
        self.method_description=method_description
        self.params=params
        self.return_value=return_value
        self.is_static=is_static
        self.method_declar=method_declar
class Var:
    class_name=""
    var_name=""
    var_type=""
    var_description=""
    def __init__(self,var_name="",var_type="",var_description="",is_static=False,var_declar=""):
        self.var_name=var_name
        self.var_type=var_type
        self.var_description=var_description
        self.is_static=is_static
'''
class_key 在java 是 interface or class
swift class struct enum protocol
java的类类型有 "class","interface","enum","other"
swift的类类型有 "class","struct","enum","protocol"

typealias 处理在类中出现的别名问题
'''
class Class_info:
    package_name=""
    class_name=""
    class_description=""
#==============================================================================
#     parent_list_class=[]
#     children_list_class=[]
#     interface_list_class=[]
#==============================================================================
    class_inherit_list=[]
    interface_list=[]
    subclass_list=[]
    Methods=[]
    Vars=[]
    def __init__(self, package_name,class_name, class_description,class_inherit,subclass,interface, Methods,Vars=None,class_type=None,typealias=[]):
        self.package_name=package_name
        self.class_name=class_name
        self.class_description=class_description
        self.class_inherit_list=class_inherit
        self.interface_list=interface
        self.subclass_list=subclass
        self.Methods=Methods
        self.Vars=Vars
        self.class_type=class_type
        self.typealias=typealias    
def convert_to_dicts(objs):
    '''把对象列表转换为字典列表'''
    obj_arr = []
     
    for o in objs:
        #把Object对象转换成Dict对象
        dict = {}
        dict.update(o.__dict__)
        obj_arr.append(dict)
     
    return obj_arr       