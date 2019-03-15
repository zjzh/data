# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:48:17 2019

@author: dell
"""
import json
with open(file_name) as f:
    data = json.load(f)
    class_description=data['class_description']
    for method in data['Methods']:
        method['method_description']
        method["return_value"]['return_description']
        for para in method["params"]:
            para_descrip=para['param_description']
    for var in data['Vars']:
        var_des=var['var_description']