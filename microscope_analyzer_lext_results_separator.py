# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:04:42 2022

@author: HYDN02
"""
from glob import glob
import os
import re


folder = "C:\\Users\\HYDN02\\Seafile\\IJPrinting_Edgar-Florian\\20220615_NMP\\"

results_img = ["bubble_area","bubble_counts","collage"]

for res in results_img:
    
    ## Create folder
    res_folder = folder+"Results-"+res+"\\"
    os.makedirs(res_folder)
    
    ## Make a list of all images
    file_list = glob(folder+"*\\*"+res+".png")

    ## Move images to results folder
    for file in file_list:
        filename = file.split("\\")[-1]
        os.replace(file, res_folder+filename)
