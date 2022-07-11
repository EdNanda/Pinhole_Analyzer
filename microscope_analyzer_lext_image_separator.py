# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:04:42 2022

@author: HYDN02
"""
from glob import glob
import os
import re


folder = "C:\\Users\\HYDN02\\Seafile\\IJPrinting_Edgar-Florian\\20220615_NMP\\"

file_list = glob(folder+"*\\*.bmp")
file_list = sorted(file_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
# print(file_list)

counter = 0## counter for sets of pictures
count2 = 0## picture counter to 25
new_folder = "\\0_0_0"## dummy name

for c, file in enumerate(file_list):
    ## Extracting info from file
    name = file.split("\\")[-1].rsplit("_",2)[0]
    sample = file.split("\\")[-1]
    number = int(file.split("\\")[-1].rsplit("_",2)[1])
    


    ## TODO: count2 is experimental
    if number%26 != 0 and name == new_folder.split("\\")[-2].rsplit("_",1)[0]:
        count2 += 1
        print(count2)
    else:
        new_folder = f"{folder}{name}_{counter:02d}\\"
        
        ## Make new folder if none existant
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        print(counter, new_folder)
        counter += 1
        count2 = 0

    ## Move file to new location
    os.replace(file, new_folder+sample)