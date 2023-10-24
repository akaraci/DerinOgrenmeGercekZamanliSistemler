# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:38:56 2023

@author: akara
"""
##Labelimg ile oluşturulan xml dosyasında yerel path dosyaya eklenmektedir.
#Bunu düzeltmek için aşağıdaki kod kullanılabilir.

import xml.etree.ElementTree as ET

import os
#tarama=os.scandir("origa/train/Annotations")
tarama=os.scandir("origa/validation/Annotations")
for belge in tarama:
    mytree = ET.parse(belge.path)
    myroot = mytree.getroot()      
    myroot[2].text=myroot[1].text
    mytree.write(belge.path)

