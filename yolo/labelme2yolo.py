# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:09:21 2024

@author: akara
"""

import json
import os
from glob import glob

# Labelme JSON dosyalarının ve çıktı YOLO dosyalarının dizinleri
labelme_dir = 'd:/deneme'  #labelme .json dosya yolu
yolo_dir = 'd:/deneme/yolo' #Yolo .txt dosyasının kayededileceği path
os.makedirs(yolo_dir, exist_ok=True)

# Sınıf isimleri
# Kendi sınıf isimlerinizi buraya ekleyin.
class_names = ["optic"]  

# YOLO formatına dönüştürme işlevi
def labelme_to_yolo(labelme_file):
    with open(labelme_file, 'r') as f:
        data = json.load(f)
    
    # Görüntü boyutları
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    yolo_labels = []
    for shape in data['shapes']:
        class_name = shape['label']
        if class_name not in class_names:
            continue
        class_id = class_names.index(class_name) 
        
        # Poligon koordinatlarını al
        points = shape['points']
        normalized_points = []
        for x, y in points:
            normalized_x = x / img_width
            normalized_y = y / img_height
            normalized_points.extend([normalized_x, normalized_y])
        # YOLO formatında her poligon için bir satır oluşturun
        yolo_label = f"{class_id} " + " ".join(map(str, normalized_points))
        yolo_labels.append(yolo_label)
    return yolo_labels

# Tüm JSON dosyalarını dönüştürün
for json_file in glob(os.path.join(labelme_dir, '*.json')):
    yolo_labels = labelme_to_yolo(json_file)
    
    # YOLO etiket dosyasını kaydedin
    txt_file = os.path.join(yolo_dir, os.path.basename(json_file).replace('.json', '.txt'))
    with open(txt_file, 'w') as f:
        f.write('\n'.join(yolo_labels))
