# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation and https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class Mapillary(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=66,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(1024, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Mapillary, self).__init__(ignore_label, base_size,crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()


        self.ignore_label = ignore_label
        
        self.color_list = [ [165, 42, 42], [0, 192, 0], [196,196,196], [190, 153, 153], [180, 165, 180], [90,120,150],
                            [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170],  [250, 170, 160],  [96, 96, 96],
                            [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100],  [70, 70, 70],
                            [150, 120, 90], [220, 20, 60], [255, 0, 0],  [255, 0, 100],  [255, 0, 200], [200, 128, 128], 
                            [255, 255, 255], [64, 170, 64], [230, 160, 50],  [70, 130, 180],  [190, 255, 255], [152, 251, 152], 
                            [107, 142, 35],  [0, 170, 30],  [255, 255, 128], [250, 0, 30], [100, 140, 180],  [220, 220, 220],
                            [220, 128, 128],  [222, 40, 40],  [100, 170, 30],  [40, 40, 40],  [33, 33, 33],  [100, 128, 160], 
                            [142, 0, 0],  [70, 100, 150],  [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80], 
                            [250, 170, 30],  [192, 192, 192],  [220, 220, 0],  [140, 140, 20], [119, 11, 32], [150, 0, 255], 
                            [0, 60, 100], [0, 0, 142], [0, 0, 90],  [0, 0, 230], [0, 80, 100], [128, 64, 64], 
                            [0, 0, 110], [0, 0, 70],  [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]
                          ]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root,'mapillary',item["img"])).convert('RGB')
#         image = image.resize((1536, 768), Image.BILINEAR)
#     label = label.resize((2048, 1024), Image.NEAREST)
        image = np.array(image)
        size = image.shape
        color_map = Image.open(os.path.join(self.root,'mapillary',item["label"])).convert('RGB')
#             image = image.resize((2048, 1024), Image.BILINEAR)
#         color_map = color_map.resize((1536, 768), Image.NEAREST)
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip, edge_pad=False,edge_size=self.bd_dilate_size, city=False)

#         return image.copy(), label.copy(), edge.copy(), np.array(size), name
        return image.copy(), label.copy(),  np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
