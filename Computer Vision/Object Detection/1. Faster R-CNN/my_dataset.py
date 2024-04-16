from torch.utils.data import Dataset
import os
import torch
import json
import xml.etree.ElementTree as ET
from xml import etree
from xml.etree.ElementTree import XML, fromstring

class VOCDataset(Dataset):
    """ Read and Parse VOC Dataset"""
    def __init__(self, voc_root, transforms, tran_set=True):
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.img_root = os.path.join(self.root, "JPEGImages")

        if tran_set:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "train.txt")
        else:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "val.txt")

        with open(txt_list) as read:

            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # Read Classes Index
        try:
            json_file = open("./passcal_voc_class.json", "r")
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(1)

        self.transfroms = transforms


        pass

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        xml_path = self.xml_list[index]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml_tree = etree.fromstring(xml_str)
        pass