from mmdet.datasets import DATASETS

from .coco_video_dataset_no_joint import CocoVideoDatasetNoJoint


@DATASETS.register_module()
class BDDVideoDataset(CocoVideoDatasetNoJoint):

    CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',  'bicycle')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
