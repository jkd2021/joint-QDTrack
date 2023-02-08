#  -*-  coding:utf-8  -*- 
#  ------------------------------
#  Author: Kangdong Jin
#    Date: 2022/6/27 18:26
#     For: 
#  ------------------------------
import torch.utils.cpp_extension
import torch
from mmdet.apis import init_detector, inference_detector

print("Runtime CUDA version: {}".format(torch.utils.cpp_extension.CUDA_HOME))
print("Compile CUDA version: {}".format(torch.version.cuda))
print("Cuda is available: {}".format(torch.cuda.is_available()))

