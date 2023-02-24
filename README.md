# Implementations of Video Instance Segmentation methods on amodal video datasets 

## Datasets:
[SAIL-VOS](https://sailvos.web.illinois.edu/_site/index.html) & SAIL-VOScut (videos split into video-cuts without abrupt scene change)

amodal annotations: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; visible annotations:

<img src="figures/amo_anno.png" width="350"> &emsp; &emsp; <img src="figures/vis_anno.png" width="350">

## Amodal & Visible: QDTrack-mots-joint(+)
Using joint construction of the functional heads (Mask Heads / BBox Heads) in the original Mask R-CNN architecture of [QDTrack-mots](https://github.com/SysCV/qdtrack) for joint training research.

&emsp; QDTrack-mots-joint testing results:

amodal results: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; visible results:

<img src="figures/amo_joint.png" width="350"> &emsp; &emsp; <img src="figures/vis_joint.png" width="350">

## Amodal / Visible: (Amodal)QDTrack-mots & (Amodal)PCAN
Please refer to [QDTrack](https://github.com/SysCV/qdtrack) & [PCAN](https://github.com/SysCV/pcan) for details.

## Installation

## Usages

## References
```
@inproceedings{pang2021quasi,
  title={Quasi-dense similarity learning for multiple object tracking},
  author={Pang, Jiangmiao and Qiu, Linlu and Li, Xia and Chen, Haofeng and Li, Qi and Darrell, Trevor and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={164--173},
  year={2021}
}
@inproceedings{pcan,
  title={Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation},
  author={Ke, Lei and Li, Xia and Danelljan, Martin and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
