---
title: 'HTCA'
disqus: readme
---

Hybrid Task Cascade with Attention: A New Framework for Instance Segmentation
===

[Hybrid Task Cascade with Attention](https://https://github.com/rmanzanedo/HTCA-Hybrid_Task_Cascade_with_Attention/blob/master/Hybrid_Task_Cascade_with_Attention.pdf) (HTCA) is a new framework that uses EfficientNet with BiFPN as backbone, and adds Spatial Attention Modules (SAM) to the Hybrid Task Cascade(HTC) branches to enhance the previous results on COCO datasets.

This repository also has an analysis of the performance of multiple State-of-The-Art models with EfficientNet with the BiFPN as backbone.


## Instalation guide

It requires Python 3.8+, CUDA 11.1+ and PyTorch 1.9+.

Then install [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html),
```
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
```

Install [mmdet](https://mmdetection.readthedocs.io/en/stable/get_started.html) from the source

```
cd mmdetection
!pip install -e .
```

Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) from source

```
cd ..
python -m pip install -e detectron2
```

Install EfficientDet from source

```
cd efficientdet
pip install -r requirements.txt
```



Dataset
---
All models require [COCO dataset](https://cocodataset.org/#home) and HTC also requires [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
datasets
├── coco
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   ├── test2017
|   ├── stuffthingmaps

HTCA
├── centermask2
├── checkpoints
├── detectron2
├── efficientdet
│   ...

```




<!-- ## Appendix and FAQ -->



## Citing HTCA

If you use HTCA in your research or wish to refer to the baseline results published, please use the following BibTeX entry.

```BibTeX
@misc{rm2022htca,
  author =       {Ricardo Manzanedo},
  title =        {HTCA: Hybrid Task Cascade with Attention},
  howpublished = {\url{44444}},
  year =         {2022}
}
```

:::info
**Find this document incomplete?** Leave a comment!
:::
