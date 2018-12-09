# RAS-ECCV18-caffemodel-to-pytorch
This repository converts a pretrained RAS caffemodel to a pytorch model. RAS denotes "Reverse Attention for Salient Object Detection" (https://github.com/ShuhanChen/RAS_ECCV18). 

If you use this code, you need to cite:
```
@inproceedings{chen2018eccv, 
  author={Chen, Shuhan and Tan, Xiuli and Wang, Ben and Hu, Xuelong}, 
  booktitle={European Conference on Computer Vision}, 
  title={Reverse Attention for Salient Object Detection}, 
  year={2018}
} 
```
How to use:
1. Install caffe_dss (https://github.com/Andrew-Qibin/caffe_dss)
2. Install pytorch-0.4.0
3. Download 'deploy.prototxt' and 'ras_iter_10000.caffemodel' from https://github.com/ShuhanChen/RAS_ECCV18.
4. Modify imname as the path of some testing image you use.
