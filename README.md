### mmcv=2.1.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
### torch=2.1.0
### cuda=11.8
pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
### mmseg=0.22.0
pip install mmsegmentation
pip install -v -e .
### gdal
conda install GDAL
### ftfy and regex
pip install ftfy
pip install regex



## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{11021577,
  author={Fu, Siming and Dong, Sijun and Meng, Xiaoliang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={WSCDNet: A Window Structural Similarity Guided Deep Feature Recalibration Method for Remote Sensing Image Change Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Feature extraction;Remote sensing;Transformers;Information exchange;Computer architecture;Computational modeling;Accuracy;Semantics;Noise;Neural networks;Change detection;differential feature;feature fusion;structural similarity index},
  doi={10.1109/JSTARS.2025.3576127}}

```
## Acknowledgments

 Our code is inspired and revised by [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation),  [timm](https://github.com/huggingface/pytorch-image-models). Thanks  for their great work!!  
