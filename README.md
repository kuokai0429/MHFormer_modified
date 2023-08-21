# A modified version of MHFormer

Modified from official implementation of CVPR 2022 paper (**MHFormer**) Multi-Hypothesis Transformer for 3D Human Pose Estimation.

> **MHFormer** ([Github Link](https://github.com/Vegetebird/MHFormer)) <br>
[Multi-Hypothesis Transformer for 3D Human Pose Estimation](https://arxiv.org/pdf/2111.12707),            
> Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, Luc Van Gool,        
> *In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022*


## 環境安裝 (部分套件)

- Create a conda environment: ```conda create -n mhformer python=3.9```
- ```pip3 install -r requirements.txt```
  
## 資料集設置 Dataset Setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```


## 預訓練權重 Pretrained Model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing), please download it and put it in the './checkpoint/pretrained' directory. 

## 模型評估 Evaluate

To test on a 81-frames pretrained model on Human3.6M:

```bash
python main.py --test --previous_dir 'checkpoint/0512_1329_17_81' --frames 81
( python main.py --test --previous_dir 'checkpoint/pretrained/351' --frames 351 )
```


## 模型訓練 Training

To train a 81-frames model on Human3.6M:

```bash
python main.py --frames 81 --batch_size 256
( python main.py --frames 351 --batch_size 128 )
```


## 模型推論 Run Inference
First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 
Then, you need to put your in-the-wild videos in the './demo/video' directory. 

Run the command below:
```bash
python demo/vis_longframes.py --video f1_3m.mp4
( python demo/vis.py --video sample_video.mp4 )
```

Sample demo output:

<p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>


## 模型結果比對 BenchMark
To generate benchmark results for ground truth, mhformer and poseformer predictions.

Run the command below:
```bash
python demo/vis_benchmark.py --action Walking --subject S11 --with_gt
( python demo/vis_benchmark.py --action "Phoning 2" --subject S11 --with_gt )
```

## 其他

Check Note.txt for detailed command guidance.


## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
