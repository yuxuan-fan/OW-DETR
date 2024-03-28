# OW-DETR: Open-world Detection Transformer (CVPR 2022)

[`Paper`](https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.pdf) [`Video`](https://www.youtube.com/watch?v=saO8RHCpnaY) [`slides`](https://docs.google.com/presentation/d/1I1OyoRbKqvwB_dSLM8ybSXrB74crPX2a9R9yyWvABDc/edit?usp=sharing) [`summary slide`](https://docs.google.com/presentation/d/1zABTrvkaYlqb7u6xWv1JPIHFdsAyRkAggnmj33kuwsE/edit?usp=sharing)

#### [Akshita Gupta](https://akshitac8.github.io/)<sup>\*</sup>, [Sanath Narayan](https://sites.google.com/view/sanath-narayan)<sup>\*</sup>, [K J Joseph](https://josephkj.in), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) ####

(:star2: denotes equal contribution)

# Introduction






# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n owdetr python=3.7 pip
conda activate owdetr
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Backbone features

Download the self-supervised backbone from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


# Dataset & Results
下载coco数据集
要分别下载coco数据集的训练集和验证集，然后将他们移动到JEPGImages文件夹下。
```
#下载coco数据集
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/zips/val2017.zip
```

解压移动
由于train2017中的文件很多，所以我直接把train解压到目标目录下并改名成JEPGImages，然后把val2017解压后移动过去。

```
#解压到指定文件夹，然后改名
!unzip /content/drive/MyDrive/train2017.zip -d /content/drive/MyDrive/OW-DETR/data/OWDETR/VOC2007
!unzip /content/drive/MyDrive/val2017.zip -d /content/drive/MyDrive/OW-DETR/data/coco
!mv /content/drive/MyDrive/OW-DETR/data/coco/val2017/*.jpg /content/drive/MyDrive/OW-DETR/data/OWDETR/VOC2007/JPEGImages/.
```

但是解压到云盘有一个很大的问题，由于文件很大，解压到云盘会很慢，因为云盘主要用途是用来存东西的，而不是用来跑代码的，所以我建议直接解压到/content目录下
```
#解压到指定文件夹，然后改名
!unzip /content/drive/MyDrive/train2017.zip -d /content//OW-DETR/data/OWDETR/VOC2007
!unzip /content/drive/MyDrive/val2017.zip -d /content//OW-DETR/data/coco
!mv /content//OW-DETR/data/coco/val2017/*.jpg /content/OW-DETR/data/OWDETR/VOC2007/JPEGImages/.
```
### OWOD proposed splits
<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

The splits are present inside `data/VOC2007/OWOD/ImageSets/` folder. The remaining dataset can be downloaded using this [link](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k?usp=sharing)

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">4.9</td>
        <td align="center">56.0</td>
        <td align="center">2.9</td>
        <td align="center">39.4</td>
        <td align="center">3.9</td>
        <td align="center">29.7</td>
        <td align="center">25.3</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">7.5</td>
        <td align="center">59.2</td>
        <td align="center">6.2</td>
        <td align="center">42.9</td>
        <td align="center">5.7</td>
        <td align="center">30.8</td>
        <td align="center">27.8</td>
    </tr>
</table>



### Our proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

#### Dataset Preparation

The splits are present inside `data/VOC2007/OWDETR/ImageSets/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/VOC2007/OWDETR/JPEGImages/
mkdir data/VOC2007/OWDETR/Annotations/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
OW-DETR/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
```
cd OW-DETR/data
mv data/coco/train2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
```
5. Use the code `coco2voc.py` for converting json annotations to xml files.

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWDETR/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```


Currently, Dataloader and Evaluator followed for OW-DETR is in VOC format.

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">1.5</td>
        <td align="center">61.4</td>
        <td align="center">3.9</td>
        <td align="center">40.6</td>
        <td align="center">3.6</td>
        <td align="center">33.7</td>
        <td align="center">31.8</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">5.7</td>
        <td align="center">71.5</td>
        <td align="center">6.2</td>
        <td align="center">43.8</td>
        <td align="center">6.9</td>
        <td align="center">38.5</td>
        <td align="center">33.1</td>
    </tr>
</table>

    
# Training

#### Training on single node

To train OW-DETR on a single node with 8 GPUS, run
```bash
./run.sh
```

#### Training on slurm cluster

To train OW-DETR on a slurm cluster having 2 nodes with 8 GPUS each, run
```bash
sbatch run_slurm.sh
```

# Evaluation

For reproducing any of the above mentioned results please run the `run_eval.sh` file and add pretrained weights accordingly.


**Note:**
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.
