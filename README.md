# OW-DETR: Open-world Detection Transformer (CVPR 2022)





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
权重文件

Download the self-supervised backbone from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
(这make脚本里就是执行setup)
!python /content/drive/MyDrive/OW-DETR/models/ops/setup.py build install

这一步可以解决ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'这个报错。
应该是指定路径下载多头注意力模块

# unit test (should see all checking is True)
# 不知道他在test什么，好像是测试和说明特定函数在不同精度和通道数下的前向计算和梯度计算的准确性。
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

"data/VOC2007/OWOD/ImageSets/" 文件夹中存在分割数据集的文件。这些分割通常指的是将一个数据集按照不同的用途划分为训练集、验证集和测试集等子集。

可以使用特定链接下载其余的数据集

The splits are present inside `data/VOC2007/OWOD/ImageSets/` folder. The remaining dataset can be downloaded using this [link](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k?usp=sharing)

The files should be organized in the following structure:
<<<<<<< Updated upstream
扯淡
=======

>>>>>>> Stashed changes
```
原作有误已修改
OW-DETR/
└── data/
    └── OWOD/
        └── VOC2007/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```

如果只是想看一下跑通的效果，可以直接下载作者给的OWOD的图片和标注，放在OWDETR中，也是可以跑通的。

JEPGImages\
Annotations\
解压到/content目录下

```
!unzip /content/drive/MyDrive/Annotations -d /content/OW-DETR/data/OWDETR/VOC2007
!unzip /content/drive/MyDrive/JEPGImages -d /content/OW-DETR/data/OWDETR/VOC2007
```
然后在ImageSets文件夹下新建一个Main文件夹，把本来ImageSets下的txt文件都移动到Main文件夹下，准备工作就完成了，之后就可以开始训练了。


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



### OWDETR splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

#### Dataset Preparation

划分文件在VOC2007 JPEGIMAGES里

The splits are present inside `data/OWDETR/VOC2007/JPEGImages/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/OWDETR/VOC2007/JPEGImages/
mkdir data/OWDETR/VOC2007/Annotations/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. 分别下载coco数据集的训练集和验证集，然后将他们移动到JEPGImages文件夹下。
```
建议是直接下载到JPEG但是空间不一定够
#下载coco数据集
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/zips/val2017.zip
```

3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
OW-DETR/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```

4. Move all images from `train2017/` and `val2017/` to `/VOC2007/OWDETR/JPEGImages` folder.
质疑，为什么不直接解压到/VOC2007/OWDETR/JPEGImages

反正就是把train2017、val2017里面所有的.jpg都移动到data/VOC2007/OWDETR/JPEGImages/里面
```
cd OW-DETR/data
mv data/coco/train2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
```
5. Use the code `coco2voc.py` for converting json annotations to xml files.

使用代码 coco2voc.py 将 json 注释转换为 xml 文件。

先下载coco的annotation文件，然后解压

注意下载到哪

```commandline
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip /content/drive/MyDrive/annotations_trainval2017.zip -d /content/OW-DETR/data/coco/
```
然后coco2voc.py文件里写的是解压到data/OWDETR/VOC2007/Annotations（我改过，不一定对），小叶写的是
    target_folder = '/content/drive/MyDrive/OW-DETR/data/OWDETR/VOC2007' 但是我感觉annotations文件夹就没用了

The files should be organized in the following structure:
```
原作有误已修改
OW-DETR/
└── data/
    └── OWDETR/
        └── VOC2007/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```
还有一个问题，源码中使用了train.txt，但是作者没有给所有的train.txt，而是给了t1_train,t2_train,t3_train,t4_train
</b>小叶在努力搞科研是把这段代码复制了四遍，然后把train.txt换成t1_train到t4_train，这样子最简单，并且要在ImageSets文件夹下新建一个Main文件夹，把本来ImageSets下的txt文件都移动到Main文件夹下，要不然下面训练也会报错。

这里我先试一下把那四个txt合并为一个train.txt

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
