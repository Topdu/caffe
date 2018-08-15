这个TextBoxes分支没有修改原版代码，仅仅只是将代码移植到ssd-windows版本上面，做出来的一个caffe-windows-TextBoxes 仅仅是为了在windows上学习TextBoxes
首先看一下ssd和TextBoxes原版代码

# SSD: Single Shot MultiBox Detector

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](http://research.google.com/pubs/DragomirAnguelov.html), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), Cheng-Yang Fu, [Alexander C. Berg](http://acberg.com).


### Introduction

SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325).

<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">
</p>

<center>

| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes |
|:-------|:-----:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | 300 |
| [Faster R-CNN (ZF)](https://github.com/ShaoqingRen/faster_rcnn) | 62.1 | 17 | 300 |
| [YOLO](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 |
| [Fast YOLO](http://pjreddie.com/darknet/yolo/) | 52.7 | 155 | 98 |
| SSD300 (VGG16) | 72.1 | 58 | 7308 |
| SSD300 (VGG16, cuDNN v5) | 72.1 | 72 | 7308 |
| SSD500 (VGG16) | **75.1** | 23 | 20097 |

</center>

### Citing SSD

Please cite SSD in your publications if it helps your research:

    @article{liu15ssd,
      Title = {{SSD}: Single Shot MultiBox Detector},
      Author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      Journal = {arXiv preprint arXiv:1512.02325},
      Year = {2015}
    }
    
## Windows Setup

This branch of Caffe extends [BVLC-led Caffe](https://github.com/BVLC/caffe) by adding Windows support and other functionalities commonly used by Microsoft's researchers, such as managed-code wrapper, [Faster-RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf), [R-FCN](https://arxiv.org/pdf/1605.06409v2.pdf), etc.

**Contact**: Kenneth Tran (ktran@microsoft.com)

---

# Caffe

|  **`Linux (CPU)`**   |  **`Windows (CPU)`** |
|-------------------|----------------------|
| [![Travis Build Status](https://api.travis-ci.org/Microsoft/caffe.svg?branch=master)](https://travis-ci.org/Microsoft/caffe) | [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/58wvckt0rcqtwnr5/branch/master?svg=true)](https://ci.appveyor.com/project/pavlejosipovic/caffe-3a30a) |              

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

**Requirements**: Visual Studio 2013

### Pre-Build Steps
Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`

By defaults Windows build requires `CUDA` and `cuDNN` libraries.
Both can be disabled by adjusting build variables in `.\windows\CommonSettings.props`.
Python support is disabled by default, but can be enabled via `.\windows\CommonSettings.props` as well.
3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).
If you don't have CUDA installed, you can experiment with CPU_ONLY build.
In `.\windows\CommonSettings.props` set `CpuOnlyBuild` to `true` and set `UseCuDNN` to `false`.

### cuDNN
Download `cuDNN v4` or `cuDNN v5` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to %CUDA_PATH% (environment variable set by CUDA installer).
Alternatively, you can unpack zip to any location and set `CuDnnPath` to point to this location in `.\windows\CommonSettings.props`.
`CuDnnPath` defined in `.\windows\CommonSettings.props`.
Also, you can disable cuDNN by setting `UseCuDNN` to `false` in the property file.

### Python
To build Caffe Python wrapper set `PythonSupport` to `true` in `.\windows\CommonSettings.props`.
Download Miniconda 2.7 64-bit Windows installer [from Miniconda website] (http://conda.pydata.org/miniconda.html).
Install for all users and add Python to PATH (through installer).

Run the following commands from elevated command prompt:

```
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf
```

#### Remark
After you have built solution with Python support, in order to use it you have to either:  
* set `PythonPath` environment variable to point to `<caffe_root>\Build\x64\Release\pycaffe`, or
* copy folder `<caffe_root>\Build\x64\Release\pycaffe\caffe` under `<python_root>\lib\site-packages`.

### Matlab
To build Caffe Matlab wrapper set `MatlabSupport` to `true` and `MatlabDir` to the root of your Matlab installation in `.\windows\CommonSettings.props`.

#### Remark
After you have built solution with Matlab support, in order to use it you have to:
* add the generated `matcaffe` folder to Matlab search path, and
* add `<caffe_root>\Build\x64\Release` to your system path.

### Build
Now, you should be able to build `.\windows\Caffe.sln`

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `%CAFFE_ROOT%`
  ```Shell
  git clone https://github.com/conner99/caffe.git
  cd caffe
  git checkout ssd-microsoft
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `%CAFFE_ROOT%\models\VGGNet\`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `%CAFFE_ROOT%\data\VOC0712`
  
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```Explorer
# Extract the data manual.
%CAFFE_ROOT%\data\VOC0712\
		VOC2007\Annotations
		VOC2007\JPEGImages
		VOC2012\Annotations
		VOC2012\JPEGImages

  ```

3. Create the LMDB file.
  ```Shell
  cd %CAFFE_ROOT%
  # Create test_name_size.txt in data\VOC0712\
  .\data\VOC0712\get_image_size.bat
  # You can modify the parameters in create_data.bat if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - %CAFFE_ROOT%\data\VOC0712\trainval_lmdb
  #   - %CAFFE_ROOT%\data\VOC0712\test_lmdb
  .\data\VOC0712\create_data.bat
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - %CAFFE_ROOT%\models\VGGNet\VOC0712\SSD_300x300\
  # and job file, log file, and the python script in:
  #   - %CAFFE_ROOT%\jobs\VGGNet\VOC0712\SSD_300x300\
  # and save temporary evaluation results in:
  #   - %CAFFE_ROOT%\data\VOC2007\results\SSD_300x300\
  # It should reach 72.* mAP at 60k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples\ssd\score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples\ssd\ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out `examples/ssd_detect.ipynb` or `examples/ssd/ssd_detect.cpp` on how to detect objects using a SSD model.

5. To train on other dataset, please refer to data/OTHERDATASET for more details.
We currently add support for MSCOCO and ILSVRC2016.

### Models
1. Models trained on VOC0712: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_500x500.tar.gz)

2. Models trained on MSCOCO trainval35k: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_500x500.tar.gz)

3. Models trained on ILSVRC2015 trainval1: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_500x500.tar.gz) (46.4 mAP on val2)



# TextBoxes: A Fast Text Detector with a Single Deep Neural Network

Recommend: [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus) is an extended work of TextBoxes, which supports oriented scene text detection. The recognition part is also included in [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus).

### Introduction
This paper presents an end-to-end trainable fast scene text detector, named TextBoxes, which detects scene text with both high accuracy and efficiency in a single network forward pass, involving no post-process except for a standard nonmaximum suppression. For more details, please refer to our [paper](https://arxiv.org/abs/1611.06779).

### Citing TextBoxes
Please cite TextBoxes in your publications if it helps your research:

    @inproceedings{LiaoSBWL17,
      author    = {Minghui Liao and
                   Baoguang Shi and
                   Xiang Bai and
                   Xinggang Wang and
                   Wenyu Liu},
      title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
      booktitle = {AAAI},
      year      = {2017}
    }


### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Test](#test)
4. [Train](#train)
5. [Performance](#performance)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/MhLiao/TextBoxes.git
  
  cd TextBoxes
  
  make -j8
  
  make py
  ```

### Download
1. Models trained on ICDAR 2013: [Dropbox link](https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1qY73XHq)
2. Fully convolutional reduced (atrous) VGGNet: [Dropbox link](https://www.dropbox.com/s/qxc64az0a21vodt/VGG_ILSVRC_16_layers_fc_reduced.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1slQyMiL)
3. Compiled mex file for evaluation(for multi-scale test evaluation: evaluation_nms.m): [Dropbox link](https://www.dropbox.com/s/xtjuwvphxnz1nl8/polygon_intersect.mexa64?dl=0) [BaiduYun link](http://pan.baidu.com/s/1jIe9UWA)


### Test
1. run "python examples/demo.py".
2. You can modify the "use_multi_scale" in the "examples/demo.py" script to control whether to use multi-scale or not.
3. The results are saved in the "examples/results/".


### Train
1. Train about 50k iterions on Synthetic data which refered in the paper.
2. Train about 2k iterions on corresponding training data such as ICDAR 2013 and SVT.
3. For more information, such as learning rate setting, please refer to the paper.

### Performance
1. Using the given test code, you can achieve an F-measure of about 80% on ICDAR 2013 with a single scale.
2. Using the given multi-scale test code, you can achieve an F-measure of about 85% on ICDAR 2013 with a non-maximum suppression.
3. More performance information, please refer to the paper and Task1 and Task4 of Challenge2 on the ICDAR 2015 website: http://rrc.cvc.uab.es/?ch=2&com=evaluation

### Data preparation for training
The reference xml file is as following:
  
        <?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>158</xmin>
                    <ymin>128</ymin>
                    <xmax>411</xmax>
                    <ymax>181</ymax>
                </bndbox>
            </object>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>443</xmin>
                    <ymin>128</ymin>
                    <xmax>501</xmax>
                    <ymax>169</ymax>
                </bndbox>
            </object>
            <folder></folder>
            <filename>100.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
                <depth>3</depth>
            </size>
        </annotation>

Please let me know if you encounter any issues.


