# QCNet training code on ETRI Trajectory Dataset (ETD)

We slightly modified the original source code (https://github.com/ZikangZhou/QCNet) to make it compatible with our ETD. The deep learning environment can be easily created following the instruction (https://github.com/d1024choi/QCNetDocker). In this guide, we would like to walk you through how to train and test QCNet on ETD. If you need more details about ETD, visit the corresponding repository (https://github.com/d1024choi/ETRITrajPredChallenage). 

## 1.Convert Argoverse2-type dataset into QCNet-compatible one

Initially, you are given two sets, one for training ('train' folder) and the other for test ('test_masked' folder). Convert scene sample files in the folders to QCNet-compatible ones using 'argoverse2_to_qcnet_format.py' in https://github.com/d1024choi/ETRITrajPredChallenage. 

## 2.Split Dataset

To train and validate QCNet, we split samples in 'train' folder into two groups, one for training and the other for validation, according to 'train_list.txt' and 'val_list.txt'. To calculate the average FLOPs for QCNet, we copy test samples in 'test_masked' folder according to 'test_flops_list.txt'. Eventually, after going through all the processes, we have the following four folders:

* 'train_qcnet' : converted samples from 'train' folder to train QCNet
* 'val_qcnet' : converted samples from 'train' folder to validate QCNet
* 'test_qcnet' : converted samples from 'test_masked' to test QCNet 
* 'test_flops_qcnet' : converted samples from 'test_masked' to calculate FLOPs of QCNet 


## 3.Train
 
Run the following command:
```
python train.py
```

Before running the training, you must properly set dataset location parameters like

```
parser.add_argument('--root', type=str, default="/workspace/av2format")
parser.add_argument('--train_processed_dir', type=str, default="train_qcnet")
parser.add_argument('--val_processed_dir', type=str, default="val_qcnet")
parser.add_argument('--test_processed_dir', type=str, default="test_qcnet")
```
We stored our data samples with the following data folder structure
```
av2format --- train_qcnet
            |
            - val_qcnet
            |
            - test_qcnet
            |
            - test_flops_qcnet
```

You can control parameters for the training by editing train.py. For example, by setting --device to 8, you can use 8 GPUs for multi-gpu training. Once you start training, a folder by the name of 'version_$number$' is automatically created and the intermediate trained parameters are stored in the folder. 


## 4.Test
 
Run the following command:
```
python test.py
```
When testing your trained model, please set --batch_size to 1 to see the average time spent per scene. Once you start testing, a folder by the name of 'version_$number$/test_results' is automatically created and the prediction results are stored in the folder. You can submit the prediction results for the competition leaderboard.

## 5.Visualization
 
You can visualize your predictions by running
```
python visualization.py
```
## 6. Calculate FLOPs

You can calculate FLOPs of QCNet by running

```
python calculate_FLOPs.py
```

The following parameters must be properly set like

```
    parser.add_argument('--root', type=str, default="/home/dooseop/DATASET/ETRI/av2format")
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--train_processed_dir', type=str, default="train_qcnet")
    parser.add_argument('--val_processed_dir', type=str, default="val_qcnet")
    parser.add_argument('--test_processed_dir', type=str, default="test_flops_qcnet")
```
