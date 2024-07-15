<h1 align='center'>MambaVision</h1>

This is a warehouse for MambaVision-Pytorch-model, can be used to train your image-datasets for vision tasks.

### [Official Source Code](https://github.com/NVlabs/MambaVision)  
### [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/pdf/2407.08083)  
### [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)   

![image](https://github.com/jiaowoguanren0615/MambaVision/blob/main/sample_png/1720946952217.jpg)  
![image](https://github.com/jiaowoguanren0615/MambaVision/blob/main/sample_png/1720947030019.jpg)  


## Preparation
### Install mamba_ssm & causal_conv1d
[Create Environment Tutorial](https://github.com/jiaowoguanren0615/Install_Mamba)  

### Download the dataset(For image classification): 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)  

### Download the dataset(For image segmentation): 
[Segmentation Datasets(including VOC, ADE20K, COCO, COCOStuff, Hubmap, Synapse, CityScapes)](https://pan.baidu.com/s/1LLyIlP3sjuoFAwTBaYflRQ?pwd=0615)  

### Image Processing For Segmentation
[Processing Dataset Tutorial](https://github.com/jiaowoguanren0615/Segmentation_Factory/tree/main/datasets)  

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: MambaVision Model
    ├── build_cls_models.py: Construct MambaVision models for image classification
    ├── build_seg_models.py: Construct MambaVision models for image segmentation
    ├── helpers.py: Compute scaled dot product attention
    ├── seg_model_backbones.py: Construct MambaVision as segmentation-model backbones
    ├── segformer_head.py: Construct Segformer Head
    ├── upernet_head.py: Construct UperNet Head
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── lr_decay.py: Define "inverse_sqrt_lr_decay" function for "Adafactor" optimizer
    ├── lr_sched.py: Define "adjust_learning_rate" function
    ├── optimizer.py: Define Sophia & Adafactor & LAMB optimizer(for mambavision models training)
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___,  ___data_len___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  
Moreover, you can set the ___opt_auc___ parameter to True if you want to optimize your model for a better performance(maybe~).


## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.finetune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```


## Citation
```
@inproceedings{Hatamizadeh2024MambaVisionAH,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Ali Hatamizadeh and Jan Kautz},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:271097462}
}
```

```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

```
@article{you2019large,
  title={Large batch optimization for deep learning: Training bert in 76 minutes},
  author={You, Yang and Li, Jing and Reddi, Sashank and Hseu, Jonathan and Kumar, Sanjiv and Bhojanapalli, Srinadh and Song, Xiaodan and Demmel, James and Keutzer, Kurt and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:1904.00962},
  year={2019}
}
```

```
@inproceedings{xiao2018unified,
  title={Unified perceptual parsing for scene understanding},
  author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={418--434},
  year={2018}
}
```

```
@inproceedings{he2017mask,
  title={Mask r-cnn},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2961--2969},
  year={2017}
}
```