# DSNet
a novel real-time model in semantic segmentation
This is the implementation for DSNet. 

## **environment**: 
PyTroch 1.10 

python 3.8

4\*RTX4090 or 8\*RTX4090 
      
      pip install -r requirements.txt

## **Train and Inference speed**:
This implementation is based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation). Please refer to their repository for installation and dataset preparation.The inference speed is tested on single RTX 3090 or RTX4090.
     
      

### Train
      
      python -m torch.distributed.launch --nproc_per_node=4 DSNet/tools/train.py

### Inference speed

      python DSNet/models/speed/dsnet_speed.py

## Weight
### DSNet-Base:
  
  DSNet_Base_imagenet：[ Baidu drive](https://pan.baidu.com/s/1acGfjtF1eHb3hNxyHcsJTA?pwd=a123) ,[google drive](https://drive.google.com/file/d/1LqmgL4thNJFcMWRYaXJUFNTy2y5FvZ8E/view?usp=sharing)
  
  ADE20K: 43.44%mIOU: [ Baidu drive](https://pan.baidu.com/s/1TKBFtCj6gwMq97NjYsmPOQ?pwd=a123), [google drive](https://drive.google.com/file/d/1hr9BlqgI4t4djibyj1fCW2LTMvlFFDWP/view?usp=sharing)
  
  BDD10K: 64.6%mIOU： [ Baidu drive](https://pan.baidu.com/s/13Hvi6he0hZgciff7tBUo0A?pwd=a123), [google drive]( https://drive.google.com/file/d/1IqMornjPHMVYHWdGhl-Jr1J4FvcZotoj/view?usp=sharing)
                      
  Camvid: 83.32%mIOU: [ Baidu drive](https://pan.baidu.com/s/1Q-e-_s-vsgn14S8GoBoTlA?pwd=a123), [google drive](https://drive.google.com/file/d/141889Jei9rcgJ9wSiFF8rNvUDUiV8SqI/view?usp=sharing)           

### DSNet:

  DSNet_imagenet: [ Baidu drive](https://pan.baidu.com/s/1wMPH5ZNKwHIyFJ6Pp9-n2w?pwd=a123), [google drive](https://drive.google.com/file/d/1Cb3nd69IjQjjK_r8jXSMON4cHQ76MWbR/view?usp=sharing)
  
  ADE20k 40.0%mIOU: [ Baidu drive](https://pan.baidu.com/s/17CH66GTI2YEXMq7eXnK0xQ?pwd=a123), [google drive](https://drive.google.com/file/d/1J-qf5blQ71HGy4EStqMMg-NT1sO1CyUV/view?usp=sharing)
  
  BDD10K 62.8%mIOU: [ Baidu drive](https://pan.baidu.com/s/1tPQHC1LTE6tlueXvabU1-Q?pwd=a123), [google drive](https://drive.google.com/file/d/192T2dauq_cA1bBkmiRwYKWdYw27lxZIG/view?usp=sharing)

