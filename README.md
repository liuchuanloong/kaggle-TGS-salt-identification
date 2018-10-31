# Kaggle - TGS Salt Identification challenge
[open source 60th solution(code)](https://github.com/liuchuanloong/kaggle-TGS-salt-identification) for kaggle TGS salt identification challenge
this is my first kaggle competition, I worked hard in last month competition times, thanks for all people shared their ideas, in especial [Jack (Jiaxin) Shao](https://www.kaggle.com/shaojiaxin), [peter](https://www.kaggle.com/pestipeti) and [heng](https://www.kaggle.com/hengck23), they give me a lot of encouragement, I have gained so much from this competition, and 
thanks for [Eduardo Rocha de Andrade's code for our baseline model ](https://github.com/arc144/Kaggle-TGS-Salt-Identification)
## our final solution

>Input: 101 -> resize to 202 -> pad to 256 \
>Encoder: ResNet34 pretrained on ImageNet \
>Centerblock: FPA model for attention \
>Decoder: conv3x3 + GAU 

## Training overview: 
- Optimizer: Lovasz Loss, SGD, Batch size: 32, Reduce LR on cosine annealing, 100 epochs each, LR starting from 0.01  
1. single model ResNet34 got 0.856 public LB (0.878 private LB) 
2. 5-fold Average Ensemble ResNet34 got 0.859 public LB (0.881 private LB) 
3. all threshold we used 0.45 for best score 
4. Transfer above best model for 5-fold Average Ensemble ResNet34 got 0.864 public LB (0.883 private LB) 

## Augmentations
>we used [heng's](https://www.kaggle.com/hengck23/competitions) augmentations [code](https://drive.google.com/drive/folders/18_gAnL1GMD7Ogyz4T3Y0l_UD31qagsc-?usp=sharing), we first used on keras model but did not work , in pytorch, it worked perfect

>>do_horizontal_flip2 \
>>do_random_shift_scale_crop_pad2  0.2  \
>>do_horizontal_shear2             (-0.07, +0.07) \
>>do_shift_scale_rotate2 rotate    15 \
>>do_elastic_transform2            (0, 0.15) \
>>do_brightness_shift              (-0.1, +0.1) \
>>do_brightness_multiply           (-0.08, +0.08) \
>>do_gamma                         (-0.08, +0.08) 

## preprocessing
> [depth channels](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949#385778) \
> Stratified in this [kernel](https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss) rather than simply cov_to_class, this give me 0.03 boost

## implenment
>>pytorch >= 0.4.1 \
>>python = 3.6

## Nets
>### U-net
>>#### Encoder Architecture
>>[x] Implement ResNet34 encoded U-Net\
>>[x] Use pretrain weights\
>>[x] Compare pretrained encoder with full initialized\
>>[x] Implement deeper encoders (ResNet50,101,152)
>>#### Decoder Architecture
>>[x] Compare transposed convolutions with bilinear upsampling\
>>[x] No pooling in center block: 4 upsamplings\
>>[x] Add pooling in center block: 5 upsamplings\
>>[x] Try spatial dropout\
>>[x] Try spatial pyramid pooling\
>>[x] Replace sesc decode as [GAU](https://arxiv.org/abs/1805.10180)\
>>[ ] Replace part of the decoder with dilated convs
>>#### [Hyper columns](https://arxiv.org/pdf/1411.5752.pdf)
>>[x] Implement hyper columns on decoder\
>>[ ] Increment hyper columns using adptation convolutions
>>#### [Squeeze and Excitation Block](https://arxiv.org/pdf/1803.02579.pdf)
>>[x] Implement sSE and cSC on decoder\
>>[x] Implement sSE and cSC on encoder\
>>[x] Use cSE block as input to image depth
>>#### [Pyramid Pooling Module](https://arxiv.org/pdf/1612.01105.pdf)
>>[x] Implement PPM on U-Net's center block\
>>[ ] Implement [OC-ASP Module](https://arxiv.org/pdf/1809.00916.pdf)
>>#### [Feature Pyramid Attention](https://arxiv.org/abs/1805.10180)
>>[x] Implement FPA on Unet center block\

>### FPNet
>[x] Implement FPNet 
>>[ ] Add SE Blocks

>### [RefineNet](https://arxiv.org/pdf/1611.06612.pdf) 
>[x] Implement RefineNet\
[x] Add SE blocks\
[ ] Implement dense connections in RefineBlocks\
[ ] Use PPM or ASSP instead of ChainPoolingModule\
[ ] Use hypercolumns

>### [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf) 
>[ ] Implement DeepLabv3\
[ ] Add SE blocks\
[ ] Implement [OC-ASP Module](https://arxiv.org/pdf/1809.00916.pdf)

## Training Procedure
>### Learning Rate Scheduler
>[x] Implement learnig rate reduction on Plateau\
[x] Implement learnig rate reduction on Milestones
>>##### [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf)
>>[x] Implement cosine annealing with warm restart\
[x] Verify if getting predictions for the M last restarts can produce a good ensemble
>### Loss Function
>[x] Implement Binary Cross Entropy\
[x] Implement Sørensen–Dice loss (IoU)\
[x] Implement Hybrid BCE+Dice\
[x] Implement [Lovász-hinge loss](https://arxiv.org/pdf/1705.08790.pdf)\
[x] Implement Hybrid BCE+Lovás loss\
[ ] Add Regional Loss: [Adaptive Affinity Field](https://arxiv.org/pdf/1803.10335.pdf)
>### Cross-validation
> [x] Implement 5Fold cross-validation\
[x] Implement stratified cross-validation by mask coverage\
[x] Ensemble predictions (average) on 5 folds
>### Data Augmentation
>[x] Implement basic augmentations (flips, shift, scale, crop, elastic, rotate, shear, gamma, brightness)\
[x] Use additional dataset made of masks with small salt coverage\
[ ] Test other types of augmentations
>### Test Time Augmentation
>[x] Implement simple flip TTA\
[ ] Implement other types of TTA

