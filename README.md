# Kaggle - TGS Salt Identification challenge
## Nets
>###U-net
>>#### Encoder Architecture
>>[x] Implement ResNet34 encoded U-Net\
[x] Use pretrain weights\
[x] Compare pretrained encoder with full initialized\
[X] Implement deeper encoders (ResNet50,101,152)
>>#### Decoder Architecture
>>[x] Compare transposed convolutions with bilinear upsampling\
[x] No pooling in center block: 4 upsamplings\
[x] Add pooling in center block: 5 upsamplings\
[x] Try spatial dropout\
[ ] Replace part of the decoder with dilated convs
>>#### [Hyper columns](https://arxiv.org/pdf/1411.5752.pdf)
>>[x] Implement hyper columns on decoder\
[ ] Increment hyper columns using adptation convolutions
>>#### [Squeeze and Excitation Block](https://arxiv.org/pdf/1803.02579.pdf)
>>[x] Implement sSE and cSC on decoder\
[x] Implement sSE and cSC on encoder\
[x] Use cSE block as input to image depth
>>#### [Pyramid Pooling Module](https://arxiv.org/pdf/1612.01105.pdf)
>>[x] Implement PPM on U-Net's center block\
>>[ ] Implement [OC-ASP Module](https://arxiv.org/pdf/1809.00916.pdf)


>###FPNet
>[x] Implement FPNet\
[ ] Add SE Blocks

>###[RefineNet](https://arxiv.org/pdf/1611.06612.pdf)
>[x] Implement RefineNet\
[x] Add SE blocks\
[ ] Implement dense connections in RefineBlocks\
[ ] Use PPM or ASSP instead of ChainPoolingModule\
[ ] Use hypercolumns

>###[DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf)
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

