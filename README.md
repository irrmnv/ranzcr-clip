# RANZCR CLiP - Catheter and Line Position Challenge

Solution for [Kaggle RANZCR CLiP Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification) that earned me silver medal (18th place on the public lb with 0.971 AUC and 45th on the private 0.972 AUC).

## Reference
* [Flexible library for training NNs in pytorch](https://github.com/lRomul/argus)
* [3-stage training notebook of @yasufuminakama](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step1)
* [rwightman's repo with SOTA models for PyTorch](https://github.com/rwightman/pytorch-image-models)


## For inference
* [Pretrained weights (ResNet200D, EfficientNetB7)](https://www.kaggle.com/irrmnv/ranzcr-clip-models)
* [Inference kernel](https://www.kaggle.com/irrmnv/ranzcr-clip-inference)


## Solution
* 3 stage training:
    * teacher with annotated data (BCE)
    * student-teacher (BCE and MSE on outputs of teacher's features layer)
    * student finetune without annotated data
* Blend of two models:
    * ResNet200D on 640 crop
    * EfficientNetB7 on 1000 crop
* For me, key point is a precise transferring of teacher's features so:
    * try not to overfit teacher (0.5 dropout)
    * try to use old-fashion sequential freezing of last layers before start training without annotated images (its just my intuition, I didnt test it)
* I started the competition 10 days before the end. It took:
    * around 5-6 days for training ResNet200D (5 fold) on 2x1080Ti and getting some intuition about the data
    * 2-3 days for scaling pipeline for EfficientNetB7 on 8x3090
    * 2 days for fighting my intuition about the usefulness of pseudolabeling. In my finale pipeline there was no pseudolabels, so I didnt't have a chance to compete for gold medal
