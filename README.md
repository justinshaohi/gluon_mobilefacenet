# gluon_mobilefacenet
using mxnet gluon api build mobilefacenet model to training face data

# prepare data
get train data from [dataset_zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

# train
```shell
python train.py --args
```
# verification
verification in lfw,and accuary is 98.6%
```shell
python verification.py
```
