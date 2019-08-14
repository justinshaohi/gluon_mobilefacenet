import math
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

class L2Softmax(SoftmaxCrossEntropyLoss):

    def __init__(self, classes, alpha, p=0.9, from_normx=False,
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super(L2Softmax,self).__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, "For given probability of p={}, alpha should higher than {}.".format(p, alpha_low)
        self.alpha = alpha
        self._from_normx = from_normx

    def hybrid_forward(self, F, x, label, sample_weight=None):
        if not self._from_normx:
            x = F.L2Normalization(x, mode='instance', name='fc1n')
        fc7 = x * self.alpha
        return super(L2Softmax,self).hybrid_forward(F, pred=fc7, label=label, sample_weight=sample_weight)

class ArcLoss(SoftmaxCrossEntropyLoss):

    def __init__(self, classes, m=0.5, s=64, easy_margin=True, dtype="float32", **kwargs):
        super(ArcLoss,self).__init__(**kwargs)
        assert s > 0.
        assert 0 <= m < (math.pi / 2)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self._classes = classes
        self.easy_margin = easy_margin
        self._dtype = dtype

    def hybrid_forward(self, F, pred, label, sample_weight=None, *args, **kwargs):
        cos_t = F.pick(pred, label, axis=1)
        if self.easy_margin:
            cond = F.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - self.threshold
            cond = F.Activation(data=cond_v, act_type='relu')


        new_zy = F.cos(F.arccos(cos_t) + self.m)
        if self.easy_margin:
            zy_keep = cos_t
        else:
            zy_keep = cos_t - self.mm
        new_zy = F.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t
        diff = F.expand_dims(diff, 1)
        gt_one_hot = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0, dtype=self._dtype)
        body = F.broadcast_mul(gt_one_hot, diff)
        pred = pred + body
        pred = pred * self.s

        return super(ArcLoss,self).hybrid_forward(F, pred=pred, label=label, sample_weight=sample_weight)