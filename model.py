import mxnet as mx


def make_conv(stage_idx,channels=1,kernel=1,stride=1, pad=0,num_group=1, active=True):
    out=mx.gluon.nn.HybridSequential(prefix='stage{}'.format(stage_idx))
    out.add(mx.gluon.nn.Conv2D(channels,kernel,stride,pad,groups=num_group,use_bias=False))
    out.add(mx.gluon.nn.BatchNorm(scale=True))
    if active:
        out.add(mx.gluon.nn.PReLU())
    return out

def make_bottleneck(stage_idx,layers, channels, stride, t, in_channels=0):
    layer = mx.gluon.nn.HybridSequential(prefix='stage{}'.format(stage_idx))
    with layer.name_scope():
        layer.add(Bottleneck(in_channels=in_channels, channels=channels, t=t, stride=stride))
        for _ in range(layers - 1):
            layer.add(Bottleneck(channels, channels, t, 1))
    return layer


class Bottleneck(mx.gluon.nn.HybridBlock):

    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = mx.gluon.nn.HybridSequential()
            self.out.add(make_conv(0, in_channels * t),
                         make_conv(1, in_channels * t, kernel=3, stride=stride,
                                    pad=1, num_group=in_channels * t),
                         make_conv(2, channels, active=False))

    def hybrid_forward(self, F, x, **kwargs):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class NormDense(mx.gluon.nn.HybridBlock):
    def __init__(self, classes, weight_norm=False, feature_norm=False,
                 dtype='float32', weight_initializer=None, in_units=0, **kwargs):
        super(NormDense,self).__init__(**kwargs)
        self._weight_norm = weight_norm
        self._feature_norm = feature_norm

        self._classes = classes
        self._in_units = in_units
        if weight_norm:
            assert in_units > 0, "Weight shape cannot be inferred auto when use weight norm, " \
                                 "in_units should be given."
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(classes, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight, *args, **kwargs):
        if self._weight_norm:
            weight = F.L2Normalization(weight, mode='instance')
        if self._feature_norm:
            x = F.L2Normalization(x, mode='instance', name='fc1n')
        return F.FullyConnected(data=x, weight=weight, no_bias=True,
                                num_hidden=self._classes, name='fc7')

    def __repr__(self):
        s = '{name}({layout})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class MobileFacenet(mx.gluon.nn.HybridBlock):

    def __init__(self,classes=-1,embedding_size=128,weight_norm=True, feature_norm=True,**kwargs):
        super(MobileFacenet, self).__init__(**kwargs)
        self._classes=classes
        self._fn=feature_norm
        with self.name_scope():
            self.features = mx.gluon.nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                self.features.add(make_conv(0, 64, kernel=3, stride=2, pad=1),
                                  make_conv(0, 64, kernel=3, stride=1, pad=1, num_group=64))

                self.features.add(
                    make_bottleneck(1, layers=5, channels=64, stride=2, t=2, in_channels=64),
                    make_bottleneck(2, layers=1, channels=128, stride=2, t=4, in_channels=64),
                    make_bottleneck(3, layers=6, channels=128, stride=1, t=2, in_channels=128),
                    make_bottleneck(4, layers=1, channels=128, stride=2, t=4, in_channels=128),
                    make_bottleneck(5, layers=2, channels=128, stride=1, t=2, in_channels=128))

                self.features.add(make_conv(6, 512),
                                  make_conv(6, 512, kernel=7, num_group=512, active=False),
                                  mx.gluon.nn.Conv2D(128, 1, use_bias=False),
                                  mx.gluon.nn.BatchNorm(scale=False, center=False),
                                  mx.gluon.nn.Flatten())
                if classes>0:
                    self.output=NormDense(classes, weight_norm, feature_norm,
                                    in_units=embedding_size, prefix='output_')

    def hybrid_forward(self, F, x, *args, **kwargs):
            embedding = self.features(x)
            if self._classes>0:
                out=self.output(embedding)
                return out
            else:
                if self._fn:
                    embedding = F.L2Normalization(embedding, mode='instance')
                    return embedding
                return embedding