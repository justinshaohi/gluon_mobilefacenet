import os
import mxnet as mx
from data import ValDataset
from metric import FaceVerification
from sklearn import preprocessing

def verification(nfolds=10, norm=True):
    ctx = mx.gpu()
    batch_size = 64
    transform_test = mx.gluon.data.vision.transforms.Compose([
        mx.gluon.data.vision.transforms.ToTensor()
    ])
    val_dataset = ValDataset('lfw', './data',transform=transform_test)
    val_data = mx.gluon.data.DataLoader(val_dataset, batch_size=batch_size, last_batch='keep', num_workers=8)
    sym_file = os.path.join('./out', 'mobilefacenet-symbol.json')
    params_file = os.path.join('./out', 'mobilefacenet-0006.params')
    sym = mx.sym.load(sym_file)
    internals = sym.get_internals()
    sym_out = internals.list_outputs()
    outputs = []
    for out in sym_out:
        outputs.append(internals[out])
        if 'mobilefacenet0_features_flatten0_flatten0_output' == out:
            break
    net = mx.gluon.SymbolBlock(outputs, mx.sym.var('data'))
    net.collect_params().load(params_file, ctx=ctx, ignore_extra=True)
    metric = FaceVerification(nfolds)
    metric.reset()
    for (data0,data1),label in val_data:
        data0=data0.as_in_context(ctx)
        data1=data1.as_in_context(ctx)
        label=label.as_in_context(ctx)

        res0=net(data0)
        res1=net(data1)

        embedding0=res0[-1]
        embedding1=res1[-1]

        if norm:
            embedding0=preprocessing.normalize(embedding0.asnumpy())
            embedding1=preprocessing.normalize(embedding1.asnumpy())

        metric.update(label,embedding0,embedding1)

    tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
    print "{}: {:.6f}".format('lfw', accuracy)

if __name__=='__main__':
    verification()