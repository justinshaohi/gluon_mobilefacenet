import os
import time
import datetime
import logging
import argparse
import mxnet as mx
from mxnet import autograd
from model import MobileFacenet
from data import TrainRecordDataset
from loss import L2Softmax

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='./data',help='rec data path')
parser.add_argument('--batch_size',type=int,default=128,help='training batch size')
parser.add_argument('--epoch',type=int,default=50,help='training epoch num')
parser.add_argument('--out',type=str,default='./out',help='path to save training log and params')
parser.add_argument('--interval',type=int,default=100,help='training console log print interval')
parser.add_argument('--model',type=str,default='',help='pretrained model')

args=parser.parse_args()

formatter=logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
log_file_handler=logging.FileHandler(os.path.join(args.out,'train-{}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),mode='w')
log_stream_handler=logging.StreamHandler()
log_file_handler.setFormatter(formatter)
log_stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)


ctx=mx.gpu()

transform = mx.gluon.data.vision.transforms.Compose([
    mx.gluon.data.vision.transforms.RandomBrightness(0.3),
    mx.gluon.data.vision.transforms.RandomContrast(0.3),
    mx.gluon.data.vision.transforms.RandomSaturation(0.3),
    mx.gluon.data.vision.transforms.RandomFlipLeftRight(),
    mx.gluon.data.vision.transforms.ToTensor()
])

def transform_train(data, label=None):
    im = transform(data)
    return im, label


def check_data(data_dir):
    rec_data=os.path.join(data_dir,'train.rec')
    idx_data=os.path.join(data_dir,'train.idx')
    property_file=os.path.join(data_dir,'train.idx')
    assert os.path.exists(rec_data) and os.path.exists(idx_data) and os.path.exists(property_file),'data directory is error'


def train(data_dir,batch_size,epoch_num,interval,train_out,model):
    train_dataset=TrainRecordDataset(data_dir,transform=transform_train)
    train_data=mx.gluon.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)

    if not os.path.exists(model):
        class_num=train_dataset.num_classes
        net=MobileFacenet(classes=class_num)
        net.hybridize()
        net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)

    lr_sch=mx.lr_scheduler.MultiFactorScheduler(base_lr=0.1,step=[140000,200000,240000],factor=0.1)
    trainer=mx.gluon.Trainer(net.collect_params(),'nag',{'learning_rate': 0.1,'wd': 5e-4, 'momentum': 0.9,'lr_scheduler':lr_sch})
    SML = L2Softmax(class_num, alpha=64, from_normx=True)

    train_loss = mx.metric.Loss()
    train_acc = mx.metric.Accuracy()

    step=0
    train_save=os.path.join(train_out,'mobilefacenet')
    for epoch in range(1,int(epoch_num + 1)):
        for data,labels in train_data:
            tic = time.time()
            data=data.as_in_context(ctx)
            labels=labels.as_in_context(ctx)
            with autograd.record():
                outputs=net(data)
                loss=SML(outputs,labels)
            loss.backward()
            trainer.step(batch_size)

            train_loss.update(0, loss)
            train_acc.update(labels=labels,preds=outputs)

            step=step+1
            step_cost_time=time.time()-tic
            if step % interval==0:
                _,loss_value=train_loss.get()
                train_loss.reset()
                _, acc_value = train_acc.get()
                train_acc.reset()

                logger.info('epoch {} step {} :lr: {}, loss {:.3f}, acc {:.6f}, per step cost time {:.3f} sec'.format(epoch,step,trainer.learning_rate,loss_value,acc_value,step_cost_time))


        logger.info('save model to {}_{:{}>4}.params'.format(train_save,epoch,'0'))
        net.export(train_save,epoch=epoch)


if __name__ =='__main__':
    data_dir=args.data
    batch_size=args.batch_size
    epoch_num=args.epoch
    interval=args.interval
    train_out=args.out
    model=args.model
    logger.info('train config: data {} , batch_size {} , epoch_num {}, interval {} , train out save to {} ,pretrained model {}'
                .format(data_dir,batch_size,epoch_num,interval,train_out,model))
    check_data(data_dir)
    train(data_dir,batch_size,epoch_num,interval,train_out,model)