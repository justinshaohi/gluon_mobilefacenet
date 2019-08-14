import os
import pickle
import numpy as np

from mxnet import recordio, image,ndarray
from mxnet.gluon.data import Dataset


def _check_valid_image(s):
    return False if len(s) ==0 else True

class TrainRecordDataset(Dataset):

    def __init__(self, root, flag=1, transform=None):
        filename=os.path.join(root,'train.rec')
        self.filename = filename
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self._record = recordio.MXIndexedRecordIO(self.idx_file, self.filename, 'r')
        prop = open(os.path.join(root,"property"), "r").read().strip().split(',')
        self._flag = flag
        self._transform = transform

        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]

    def __getitem__(self, idx):
        while True:
            record = self._record.read_idx(idx)
            header, img = recordio.unpack(record)
            if _check_valid_image(img):
                decoded_img = image.imdecode(img, self._flag)
            else:
                idx = np.random.randint(low=0, high=len(self))
                continue
            if self._transform is not None:
                return self._transform(decoded_img, header.label)
            return decoded_img, header.label

    def __len__(self):
        return len(self._record.keys)

class ValDataset(Dataset):

    def __init__(self, name, root, transform=None):
        self._transform = transform
        self.name = name
        with open(os.path.join(root, "{}.bin".format(name)), 'rb') as f:
            #self.bins, self.issame_list = pickle.load(f, encoding='iso-8859-1')
            self.bins, self.issame_list = pickle.load(f)
        self._do_encode = not isinstance(self.bins[0], np.ndarray)

    def __getitem__(self, idx):
        img0 = self._decode(self.bins[2 * idx])
        img1 = self._decode(self.bins[2 * idx + 1])

        issame = 1 if self.issame_list[idx] else 0

        if self._transform is not None:
            img0 = self._transform(img0)
            img1 = self._transform(img1)

        return (img0, img1), issame
        #return (img0, img1), ndarray.array(issame)

    def __len__(self):
        return len(self.issame_list)

    def _decode(self, im):
        # if self._do_encode:
        #     im = im.encode("iso-8859-1")
        return image.imdecode(im)

