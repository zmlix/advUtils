from collections import Iterable
import os
import torch
import numpy as np
from PIL import Image
from ImageUtils import ImageUtils
from AlgorithmUtils import AlgorithmUtils
from PlotUtils import PlotUtils


class Utils(ImageUtils, AlgorithmUtils, PlotUtils):
    def __init__(self) -> None:
        super().__init__()

    def frontward(self, model, imgs):
        ''' 前向传播 '''
        predictions = model(imgs)
        labels = predictions.softmax(1).max(1).indices.cpu().detach()
        labels = labels[0].item() if len(labels) == 1 else labels
        return (labels, predictions)

    def argmax(self, p, axis=1):
        ''' 获取预测标签 '''
        return np.argmax(p.cpu().detach().numpy(), axis)

    def topN(self, x, n=0):
        return x.sort(dim=1, descending=True)[0][0][n]

    def getSoftmax(self, x):
        softmax = torch.nn.Softmax(1)(x)
        return softmax

    def confidence(self, x):
        c = x.softmax(1).max(1).values.cpu().detach()
        c = c[0].item() if len(c) == 1 else c
        return c

    def diff_image(self, imgA, imgB):
        ''' 比较两张图片的差异 '''
        # if self.isNotNormalize(imgA):
        #     imgA = self.Normalize(imgA)
        # if self.isNotNormalize(imgB):
        #     imgB = self.Normalize(imgB)

        diff = imgB - imgA
        # diff = (diff - diff.min()) / (diff.max() - diff.min())
        diff = diff / abs(diff).max() / 2.0 + 0.5

        return diff

    def dataLabelOfImageNet2012(self, url=None):
        ''' 获取ImageNet2012数据集的标签数据 '''
        dirname, _ = os.path.split(os.path.abspath(__file__))
        url = dirname + '/ImageNet2012.txt' if url == None else url
        data = open(url, mode='r', encoding='utf-8')

        labels = list()
        for label in data.readlines():
            label = label.strip('\n')
            tmp = label.split(' ')
            name = ''.join(tmp[2:])
            labels.append([int(tmp[0]), tmp[1], name.split(',')[0], name])

        def run(idxs):
            if isinstance(idxs, Iterable) and isinstance(idxs, torch.Tensor):
                res = list()
                for idx_ in idxs:
                    idx = str(idx_.item())
                    if idx.isdigit():
                        res.append(labels[int(idx)])
                    if idx[0] == 'n':
                        res.append(
                            list(filter(lambda item: item[1] == idx,
                                        labels))[0])
                return res
            idx = str(idxs)
            if idx.isdigit():
                # print(labels)
                return labels[int(idx)]
            if idx[0] == 'n':
                return list(filter(lambda item: item[1] == idx, labels))[0]

        return run

    def getImageNetImg(self, name, idx=-1, num=1, w=224, h=224, label=False):
        if label:
            label_ = self.dataLabelOfImageNet2012()(name)
        path = '/home/data/imagenetval/' + name + '/'
        categories = os.listdir(path)
        categoriesLen = len(categories)
        # category = categories[categories.index(name)]
        res = []
        img = None

        if num < 0:
            for file_name in categories:
                img = Image.open(path + file_name).resize((w, h))
                if img.mode != 'RGB':
                    continue
                if label:
                    res.append((img, label_))
                else:
                    res.append(img)
            return res

        cnt = 0
        while cnt < min(num, categoriesLen):
            if idx == -1:
                img = Image.open(
                    path +
                    categories[np.random.randint(0, categoriesLen)]).resize(
                        (w, h))
                if img.mode != 'RGB':
                    continue
                if label:
                    res.append((img, label_))
                else:
                    res.append(img)
            else:
                img = Image.open(path + categories[idx]).resize((w, h))
                if img.mode != 'RGB':
                    print("image is not RGB")
                    return []
                if label:
                    res.append((img, label_))
                else:
                    res.append(img)
            cnt += 1

        return res[0] if len(res) == 1 else res

    def getImageNetImgRandom(self, num=1, w=224, h=224, label=False):
        import os
        root_path = '/home/data/imagenetval/'
        root_categories = os.listdir(root_path)
        categoriesLen = len(root_categories)

        res = []
        img = None
        cnt = 0
        while cnt < num:
            randomIdx = np.random.randint(0, categoriesLen)
            if label:
                label_ = self.dataLabelOfImageNet2012()(
                    root_categories[randomIdx])
            path = root_path + root_categories[randomIdx] + '/'

            categories = os.listdir(path)
            categoriesLen = len(categories)
            # category = categories[categories.index(name)]
            img = Image.open(
                path + categories[np.random.randint(0, categoriesLen)]).resize(
                    (w, h))
            if img.mode != 'RGB':
                continue
            if label:
                res.append((img, label_))
            else:
                res.append(Image.open(img))
            cnt += 1

        return res[0] if len(res) == 1 else res

    def PredictImage(self, model, imgs, print_=False):
        labelInfo = self.dataLabelOfImageNet2012()
        labels, predictions = self.frontward(model, imgs)
        if print_:
            print(f'{labelInfo(labels)} {self.confidence(predictions)}')
        return labelInfo(labels), self.confidence(predictions)

    def sequential(self, sq):
        def s(x):
            for func in sq:
                if isinstance(func, tuple) or isinstance(func, list):
                    x = func[0](x, **func[1])
                else:
                    x = func(x)
            return x

        return s

    def labelsError(self, a, b, error=True):
        a = list(zip(*a))[0]
        b = list(zip(*b))[0]
        if error:
            return len(a) - (np.array(a) == np.array(b)).sum()
        else:
            return (np.array(a) == np.array(b)).sum()
