import os, torch
import numpy as np
from PIL import Image
from Utils.ImageUtils import ImageUtils
from Utils.AlgorithmUtils import AlgorithmUtils
from Utils.PlotUtils import PlotUtils


class Utils(ImageUtils, AlgorithmUtils, PlotUtils):
    def __init__(self) -> None:
        super().__init__()

    ''' 前向传播 '''

    def frontward(self, model, img):
        prediction = model(img)
        label = np.argmax(prediction.cpu().detach().numpy())
        return (label, prediction)

    ''' 获取预测标签 '''

    def argmax(self, p, axis=1):
        return np.argmax(p.cpu().detach().numpy(), axis)

    def topN(self, x, n=0):
        return x.sort(dim=1, descending=True)[0][0][n]

    def getSoftmax(self, x):
        softmax = torch.nn.Softmax(1)(x)
        return softmax

    def confidence(self, x):
        return x.softmax(1).max().item()

    ''' 比较两张图片的差异 '''

    def diff_image(self, imgA, imgB):
        # if self.isNotNormalize(imgA):
        #     imgA = self.Normalize(imgA)
        # if self.isNotNormalize(imgB):
        #     imgB = self.Normalize(imgB)

        diff = imgB - imgA
        # diff = (diff - diff.min()) / (diff.max() - diff.min())
        diff = diff / abs(diff).max() / 2.0 + 0.5

        return diff

    ''' 获取ImageNet2012数据集的标签数据 '''

    def dataLabelOfImageNet2012(self, url=None):
        url = './ImageNet2012.txt' if url == None else url
        data = open(url, mode='r', encoding='utf-8')

        labels = list()
        for label in data.readlines():
            label = label.strip('\n')
            tmp = label.split(' ')
            name = ''.join(tmp[2:])
            labels.append([int(tmp[0]), tmp[1], name.split(',')[0], name])

        def run(idx):
            idx = str(idx)
            if idx.isdigit():
                # print(labels)
                return labels[int(idx)]
            if idx[0] == 'n':
                return list(filter(lambda item: item[1] == idx, labels))[0]

        return run

    def getImageNetImg(self, name, idx=-1, num=1, label=False):
        if label:
            label_ = self.dataLabelOfImageNet2012()(name)
        path = '/home/data/imagenetval/' + name + '/'
        categories = os.listdir(path)
        categoriesLen = len(categories)
        # category = categories[categories.index(name)]
        res = []
        for _ in range(0, min(num, categoriesLen)):
            if idx == -1:
                if label:
                    res.append((Image.open(path + categories[np.random.randint(
                        0, categoriesLen)]).resize((224, 224)), label_))
                else:
                    res.append(
                        Image.open(path + categories[np.random.randint(
                            0, categoriesLen)]).resize((224, 224)))
            else:
                if label:
                    res.append((Image.open(path + categories[idx]).resize(
                        (224, 224)), label_))
                else:
                    res.append(
                        Image.open(path + categories[idx]).resize((224, 224)))

        return res[0] if len(res) == 1 else res

    def getImageNetImgRandom(self, num=1, label=False):
        path = '/home/data/imagenetval/'
        categories = os.listdir(path)
        categoriesLen = len(categories)

        randomIdx = np.random.randint(0, categoriesLen)
        if label:
            label_ = self.dataLabelOfImageNet2012()(categories[randomIdx])
        path = path + categories[randomIdx] + '/'
        categories = os.listdir(path)
        categoriesLen = len(categories)
        # category = categories[categories.index(name)]
        res = []
        for _ in range(0, min(num, categoriesLen)):
            if label:
                res.append((Image.open(
                    path +
                    categories[np.random.randint(0, categoriesLen)]).resize(
                        (224, 224)), label_))
            else:
                res.append(
                    Image.open(path +
                               categories[np.random.randint(0, categoriesLen)]
                               ).resize((224, 224)))

        return res[0] if len(res) == 1 else res