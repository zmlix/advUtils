import os, torch, sys
import numpy as np
from PIL import Image
from ImageUtils import ImageUtils
from AlgorithmUtils import AlgorithmUtils
from PlotUtils import PlotUtils


class Utils(ImageUtils, AlgorithmUtils, PlotUtils):
    def __init__(self) -> None:
        super().__init__()

    def frontward(self, model, img):
        ''' 前向传播 '''
        prediction = model(img)
        label = np.argmax(prediction.cpu().detach().numpy())
        return (label, prediction)

    def argmax(self, p, axis=1):
        ''' 获取预测标签 '''
        return np.argmax(p.cpu().detach().numpy(), axis)

    def topN(self, x, n=0):
        return x.sort(dim=1, descending=True)[0][0][n]

    def getSoftmax(self, x):
        softmax = torch.nn.Softmax(1)(x)
        return softmax

    def confidence(self, x):
        return x.softmax(1).max().item()

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
        import os
        root_path = '/home/data/imagenetval/'
        root_categories = os.listdir(root_path)
        categoriesLen = len(root_categories)

        res = []
        for _ in range(num):
            randomIdx = np.random.randint(0, categoriesLen)
            if label:
                label_ = self.dataLabelOfImageNet2012()(
                    root_categories[randomIdx])
            path = root_path + root_categories[randomIdx] + '/'

            categories = os.listdir(path)
            categoriesLen = len(categories)
            # category = categories[categories.index(name)]
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

    def PredictImage(self, model, img, print_=False):
        labelInfo = self.dataLabelOfImageNet2012()
        label, confidence = self.frontward(model, img)
        if print_:
            print(f'{labelInfo(label)} {self.confidence(confidence)}')
        return labelInfo(label), self.confidence(confidence)

    def sequential(self, sq):
        def s(x):
            for func in sq:
                if isinstance(func, tuple) or isinstance(func, list):
                    x = func[0](x, **func[1])
                else:
                    x = func(x)
            return x

        return s