from collections import Iterable
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageUtils():
    def __init__(self) -> None:
        pass

    def type(self, data):
        return type(data)

    def isNdarray(self, data):
        return isinstance(data, np.ndarray)

    def toNumpy(self, data):
        if self.isNdarray(data):
            return data
        else:
            if isinstance(data, Iterable):
                datas = []
                for d in data:
                    datas.append(np.array(d))
                return np.array(datas)
            return np.array(data)

    def channelTypeOfImage(self, img):
        try:
            shape = img.shape
            dim = len(shape)
            if dim == 3:
                if shape[0] == 3:
                    return (0, 'CWH')
                if shape[2] == 3:
                    return (1, 'WHC')
            elif dim == 4:
                if shape[1] == 3:
                    return (2, 'NCWH')
                if shape[3] == 3:
                    return (3, 'NWHC')
            else:
                return (-1, 'Other')
        except:
            pass
            # print('not image data')

    def CWH2WHC(self, img):
        ctype = self.channelTypeOfImage(img)
        if ctype[1] in ['WHC', 'NWHC']:
            return img
        if not self.isNdarray(img):
            img = self.toNumpy(img)
        if ctype[1] == 'CWH':
            return img.transpose(1, 2, 0)
        elif ctype[1] == 'NCWH':
            return img.transpose(0, 2, 3, 1)
        else:
            print("can't CWH2WHC")

    def WHC2CWH(self, img):
        ctype = self.channelTypeOfImage(img)
        if ctype[1] in ['CWH', 'NCWH']:
            return img
        if not self.isNdarray(img):
            img = self.toNumpy(img)
        if ctype[1] == 'WHC':
            return img.transpose(2, 0, 1)
        elif ctype[1] == 'NWHC':
            return img.transpose(0, 3, 1, 2)
        else:
            print("can't WHC2CWH")

    def isNotNormalize(self, img):
        if (img > 1.0).any() or (img < 0).any():
            return True
        return False

    def Standardize(self, img, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean == None else mean
        std = [0.229, 0.224, 0.225] if std == None else std
        img = (img / 255.0 - mean) / std
        return img

    def Normalize(self, img):
        if self.isNotNormalize(img):
            return (img - img.min()) / (img.max() - img.min())
        return img

    def PreprocessImage(self, img, standardize=True):
        img = self.toNumpy(img)
        if standardize:
            img = self.Standardize(img)
        else:
            img = img / 255.0
        img = self.WHC2CWH(img)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        return img

    def restoreImage(self, img, normal=False):
        img = self.CWH2WHC(img)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img * std) + mean
        if normal:
            return np.clip(img, 0, 1)
        else:
            img = img * 255.0
            return np.clip(img, 0, 255).astype(np.uint8)

    def draw(self, img, row=1, col=1, width=15, height=4):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig = plt.figure(figsize=(width, height))
        if isinstance(img, list):
            if row * col < len(img):
                if len(img) <= 5:
                    col = len(img)
                    row = 1
                else:
                    col = 5
                    row = len(img) // col + 1
            for i in range(1, row * col + 1):
                idx = i - 1
                if idx >= len(img):
                    plt.show()
                    return
                ax = fig.add_subplot(row, col, i)
                ax.set_yticks([])
                ax.set_xticks([])
                if isinstance(img[idx], tuple) or isinstance(img[idx], list):
                    ax.set_title(img[idx][1])
                    ax.imshow(self.CWH2WHC(img[idx][0]))
                else:
                    ax.imshow(self.CWH2WHC(img[idx]))
            plt.show()
        else:
            plt.imshow(self.CWH2WHC(img))

    def tensorToNumpyImg(self, img, clip=False):
        if clip:
            return np.clip(self.CWH2WHC(img.cpu().detach().numpy()), 0, 1)
        else:
            return self.CWH2WHC(img.cpu().detach().numpy())

    def numpyToTensor(self, img, device='cpu', grad=True):
        return torch.from_numpy(img).to(
            device=device).float().requires_grad_(grad)

    def L0(self, img, img_adv, relative=False):
        deta = img - img_adv
        l0 = len(np.where(np.abs(deta) > 0.0)[0])
        if relative:
            size = (img.shape[0]) * (img.shape[1]) * (img.shape[2])
            l0 = int(l0 * 99 / size) + 1
        return l0

    def L1(self, img, img_adv, relative=False):
        deta = img - img_adv
        l1 = np.sum(np.abs(deta))
        if relative:
            l1 = int(99 * np.sum(np.abs(img[0] - img_adv[0])) /
                     np.sum(np.abs(img[0]))) + 1
        return l1

    def L2(self, img, img_adv, relative=False):
        deta = img - img_adv
        l2 = np.linalg.norm(deta)
        if relative:
            l2 = int(99 * l2 / np.linalg.norm(img)) + 1
        return l2

    def L_inf(self, img, img_adv, relative=False):
        deta = img - img_adv
        l_inf = np.max(np.abs(deta))
        if relative:
            l_inf = int(99 * l_inf / 255) + 1
        return l_inf

    def summary(self, img):
        return (img.min().item(), img.max().item(),
                not self.isNotNormalize(img))

    def show_diff(self, img, img_adv, relative=False):
        return (self.L0(img, img_adv,
                        relative), self.L1(img, img_adv, relative),
                self.L2(img, img_adv,
                        relative), self.L_inf(img, img_adv, relative))


def main():
    pass


if __name__ == "__main__":
    main()
