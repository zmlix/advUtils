from collections import Iterable
from numpy.core.fromnumeric import clip
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageUtils():
    def __init__(self) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def type(self, data):
        return type(data)

    def isNdarray(self, data):
        return isinstance(data, np.ndarray)

    def toNumpy(self, data):
        if self.isNdarray(data):
            return data
        else:
            if isinstance(data, Iterable):
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach()
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

    def restoreImage(self, img, normal=True):
        isTensor = isinstance(img, torch.Tensor)
        img = self.CWH2WHC(img)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img = (img * std) + mean
        # if normal:
        #     return np.clip(img, 0, 1)
        # else:
        #     img = img * 255.0
        #     return np.clip(img, 0, 255).astype(np.uint8)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1,
                                                           3).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1,
                                                          3).to(self.device)
        img = torch.tensor(img).to(self.device)
        # print(img.shape)
        img = (img * std) + mean
        res = None
        if normal:
            res = img.clip(0, 1).cpu().clone().detach()
        else:
            img = img * 255.0
            res = img.clip(0, 255).type(torch.uint8).cpu().clone().detach()

        if isTensor:
            return res
        else:
            return res.numpy()

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

    def norm(self, img, adv_img, L=2, dim=(1, 2, 3), mean=False):
        if mean:
            return torch.norm(img - adv_img, L, dim=dim).mean()
        else:
            return torch.norm(img - adv_img, L, dim=dim)

    def calculate_ssim(self,img1, img2,mean = True):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(img1.shape[2]):
                    ssims.append(self.ssim(img1[..., i], img2[..., i]))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        elif img1.ndim == 4:
            ssimList = []
            for idx in range(img1.shape[0]):
                if img1[idx].shape[2] == 3:
                    ssims = []
                    for i in range(img1[idx].shape[2]):
                        ssims.append(self.ssim(img1[idx][..., i], img2[idx][..., i]))
                    ssimList.append(np.array(ssims).mean())
                elif img1[idx].shape[2] == 1:
                    ssimList.append(self.ssim(np.squeeze(img1[idx]), np.squeeze(img2[idx])))
            res = torch.tensor(ssimList)
            if mean:
                return res.mean()
            else:
                return res
        else:
            raise ValueError('Wrong input image dimensions.')


    def ssim(self,img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    def advAccuracy(self,
                    true_labels,
                    classifier_labels,
                    adv_labels,
                    detail=False):
        # true labels
        a = true_labels
        # classifier labels
        b = classifier_labels
        # adv labels
        c = adv_labels
        # print(a == b)
        # print(b == c)
        # print((a == b) & (b != c))
        # print((a == b).sum(), ((a == b) & (b != c)).sum())
        if detail:
            return (a == b).sum(), (b != c).sum(), ((a == b) & (b != c)).sum()
        else:
            return ((a == b) & (b != c)).sum() / (a == b).sum()


def main():
    pass


if __name__ == "__main__":
    main()
