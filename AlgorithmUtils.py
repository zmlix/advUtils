import torch
import numpy as np


class AlgorithmUtils():
    def __init__(self) -> None:
        pass

    def OneStepFGSM(self,
                    img_,
                    model,
                    eps=0.01,
                    optimizer_=None,
                    loss_func_=None,
                    grad_=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam if optimizer_ == None else optimizer_
        img = img_.cpu().clone().detach()
        img = img.to(device)
        optimizer = optimizer([img])

        img.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False

        loss_func = torch.nn.CrossEntropyLoss(
        ) if loss_func_ == None else loss_func_

        grads = 0

        target, prediction = self.frontward(model, img)
        target = torch.tensor([target], device=device).long()
        loss = loss_func(prediction, target)
        label = np.argmax(prediction.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        img.data = img.data + eps * torch.sign(img.grad.data)
        grads = eps * torch.sign(img.grad.data)

        if grad_:
            return (img, grads)
        else:
            return img

    def FGSM(self,
             img_,
             model,
             epochs=100,
             eps=0.01,
             target=0,
             optimizer_=None,
             loss_func_=None,
             info=False,
             grad_=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam if optimizer_ == None else optimizer_
        img = img_.cpu().clone().detach()
        img = img.to(device)
        optimizer = optimizer([img])

        img.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False

        loss_func = torch.nn.CrossEntropyLoss(
        ) if loss_func_ == None else loss_func_
        # print(img.requires_grad)
        # img.retain_grad()
        target = torch.tensor([target], device=device).long()

        grads = 0

        for epoch in range(epochs):
            prediction = model(img)
            # print(prediction.requires_grad)
            loss = loss_func(prediction, target)
            label = np.argmax(prediction.cpu().detach().numpy())
            if info:
                print(f'epoch {epoch} loss {loss} label {label}')
            if label == target:
                print(f"{epoch} | success to attack -> {target}")
                break
            optimizer.zero_grad()
            loss.backward()
            # print(img.grad.data)
            img.data = img.data - eps * torch.sign(img.grad.data)
            grads += eps * torch.sign(img.grad.data)
            # optimizer.step()
            # img.data = (img.data - img.data.min()) / (img.data.max() - img.data.min())
        if grad_:
            return (img, grads)
        else:
            return img

    def DeepFool(self, img_, model, epochs=100, info=False, grad_=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = img_.cpu().clone().detach()
        img = img.to(device)
        img.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False

        epochs = 100
        overshoot = 0.02
        num_classes = 1000

        orig_label, output = self.frontward(model, img)
        input_shape = self.toNumpy(img.shape)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        grads = 0

        for epoch in range(epochs):
            label, scores = self.frontward(model, img)
            if info:
                print(f'epoch={epoch} label={label} score={scores[0][label]}')
            if label != orig_label:
                break
            pert = np.inf
            output[0, orig_label].backward(retain_graph=True)
            grad_orig = img.grad.data.cpu().numpy().copy()

            for k in range(num_classes):
                if k == orig_label:
                    continue
                img.grad.data.zero_()
                output[0, k].backward(retain_graph=True)
                cur_grad = img.grad.data.cpu().numpy().copy()

                w_k = cur_grad - grad_orig
                f_k = (output[0, k] - output[0, orig_label]).data.cpu().numpy()
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # print(pert,w)

            r_i = (pert + 1e-8) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)
            img.data = img.data + (1 + overshoot) * torch.from_numpy(r_i).to(
                device=device)
            grads += (1 + overshoot) * torch.from_numpy(r_i).to(device=device)

        if grad_:
            return (img, grads)
        else:
            return img


def main():
    pass


if __name__ == "__main__":
    main()
