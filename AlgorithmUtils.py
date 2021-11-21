import torch
import numpy as np


class AlgorithmUtils():
    def __init__(self) -> None:
        pass

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
                print(f"success to attack -> {target}")
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


def main():
    pass


if __name__ == "__main__":
    main()
