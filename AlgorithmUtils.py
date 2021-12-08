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

        prediction = model(img)
        target = prediction.softmax(1).max(1).indices.cpu().detach().to(
            device=device)
        loss = loss_func(prediction, target)
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
        target = torch.full((img.shape[0], ), target).to(device=device).long()

        grads = 0

        for epoch in range(epochs):
            prediction = model(img)
            # print(prediction.requires_grad)
            loss = loss_func(prediction, target)
            label = prediction.softmax(1).max(1).indices.cpu().detach().to(
                device=device)
            if info:
                print(
                    f'epoch {epoch} loss {loss} label {label} score={prediction.softmax(1)[range(img.shape[0]),label]}'
                )
            if (label == target).all() and info:
                print(f"{epoch} | success to attack -> {target}")
                break
            optimizer.zero_grad()
            loss.backward()
            # print(img.grad.data)
            rr = eps * torch.sign(img.grad.data)
            rr[label == target] = 0
            img.data = img.data - rr
            grads += rr
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

        output = model(img)
        orig_label = output.softmax(1).max(1).indices.cpu().detach().to(
            device=device)

        input_shape = self.toNumpy(img.shape)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        grads = 0

        for epoch in range(epochs):
            scores = model(img)
            label = scores.softmax(1).max(1).indices.cpu().detach().to(
                device=device)
            if info:
                print(
                    f'epoch={epoch} label={label} score={scores.softmax(1)[range(orig_label.shape[0]),label]}'
                )
            if (label != orig_label).all():
                break
            pert = np.array([np.inf for _ in range(orig_label.shape[0])])
            output[range(orig_label.shape[0]),
                   orig_label.long()].backward(
                       output[range(orig_label.shape[0]),
                              orig_label.long()].clone().detach(),
                       retain_graph=True)
            grad_orig = img.grad.data.cpu().numpy().copy()

            for k in range(num_classes):
                k_ = torch.full_like(orig_label, k)
                select_k = (k_ != orig_label) & (label == orig_label)
                select_index = torch.tensor(range(
                    orig_label.shape[0]))[select_k].long()
                img.grad.data.zero_()
                output[select_index, k].backward(output[select_index,
                                                        k].clone().detach(),
                                                 retain_graph=True)
                cur_grad = img.grad.data.cpu().numpy().copy()

                w_k = cur_grad - grad_orig
                f_k = (output[range(orig_label.shape[0]), k] -
                       output[range(orig_label.shape[0]),
                              orig_label]).data.cpu().numpy()
                pert_k = abs(f_k) / torch.norm(
                    torch.tensor(w_k), p=2,
                    dim=(1, 2, 3)).cpu().clone().detach().numpy()

                select_pert = (pert_k < pert)
                select_pert[np.arange(
                    orig_label.shape[0])[pert_k == 0]] = False
                pert[select_pert] = pert_k[select_pert]
                w[select_pert] = w_k[select_pert]

            r_i = (pert.reshape(orig_label.shape[0], 1, 1, 1) +
                   1e-8) * w / torch.norm(torch.tensor(w), 2, dim=(
                       1, 2, 3)).cpu().clone().detach().numpy().reshape(
                           orig_label.shape[0], 1, 1, 1)
            r_i = r_i.astype(np.float32)
            r_tot = np.float32(r_tot + r_i)
            rr = (1 + overshoot) * torch.from_numpy(r_tot).to(device=device)
            rr[label != orig_label] = 0
            img.data = img.data + rr
            grads += rr

        if grad_:
            return (img, grads)
        else:
            return img


def main():
    pass


if __name__ == "__main__":
    main()
