from cv2 import norm
import torch
import numpy as np


class AlgorithmUtils():
    def __init__(self) -> None:
        pass

    #code from cleverhans https://github.com/cleverhans-lab/cleverhans/blob/e5d00e537ce7ad6119ed5a8db1f0e9736d1f6e1d/cleverhans/torch/utils.py
    def clip_eta(self,eta, norm, eps):

        """
        PyTorch implementation of the clip_eta in utils_tf.
        :param eta: Tensor
        :param norm: np.inf, 1, or 2
        :param eps: float
        """

        if norm not in [np.inf, 1, 2]:
            raise ValueError("norm must be np.inf, 1, or 2.")

        avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
        reduc_ind = list(range(1, len(eta.size())))
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            if norm == 1:
                raise NotImplementedError("L1 clip is not implemented.")
                norm = torch.max(
                    avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
                )
            elif norm == 2:
                norm = torch.sqrt(
                    torch.max(
                        avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                    )
                )
            factor = torch.min(
                torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
            )
            eta *= factor
        return eta

    #code from cleverhans https://github.com/cleverhans-lab/cleverhans/blob/e5d00e537ce7ad6119ed5a8db1f0e9736d1f6e1d/cleverhans/torch/utils.py
    def optimize_linear(self,grad, eps, norm=np.inf):
        """
        Solves for the optimal input to a linear function under a norm constraint.
        Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
        :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
        :param eps: float. Scalar specifying size of constraint region
        :param norm: np.inf, 1, or 2. Order of norm constraint.
        :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
        """

        red_ind = list(range(1, len(grad.size())))
        avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
        if norm == np.inf:
            # Take sign of gradient
            optimal_perturbation = torch.sign(grad)
        elif norm == 1:
            abs_grad = torch.abs(grad)
            sign = torch.sign(grad)
            red_ind = list(range(1, len(grad.size())))
            abs_grad = torch.abs(grad)
            ori_shape = [1] * len(grad.size())
            ori_shape[0] = grad.size(0)

            max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
            max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
            num_ties = max_mask
            for red_scalar in red_ind:
                num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
            optimal_perturbation = sign * max_mask / num_ties
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
            assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
        elif norm == 2:
            square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
            optimal_perturbation = grad / torch.sqrt(square)
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = (
                optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
            )
            one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
                square > avoid_zero_div
            ).to(torch.float)
            assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
        else:
            raise NotImplementedError(
                "Only L-inf, L1 and L2 norms are " "currently implemented."
            )

        # Scale perturbation to be the solution for the norm=eps rather than
        # norm=1 problem
        scaled_perturbation = eps * optimal_perturbation
        return scaled_perturbation

    def OneStepFGSM(self,
                    img_,
                    model,
                    eps=0.01,
                    norm=np.inf,
                    target=None,
                    clip_min=None,
                    clip_max=None,
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
        loss = None

        prediction = model(img)
        if target == None:
            target = prediction.softmax(1).max(1).indices.cpu().detach().to(
                device=device)
            loss = loss_func(prediction, target)
        else:
            target = torch.tensor((target,),device=device)
            loss = -loss_func(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimal_perturbation = self.optimize_linear(img.grad, eps, norm)
        img.data = img.data + optimal_perturbation
        grads = optimal_perturbation

        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                )
            img = torch.clamp(img, clip_min, clip_max)

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
             clip=None,
             norm=np.inf,
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
            optimal_perturbation = self.optimize_linear(img.grad, eps, norm)
            # rr = eps * torch.sign(img.grad.data)

            optimal_perturbation[label == target] = 0
            if clip:
                img.data = img.data - self.clip_eta(optimal_perturbation,norm,clip)
            else:
                img.data = img.data - optimal_perturbation
            grads += optimal_perturbation
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
