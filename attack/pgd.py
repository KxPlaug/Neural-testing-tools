import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PGD:
    def __init__(self, data_min, data_max, epsilon=0.3 * 255, num_steps=50, alpha=0.01, random_init=True, norm='Inf', early_stop=True):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.num_steps = num_steps
        self.alpha = alpha
        self.random_init = random_init
        self.norm = norm
        self.early_stop = early_stop

    def __call__(self, model, data, target):
        if self.random_init:
            if self.norm == 'L2':
                random_noise = (torch.randn(*data.shape) *
                                self.epsilon / 255).to(device)
            else:  # Inf PGD
                random_noise = torch.FloatTensor(
                    *data.shape).uniform_(-self.epsilon / 255, self.epsilon / 255).to(device)
            dt = (data + random_noise).clone().detach().requires_grad_(True)
        else:
            dt = data.clone().detach().requires_grad_(True)

        target_clone = target.clone()
        hats = [[data[i:i + 1].clone()] for i in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])

        for _ in range(self.num_steps):
            output = model(dt)
            loss = self.criterion(output, target)
            loss.backward()
            data_grad = dt.grad.detach()

            if self.norm == 'L2':
                data_grad = data_grad / \
                    (data_grad.view(
                        data_grad.shape[0], -1).norm(dim=-1).view(-1, 1, 1, 1) + 1e-8)
            else:  # Inf PGD
                data_grad = data_grad.sign()

            adv_data = dt + self.alpha * data_grad
            total_grad = adv_data - data

            if self.norm == 'L2':
                total_grad = torch.renorm(total_grad.view(
                    total_grad.shape[0], -1), p=2, dim=-1, maxnorm=self.epsilon / 255).view_as(total_grad)
            else:  # Inf PGD
                total_grad = torch.clamp(
                    total_grad, -self.epsilon / 255, self.epsilon / 255)

            dt.data = torch.clamp(
                data + total_grad, self.data_min, self.data_max)
            dt.grad.zero_()

            for i, idx in enumerate(leave_index):
                hats[idx].append(dt[i:i + 1].data.clone())

            if self.early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                keep_index = (adv_pred_argmax == target).nonzero(
                    as_tuple=True)[0].detach().cpu().numpy()
                if len(keep_index) == 0:
                    break

                dt = dt[keep_index].detach().requires_grad_(True)
                data = data[keep_index]
                target = target[keep_index]
                leave_index = leave_index[keep_index]

        dt = torch.cat([hat[-1] for hat in hats], dim=0).requires_grad_(True)
        adv_pred = model(dt)
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred
