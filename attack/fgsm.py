import torch
import torch.nn as nn
import numpy as np


class FGSM:
    def __init__(self, data_min, data_max, epsilon=0.3 * 255, num_steps=50, alpha=0.01, early_stop=True):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.num_steps = num_steps
        self.alpha = alpha
        self.early_stop = early_stop

    def __call__(self, model, data, target):
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i + 1].clone()] for i in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])

        for _ in range(self.num_steps):
            output = model(dt)
            loss = self.criterion(output, target)
            loss.backward()
            data_grad = dt.grad.detach().sign()
            adv_data = dt + self.alpha * data_grad
            total_grad = torch.clamp(
                adv_data - data, -self.epsilon / 255, self.epsilon / 255)
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
