from core import FastIG, GuidedIG
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_ig(model, data, target):
    method = FastIG(model)
    return method(data, target).squeeze()


def guided_ig(model, data, target):
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    return method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
