from core import FastIG, GuidedIG, pgd_step,BIG,FGSM
import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    return method(data, target).squeeze()


def guided_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
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


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)
    # get the index of the max log-probability
    init_pred = output.max(1, keepdim=True)[1]

    top_ids = selected_ids  # only for predefined ids
    # initialize the step_grad towards all target false classes
    step_grad = 0
    # num_class = 1000 # number of total classes
    for l in top_ids:
        targeted = torch.tensor([l]).to(device)
        if targeted.item() == init_pred.item():
            if l < 999:
                # replace it with l + 1
                targeted = torch.tensor([l+1]).to(device)
            else:
                # replace it with l + 1
                targeted = torch.tensor([l-1]).to(device)
            # continue # we don't want to attack to the predicted class.

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()  # / topk
    return adv_ex

def big(model,data,target,data_min=0,data_max=1,epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map