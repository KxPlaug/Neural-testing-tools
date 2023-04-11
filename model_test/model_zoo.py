import torch
import torchvision
import models.cifar10 as cifar10_models
import models.cifar100 as cifar100_models
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name="resnet50", dataset="cifar10"):
    model = None
    
    classes_map = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
    }
    num_classes = classes_map[dataset]
    
    if dataset == "imagenet":
        model = torchvision.models.__dict__[name](pretrained=True, num_classes=num_classes)
    elif dataset == "cifar10":
        model = cifar10_models.__dict__[name](pretrained=False, num_classes=num_classes)
    elif dataset == "cifar100":
        model = cifar100_models.__dict__[name](pretrained=False, num_classes=num_classes)
    else:
        raise NotImplementedError

    if dataset != "imagenet":
        model.load_state_dict(torch.load(f"state_dicts/{dataset}/{name}.pt", map_location=device))

    model.eval()
    model.to(device)

    return model