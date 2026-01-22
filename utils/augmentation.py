import numpy as np
import torch
import torchvision.transforms as transforms

# Apply Random Cutout: Randomly masks out a square region from the input image
class RandomCutout(object):
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        Returns:
            Tensor: Cutout image.
        """
        _, h, w = img.shape
        cutout_size = self.size

        top = np.random.randint(0, h - cutout_size)
        left = np.random.randint(0, w - cutout_size)

        img[:, top:top + cutout_size, left:left + cutout_size] = 0
        return img
    


def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # print(lam)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


rand_aug_cifar10 = [
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    RandomCutout(size=8),
]

rand_aug_cifar100 = [
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
    RandomCutout(size=8), 
]

rand_aug_mnist = [
    transforms.Grayscale(3),      
    transforms.ToPILImage(),
    transforms.Resize(32),        
    transforms.RandomRotation(15), 
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
    RandomCutout(size=8),
]