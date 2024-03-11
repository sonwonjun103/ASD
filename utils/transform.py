import torchvision
import random

degree1 = random.randint(0, 90)
degree2 = random.randint(90, 180)
degree3 = random.randint(90, 135)
degree4 = random.randint(135, 180)

print(degree1, degree2, degree3, degree4)

def get_aug(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAdjustSharpness(5, p=0.5),
        torchvision.transforms.RandomRotation(degrees=degree1),
        torchvision.transforms.RandomRotation(degrees=degree2),
        torchvision.transforms.RandomRotation(degrees=degree3),
        torchvision.transforms.RandomRotation(degrees=degree4),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size)),
        torchvision.transforms.ToTensor()
    ])