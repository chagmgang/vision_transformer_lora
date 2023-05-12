from model.lora_model import LoRAVisionTransformer
from utils.build_optim import build_optimizer
import torch
import torchvision
from tqdm import tqdm

def main():

    img_size = 224
    patch_size = 16
    batch_size = 128
    num_workers = 16

    device = torch.device('cuda')
    
    model = LoRAVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        with_cp=True,
    )
    model = model.to(device)
    state_dict = torch.load('model-vit-b-checkpoint-1599.pth', map_location='cpu')
    # load state dict
    print(model.load_state_dict(state_dict, strict=False))
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # build optimizer
    optimizer = build_optimizer(
        model,
        learning_rate=1e-4 * batch_size / 256,
        weight_decay=0.05,
    )

    # build dataset
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.49421427, 0.48513138, 0.45040908),
            std=(0.06047972, 0.06123986, 0.06758436),
        ),
    ])
    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.49421427, 0.48513138, 0.45040908),
            std=(0.06047972, 0.06123986, 0.06758436),
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=train_transform,
    )
    validset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=valid_transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch  in range(10):

        model.train()
        for data in tqdm(trainloader):

            optimizer.zero_grad()
            img, label = data
            img = img.to(device)
            label = label.to(device)
            logit = model(img)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()

        logits = list()
        labels = list()
        model.eval()
        for data in tqdm(validloader):

            img, label = data
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(img)
            logits.extend(logit)
            labels.extend(label)

        logits = torch.stack(logits)
        pred = torch.argmax(logits, axis=1)
        labels = torch.stack(labels)
        acc = torch.eq(pred, labels)
        print(torch.sum(acc) / len(logits))

if __name__ == '__main__':
    main()
