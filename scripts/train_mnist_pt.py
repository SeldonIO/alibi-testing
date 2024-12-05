import argparse
import torch
import torch.nn.functional as F
import torchvision

from alibi_testing.modules import CNN
from utils import save_model


def load_data():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
        root='datasets',
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='datasets',
        train=False,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    return train_loader, test_loader


def train(model, optimizer, train_loader, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            test_loss += F.cross_entropy(out, target, reduce=None).item()
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )


def run_model(n_epochs = 1):
    train_loader, test_loader = load_data()

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, n_epochs + 1):
        train(model, optimizer, train_loader, epoch)
        test(model, test_loader)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["cnn"], default="cnn", help='Name of the model to be saved')
    parser.add_argument('--format', type=str, choices=["pt"], default="pt")
    args = parser.parse_args()

    # train model
    model = run_model()

    # save trained mode
    save_model(
        model,
        args,
        model_name=args.model,
        data="mnist",
        framework="pt",
        version=torch.__version__.split("+")[0]
    )
