from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.utils.data
from final_model import Model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(1)


def main():
    # 指定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is %s' % device)

    # 数据处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)]
    )

    # 导入训练数据
    train_data = MNIST(root='./data_set', train=True,
                       transform=transform, download=False)
    train_data = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # 导入测试数据
    test_data = MNIST(root='./data_set', train=False,
                      transform=transform, download=False)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)

    # 构造网络
    net = Model()
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = torch.linspace(1, 30, steps=30, dtype=torch.int).numpy()
    accuracy = []
    best_accuracy = 0
    best_epoch = 0
    # Model
    for epoch in range(30):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_data, start=0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # 查看网络训练情况，每500步打印一次
            running_loss += loss.item()  # .item()获得张量里的值
            if i % 500 == 499:
                print('Epoch:%d | Step: %5d | train_loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        # test
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for ind, (inputs, labels) in enumerate(test_data):
                # print(inputs.shape)  # 16,1,28,28
                # print(labels.shape)  # 16
                labels = labels.to(device)
                outputs = net(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # .item()获得张量中的值
                # print(predicted.shape)  # 16
                if epoch == 29:
                    for i in range(16):
                        if predicted[i] != labels[i]:
                            plt.figure()
                            plt.imshow(inputs[i].permute([1, 2, 0]), cmap='gray')
                            plt.title("CNN predicted: {} \n Groud Truth: {}".format(labels[i], predicted[i]))
                            plt.savefig('./wrong_predicted/error[{}_{}].jpg'.format(ind, i))

        accuracy.append(correct / total)
        if correct / total >= best_accuracy:
            best_accuracy = correct / total
            best_epoch = epoch + 1

    print('Model Finished')
    print('30th epoch test accuracy is :', accuracy[-1])
    print('Best Accuracy is :{} and corresponding epoch is :{}'.format(best_accuracy, best_epoch))

    plt.figure()
    plt.plot(num_epochs, accuracy)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.title('The performance of the final model')
    plt.savefig('Model Performance.jpg')


if __name__ == '__main__':
    main()
