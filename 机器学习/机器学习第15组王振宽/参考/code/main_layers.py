from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.utils.data
from Differ_CNN_layers import Model_1, Model_2, Model_3, Model_4
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
    test_data = torch.utils.data.DataLoader(test_data, shuffle=True, num_workers=4)

    # 构造网络
    net_1 = Model_1()
    net_1.to(device)

    net_2 = Model_2()
    net_2.to(device)

    net_3 = Model_3()
    net_3.to(device)

    net_4 = Model_4()
    net_4.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(net_1.parameters(), lr=0.001)
    optimizer_2 = optim.Adam(net_2.parameters(), lr=0.001)
    optimizer_3 = optim.Adam(net_3.parameters(), lr=0.001)
    optimizer_4 = optim.Adam(net_4.parameters(), lr=0.001)

    # 训练网络
    # net_1.train()  None BN and Dropout
    num_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    correct_1 = 0
    total_1 = 0
    accuracy_1 = []

    correct_2 = 0
    total_2 = 0
    accuracy_2 = []

    correct_3 = 0
    total_3 = 0
    accuracy_3 = []

    correct_4 = 0
    total_4 = 0
    accuracy_4 = []

    # Model_1
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_data, start=0):
            inputs, labels = data
            optimizer_1.zero_grad()
            outputs = net_1(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_1.step()
            # 查看网络训练情况，每500步打印一次
            running_loss += loss.item()   # .item()获得张量里的值
            if i % 500 == 499:
                print('Epoch:%d | Step: %5d | train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        # test accuracy
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                outputs = net_1(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_1 += labels.size(0)
                correct_1 += (predicted == labels.to(device)).sum().item()   # .item()获得张量中的值
        accuracy_1.append(correct_1 / total_1)
    print('Model_1 Finished')

    # Model_2
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_data, start=0):
            inputs, labels = data
            optimizer_2.zero_grad()
            outputs = net_2(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_2.step()
            # 查看网络训练情况，每500步打印一次
            running_loss += loss.item()  # .item()获得张量里的值
            if i % 500 == 499:
                print('Epoch:%d | Step: %5d | train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        # test accuracy
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                outputs = net_2(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_2 += labels.size(0)
                correct_2 += (predicted == labels.to(device)).sum().item()  # .item()获得张量中的值
        accuracy_2.append(correct_2 / total_2)
    print('Model_2 Finished')

    # Model_3
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_data, start=0):
            inputs, labels = data
            optimizer_3.zero_grad()
            outputs = net_3(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_3.step()
            # 查看网络训练情况，每500步打印一次
            running_loss += loss.item()  # .item()获得张量里的值
            if i % 500 == 499:
                print('Epoch:%d | Step: %5d | train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        # test accuracy
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                outputs = net_3(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_3 += labels.size(0)
                correct_3 += (predicted == labels.to(device)).sum().item()  # .item()获得张量中的值
        accuracy_3.append(correct_3 / total_3)
    print('Model_3 Finished')

    # Model_4
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_data, start=0):
            inputs, labels = data
            optimizer_4.zero_grad()
            outputs = net_4(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_4.step()
            # 查看网络训练情况，每500步打印一次
            running_loss += loss.item()  # .item()获得张量里的值
            if i % 500 == 499:
                print('Epoch:%d | Step: %5d | train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        # test accuracy
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                outputs = net_4(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_4 += labels.size(0)
                correct_4 += (predicted == labels.to(device)).sum().item()  # .item()获得张量中的值
        accuracy_4.append(correct_4 / total_4)
    print('Model_4 Finished')

    plt.figure()
    plt.plot(num_epochs, accuracy_1, label='Model_1(16)')
    plt.plot(num_epochs, accuracy_2, label='Model_2(32)')
    plt.plot(num_epochs, accuracy_3, label='Model_3(48)')
    plt.plot(num_epochs, accuracy_4, label='Model_4(64)')
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
