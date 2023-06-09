from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.utils.data
import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('device is %s' % device)

    # 数据处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 导入训练数据
    data = MNIST(root='./data_set', train=True,
                       transform=transform, download=False)
    data_loader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=4)


    for i, (image, labels) in enumerate(data_loader):
        if i >= 3:
            break
        print(i)
        print(image.shape)
        print(labels.shape)
        image = image.permute([0, 2, 3, 1])

        plt.figure()
        for j in range(12):
            plt.subplot(3, 4, j+1)
            plt.imshow(image[j], cmap='gray')
            plt.title("Ground Truth: {}".format(labels[j]))

        plt.savefig('./datashow{}'.format(i))
        # plt.show()

    # example = enumerate(data_loader)
    # batch, (example_data, example_labels) = next(example)
    # example_data = example_data.permute([0, 2, 3, 1])
    # for i in range(12):
    #     plt.subplot(3, 4, i+1)
    #     plt.imshow(example_data[i], cmap='gray')
    #     plt.title("Ground Truth: {}".format(example_labels[i]))
    # plt.show()

if __name__ == '__main__':
    main()
