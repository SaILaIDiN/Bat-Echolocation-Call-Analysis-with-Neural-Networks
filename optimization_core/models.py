import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSequential(nn.Module):
    """ Variation of Sequential class to save intermediate data flow within a Sequential """

    def __init__(self, to_select, modules_tuple):
        super().__init__()
        for idx, module in enumerate(modules_tuple):
            self.add_module(str(idx), module)
        self._to_select = to_select

    def forward(self, x):
        list = []
        for name, module in self._modules.items():

            x = module(x)
            if name in self._to_select:
                list.append(x)
        return list


# # Note: Currently the old version with 2x2 convolutions instead of 3x3 in the first two layers
class YPNet(nn.Module):
    def __init__(self, n_classes=2, dropout=0.5, n_hidden_out=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(4608, n_hidden_out)
        self.dropout1 = nn.Dropout(p=dropout, inplace=True)
        self.fc2 = nn.Linear(n_hidden_out, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, ceil_mode=True)
        x = self.batchnorm3(x)
        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


# # Note: Currently updated version with 3x3 convolutions instead of 2x2 in the first two layers
class YPNetNew(nn.Module):
    def __init__(self, n_classes=2, dropout=0.5, n_hidden_out=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='valid')
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='valid')
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding='valid')
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(2304, n_hidden_out)
        self.dropout1 = nn.Dropout(p=dropout, inplace=True)
        self.fc2 = nn.Linear(n_hidden_out, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0, ceil_mode=True)
        x = self.batchnorm3(x)
        x = x.view(-1, 2304)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class AutoencoderClassicSelect(nn.Module):
    """ For MFCCs. Larger network. """
    def __init__(self, activation="ReLU"):
        super().__init__()
        self.activation = activation
        if self.activation == "ReLU":
            activation_function = nn.ReLU()
        elif self.activation == "Tanh":
            activation_function = nn.Tanh()
        else:
            activation_function = nn.ReLU()

        to_select_in = ["0", "1", "2", "10"]
        to_select_out = ["8", "9", "10"]  # order has no effect, because forward in SelectiveSequential() decides it
        self.encoder = SelectiveSequential(
            to_select_in,
            (nn.Linear(20 * 586, 4096),
             activation_function,
             nn.Linear(4096, 2048),
             activation_function,
             nn.Linear(2048, 512),
             activation_function,
             nn.Linear(512, 128),
             activation_function,
             nn.Linear(128, 64),
             activation_function,
             nn.Linear(64, 32))
        )
        self.decoder = SelectiveSequential(
            to_select_out,
            (nn.Linear(32, 64),
             activation_function,
             nn.Linear(64, 128),
             activation_function,
             nn.Linear(128, 512),
             activation_function,
             nn.Linear(512, 2048),
             activation_function,
             nn.Linear(2048, 4096),
             activation_function,
             nn.Linear(4096, 20 * 586))
            # nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print("Length", len(encoded))
        decoded = self.decoder(encoded[-1])
        # print("Length (dec): ", len(decoded))
        return encoded, decoded


class AutoencoderClassic2Select(nn.Module):
    """ For MFCCs. Smaller network. """
    def __init__(self, activation="ReLU"):
        super().__init__()
        self.activation = activation
        if self.activation == "ReLU":
            activation_function = nn.ReLU()
        elif self.activation == "Tanh":
            activation_function = nn.Tanh()
        else:
            activation_function = nn.ReLU()

        to_select_in = ["0", "1", "2", "8"]
        to_select_out = ["6", "7", "8"]  # order has no effect, because forward in SelectiveSequential() decides it
        self.encoder = SelectiveSequential(
            to_select_in,
            (nn.Linear(20 * 586, 512),
             activation_function,
             nn.Linear(512, 256),
             activation_function,
             nn.Linear(256, 128),
             activation_function,
             nn.Linear(128, 64),
             activation_function,
             nn.Linear(64, 32))
        )
        self.decoder = SelectiveSequential(
            to_select_out,
            (nn.Linear(32, 64),
             activation_function,
             nn.Linear(64, 128),
             activation_function,
             nn.Linear(128, 256),
             activation_function,
             nn.Linear(256, 512),
             activation_function,
             nn.Linear(512, 20 * 586))
            # nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print("Length", len(encoded))
        decoded = self.decoder(encoded[-1])
        # print("Length (dec): ", len(decoded))
        return encoded, decoded


class AutoencoderClassicSelectSpec(nn.Module):
    """ This Autoencoder is to learn linear spectrograms with fully connected layers.
        Without any preceeding dimensional reduction, the size of the network is strongly restricted by the high
         resolution of the linear spectrograms, which explains the drastic reduction in the first layer.
        The number of layers is not a big deal as long as the first layer performs a proper reduction of vector size.
    """
    def __init__(self, activation="ReLU"):
        super().__init__()
        self.activation = activation
        if self.activation == "ReLU":
            activation_function = nn.ReLU()
        elif self.activation == "Tanh":
            activation_function = nn.Tanh()
        else:
            activation_function = nn.ReLU()

        to_select_in = ["0", "1", "2"]  # this is the actual name of the layer
        to_select_out = ["0", "1", "2"]  # order has no effect, because forward in SelectiveSequential() decides it
        self.encoder = SelectiveSequential(
            to_select_in,
            (nn.Linear(1025 * 586, 512),
             activation_function,
             nn.Linear(512, 128))
        )
        self.decoder = SelectiveSequential(
            to_select_out,
            (nn.Linear(128, 512),
             activation_function,
             nn.Linear(512, 1025 * 586))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print("Length", len(encoded))
        decoded = self.decoder(encoded[-1])
        # print("Length (dec): ", len(decoded))
        return encoded, decoded


class ConvAutoencoder(nn.Module):
    def __init__(self, input_mode="MFCC"):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, padding=0)
        self.pool2 = nn.MaxPool2d(2, padding=1)

        # Decoder
        self.input_mode = input_mode
        self.up1 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(8)
        self.up2 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(8)
        self.up3 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0)
        self.batchnorm6 = nn.BatchNorm2d(16)
        self.conv = nn.Conv2d(16, 1, 4, stride=1, padding=2)

    def encoder(self, image):
        conv1 = self.conv1(image)
        relu1 = F.relu(conv1)  # 28x28x16  |  1025x586x16
        pool1 = self.pool1(relu1)  # 14x14x16  | 513x293x16
        pool1 = self.batchnorm1(pool1)
        conv2 = self.conv2(pool1)  # 14x14x8  | 513x293x8
        relu2 = F.relu(conv2)
        pool2 = self.pool1(relu2)  # 7x7x8  | 257x147x8
        pool2 = self.batchnorm2(pool2)
        conv3 = self.conv3(pool2)  # 7x7x8  | 257x147x8
        relu3 = F.relu(conv3)
        pool3 = self.pool2(relu3)  # 4x4x8  | 129x74x8
        pool3 = self.batchnorm3(pool3)
        # print("Pool3 shape: ", pool3.shape)
        return pool3

    def decoder(self, encoding):
        up1 = self.up1(encoding)
        up_relu1 = F.relu(up1)
        up_relu1 = self.batchnorm4(up_relu1)
        up2 = self.up2(up_relu1)
        up_relu2 = F.relu(up2)
        up_relu2 = self.batchnorm5(up_relu2)
        up3 = self.up3(up_relu2)
        up_relu3 = F.relu(up3)
        up_relu3 = self.batchnorm6(up_relu3)
        output = self.conv(up_relu3)
        # print("output shape: ", output.shape)
        if self.input_mode == "MFCC":
            output = output[0, 0, 0:20, 1:587]  # for MFCCs
        elif self.input_mode == "LinSpec":
            output = output[0, 0, 1:1026, 1:587]  # for Linear Spectrograms
        else:
            print("Output is not refactored properly!")
        # print("output shape: ", output.shape)
        return output

    def forward(self, image):
        encoding = self.encoder(image)
        decoding = self.decoder(encoding)
        return encoding, decoding


if __name__ == "__main__":
    a = torch.rand((1, 1, 20, 586), dtype=torch.float32)
    net1 = YPNet(n_classes=10)
    output1 = net1(a)
    print("output1:", output1)
