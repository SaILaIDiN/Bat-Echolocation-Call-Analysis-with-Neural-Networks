import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from optimization_core.metrics import standard_metrics
from optimization_core.models import AutoencoderClassicSelect, AutoencoderClassic2Select, \
    AutoencoderClassicSelectSpec, ConvAutoencoder

from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import librosa.display as ld


def train_classic_AE(dataset, output_path_model=None, input_mode="MFCC", n_epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if input_mode == "MFCC":
        model = AutoencoderClassic2Select()
        # model = AutoencoderClassicSelect()
    elif input_mode == "LinSpec":
        model = AutoencoderClassicSelectSpec()
    else:
        print("No model chosen!")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    print("Length dataloader: ", len(train_loader))

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0

        counter = 0

        for batch in train_loader:
            images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
            if images.shape[3] == 586:
                counter += 1
                # print("Counter: ", counter)
                optimizer.zero_grad()

                if input_mode == "MFCC":
                    images_reshaped = images[0, 0].view(-1, 20*586)
                elif input_mode == "LinSpec":
                    images_reshaped = images[0, 0].view(-1, 1025*586)
                else:
                    print("Input image not refactored properly!")
                # print("images (reshaped): ", images_reshaped.shape)
                encoded, decoded = model(images_reshaped)
                # print("decoded: ", decoded.shape)
                # print("images: ", images.shape)
                loss = criterion(decoded[-1], images_reshaped)
                # loss = criterion(decoded, images_reshaped)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()*images.size(0)
            else:
                print("Wrong input shape!")

        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch, train_loss))
        if output_path_model:
            torch.save(model.state_dict(), output_path_model + "_epoch_" + str(epoch) + ".pth")


def train_conv_AE(dataset, output_path_model=None, input_mode="MFCC", n_epochs=5):
    """ NOTE: Currently not working due to mismatch of intermediate dimensions in the architecture """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder(input_mode=input_mode)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    print("Length dataloader: ", len(train_loader))

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0

        counter = 0

        for batch in train_loader:
            images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
            if images.shape[3] == 586:
                counter += 1
                # print("Counter: ", counter)
                optimizer.zero_grad()

                # Conv-AE requires positive values due to the activation function
                # thus add the maximum of the tensor to all tensor entries
                # currently works for batch_size=1
                min_val = torch.min(images)
                images_pos = torch.sub(images, min_val)
                encoding, outputs = model(images_pos)
                # print("outputs: ", outputs)
                # print("images: ", images_pos)
                loss = criterion(outputs, images_pos)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()*images.size(0)
            else:
                print("Wrong input shape!")

        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch, train_loss))
        if output_path_model:
            torch.save(model.state_dict(), output_path_model + "_epoch_" + str(epoch) + ".pth")


def visualize_results_classic_AE(model_weights_path, dataset, batch_size, n_batches, input_mode="MFCC"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if input_mode == "MFCC":
        model = AutoencoderClassic2Select()
        # model = AutoencoderClassicSelect()
    elif input_mode == "LinSpec":
        model = AutoencoderClassicSelectSpec()
    else:
        print("No model chosen!")
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)
    input_image_collector = []  # encoder input image
    output_image_collector = []  # decoded output image
    for index_batch in range(0, n_batches):
        batch = next(data_iter)
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)

        if input_mode == "MFCC":
            images_reshaped = images[0, 0].view(-1, 20 * 586)
        elif input_mode == "LinSpec":
            images_reshaped = images[0, 0].view(-1, 1025 * 586)
        else:
            print("Input image not refactored properly!")
        encoded_output, decoded_output = model(images_reshaped)

        # # # for SelectiveSequential AE-Version
        decoded_output = decoded_output[-1]
        # # #

        if input_mode == "MFCC":
            output = decoded_output.cpu().detach().numpy().reshape([20, 586])
        elif input_mode == "LinSpec":
            output = decoded_output.cpu().detach().numpy().reshape([1025, 586])
        else:
            print("Output image not refactored properly!")

        images = images.cpu().detach().numpy()

        output_image_collector.append(output)
        input_image_collector.append(images)

    fig, axes = plt.subplots(nrows=2, ncols=n_batches*batch_size, sharex="all", sharey="all")
    for images, row in zip([input_image_collector, output_image_collector], axes):
        for img, ax in zip(images, row):
            # ax.imshow(np.squeeze(img), cmap='gray')
            img = ld.specshow(np.squeeze(img), ax=ax)
            fig.colorbar(img, ax=ax)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


def visualize_results_conv_AE(model_weights_path, dataset, batch_size, n_batches, input_mode="MFCC"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(input_mode=input_mode)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)
    input_image_collector = []  # encoder input image
    output_image_collector = []  # decoded output image
    for index_batch in range(0, n_batches):
        batch = next(data_iter)
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
        # Conv-AE requires positive values due to the activation function
        # thus add the maximum of the tensor to all tensor entries
        # currently works for batch_size=1
        min_val = torch.min(images)
        images_pos = torch.sub(images, min_val)
        encoding, outputs = model(images_pos)

        # max_val = torch.max(outputs)
        outputs_neg = torch.add(outputs, min_val)

        outputs = outputs_neg.cpu().detach().numpy()
        images = images.squeeze().cpu().detach().numpy()

        output_image_collector.append(outputs)
        input_image_collector.append(images)

    fig, axes = plt.subplots(nrows=2, ncols=n_batches*batch_size, sharex="all", sharey="all")
    for images, row in zip([input_image_collector, output_image_collector], axes):
        for img, ax in zip(images, row):
            img = ld.specshow(np.squeeze(img), ax=ax)
            fig.colorbar(img, ax=ax)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    from data.path_provider import dir_results, dir_results_remote
    from unsupervised_analysis.datasets_for_clustering import \
        joined_noise_W_2019_lin, joined_noise_E_2019_lin, joined_noise_W_2020_lin, joined_noise_E_2020_lin, \
        dataset_joined_nyctaloid, dataset_joined_myotis, dataset_joined_plecotus, dataset_joined_psuper, \
        dataset_joined_noise

    # # Convolutional Autoencoder (FINAL)
    # Noise (LinSpec, W_2019)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_W_2019".format(arg=dir_results_remote)
    train_conv_AE(joined_noise_W_2019_lin[0], output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)
    # Noise (LinSpec, E_2019)
    model_weights_path = r"{arg}}\pth_files_AE\AE_conv\Noise\test_lin_E_2019".format(arg=dir_results_remote)
    train_conv_AE(joined_noise_E_2019_lin[0], output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)
    # Noise (LinSpec, W_2020)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_W_2020".format(arg=dir_results_remote)
    train_conv_AE(joined_noise_W_2020_lin[0], output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)
    # Noise (LinSpec, E_2020)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_E_2020".format(arg=dir_results_remote)
    train_conv_AE(joined_noise_E_2020_lin[0], output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)
    # Noise (LinSpec, All)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_All".format(arg=dir_results_remote)
    dataset_full = ConcatDataset([joined_noise_W_2019_lin[0], joined_noise_E_2019_lin[0], joined_noise_W_2020_lin[0],
                                  joined_noise_E_2020_lin[0]])
    train_conv_AE(dataset_full, output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)

    # Genus-Noise (LinSpec, All)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Genus-Noise\test_lin_GN_All".format(arg=dir_results_remote)
    list_dataset_All = [dataset_joined_nyctaloid, dataset_joined_myotis, dataset_joined_plecotus,
                        dataset_joined_psuper, dataset_joined_noise]
    dataset_full = ConcatDataset(list_dataset_All)
    train_conv_AE(dataset_full, output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)

    # Genus (LinSpec, All)
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Genus\test_lin_G_All".format(arg=dir_results_remote)
    list_dataset_All = [dataset_joined_nyctaloid, dataset_joined_myotis, dataset_joined_plecotus, dataset_joined_psuper]
    dataset_full = ConcatDataset(list_dataset_All)
    train_conv_AE(dataset_full, output_path_model=model_weights_path, input_mode="LinSpec", n_epochs=10)
