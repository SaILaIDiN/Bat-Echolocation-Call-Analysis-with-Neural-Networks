""" This script loads and evaluates pretrained Autoencoders on various datasets. """

import torch
from optimization_core.models import AutoencoderClassic2Select, AutoencoderClassicSelectSpec, ConvAutoencoder
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from optimization_core.metrics import species_to_genus_translator
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_classic_AE(model_weights_path, dataset, batch_size, n_batches, class_mode="binary", input_mode="MFCC",
                        custom_label=None, species_or_genus="species"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if input_mode == "MFCC":
        model = AutoencoderClassic2Select()
    elif input_mode == "LinSpec":
        model = AutoencoderClassicSelectSpec()
    else:
        print("No model chosen!")
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)
    input_latent_image_collector = []  # latent representation of input image from encoder
    text_label_collector = []  # label of an image in string format

    for index_batch in range(0, n_batches):
        batch = next(data_iter)
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
        labels = batch["label"]  # numpy array [2,]
        text_label = batch["text_label"]

        if species_or_genus == "genus":
            # Currently hardcoded for our case of Pipistrellus species
            # bat_class_list = ['Nyctaloid', 'Myotis', 'Psuper']
            bat_class_list = ['Nyctaloid', 'Myotis', 'Plecotus', 'Psuper']
            species_to_combine = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
            genus_name = 'Psuper'
            text_label = species_to_genus_translator(
                text_label, species_to_combine, genus_name)

        if input_mode == "MFCC":
            images_reshaped = images[0, 0].view(-1, 20 * 586)
        elif input_mode == "LinSpec":
            images_reshaped = images[0, 0].view(-1, 1025 * 586)
        else:
            print("Input image not refactored properly!")
        encoded_output, decoded_output = model(images_reshaped)

        encoded_output = encoded_output[-1]  # SelectiveSequential()

        encoded_output = encoded_output.cpu().detach().numpy()
        # print("Encoder output shape: ", encoded_output.shape)
        # print("Encoder output: ", encoded_output)

        input_latent_image_collector.append(encoded_output)
        if class_mode == "binary":
            if labels[0][0] == 0:
                text_label_collector.append("bat")
            elif labels[0][0] == 1:
                text_label_collector.append("noise")
        elif class_mode == "multi-class":
            text_label_collector.append(text_label)
        elif class_mode == "custom":
            print("Custom-Label-Mode entered!")
            text_label_collector.append(custom_label)
        else:
            print("Nothing added.")

    # # # Adjust format of data for clustering methods
    X_data = np.asarray(input_latent_image_collector).squeeze()  # shape (n_images, len(encoder_output))
    y_data = np.asarray(text_label_collector)  # shape (n_images, 1)

    feat_cols = ["pixel_"+str(i) for i in range(X_data.shape[1])]
    df = pd.DataFrame(X_data, columns=feat_cols)
    df['y'] = y_data
    # # #

    return df, feat_cols, input_latent_image_collector, text_label_collector


def evaluate_conv_AE(model_weights_path, dataset, batch_size, n_batches, class_mode="binary", input_mode="MFCC",
                     custom_label=None, species_or_genus="species"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(input_mode=input_mode)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)
    input_latent_image_collector = []  # latent representation of input image from encoder
    text_label_collector = []  # label of an image in string format

    for index_batch in range(0, n_batches):
        batch = next(data_iter)
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
        labels = batch["label"]  # numpy array [2,]
        text_label = batch["text_label"]

        if species_or_genus == "genus":
            # Currently hardcoded for our case of Pipistrellus species
            # bat_class_list = ['Nyctaloid', 'Myotis', 'Psuper']
            bat_class_list = ['Nyctaloid', 'Myotis', 'Plecotus', 'Psuper']
            species_to_combine = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
            genus_name = 'Psuper'
            text_label = species_to_genus_translator(
                text_label, species_to_combine, genus_name)

        min_val = torch.min(images)
        images_pos = torch.sub(images, min_val)
        encoded_output, decoded_output = model(images_pos)
        # encoded_output = encoded_output[2]  # SelectiveSequential()
        encoded_output = encoded_output.cpu().detach().numpy()
        # print("Encoder output shape: ", encoded_output.shape)
        # print("Encoder output: ", encoded_output)
        encoded_output = encoded_output.flatten()
        # print("Encoder output shape: ", encoded_output.shape)
        # print("Encoder output: ", encoded_output)

        input_latent_image_collector.append(encoded_output)
        if class_mode == "binary":
            if labels[0][0] == 0:
                text_label_collector.append("bat")
            elif labels[0][0] == 1:
                text_label_collector.append("noise")
        elif class_mode == "multi-class":
            text_label_collector.append(text_label)
        elif class_mode == "custom":
            print("Custom-Label-Mode entered!")
            text_label_collector.append(custom_label)
        else:
            print("Nothing added.")

    # # # Adjust format of data for clustering methods
    X_data = np.asarray(input_latent_image_collector).squeeze()  # shape (n_images, len(encoder_output))
    y_data = np.asarray(text_label_collector)  # shape (n_images, 1)

    feat_cols = ["pixel_"+str(i) for i in range(X_data.shape[1])]
    df = pd.DataFrame(X_data, columns=feat_cols)
    df['y'] = y_data
    # # #

    return df, feat_cols, input_latent_image_collector, text_label_collector


def create_scatter_plot(df, output_path, image_prefix="undefined", n_classes=2, alpha=0.5):

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pixel_0", y="pixel_1",
        hue="y",
        palette=sns.color_palette("husl", n_classes),
        data=df,
        legend="full",
        alpha=alpha
    )
    plt.savefig(os.path.join(output_path, f"{image_prefix}_AE_scatterplot.png"))
    plt.show()
