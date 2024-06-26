from torch.utils.data import DataLoader, ConcatDataset
import torch
from optimization_core.models import ConvAutoencoder
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(original, reconstructed):
    # Ensure the images have the same data type for accurate calculations
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    # Calculate the SSIM
    ssim_value = ssim(original, reconstructed, data_range=255)

    return ssim_value


def calculate_psnr(original, reconstructed):
    # Ensure the images have the same data type for accurate calculations
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    # Calculate the mean squared error (MSE)
    mse = np.mean((original - reconstructed) ** 2)

    # Calculate the maximum possible pixel value (assuming pixel values range from 0 to 255)
    max_pixel_value = 255.0

    # Calculate the PSNR using the formula: PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

    return psnr


def image_quality_conv_AE(model_weights_path, dataset, batch_size, n_batches, input_mode="MFCC"):
    """ Image quality of Autoencoder. """
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

    psnr_list = []
    for input, output in zip(input_image_collector, output_image_collector):
        psnr = calculate_psnr(input, output)
        psnr_list.append(psnr)
    psnr_avg = sum(psnr_list)/len(psnr_list)
    print("PSNR List: ", psnr_list)
    print("PSNR Average: ", psnr_avg)

    ssim_list = []
    for input, output in zip(input_image_collector, output_image_collector):
        ssim = calculate_ssim(input, output)
        ssim_list.append(ssim)
    ssim_avg = sum(ssim_list)/len(ssim_list)
    print("SSIM List: ", ssim_list)
    print("SSIM Average: ", ssim_avg)

    return


if __name__ == "__main__":
    from unsupervised_analysis.datasets_for_clustering import joined_noise_W_05_2019_lin, dataset_joined_nyctaloid, \
        dataset_joined_myotis, dataset_joined_plecotus, dataset_joined_psuper, dataset_joined_noise
    from data.path_provider import dir_results, dir_results_remote

    model_weights_path = r"{arg}\AE_conv\Genus-Noise\test_lin_GN_All_epoch_10.pth".format(arg=dir_results)
    dataset = dataset_joined_nyctaloid
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")
    dataset = dataset_joined_myotis
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")
    dataset = dataset_joined_plecotus
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")
    dataset = dataset_joined_psuper
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")
    dataset = dataset_joined_noise
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")

    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_W_2019_epoch_8.pth".format(arg=dir_results)
    dataset = joined_noise_W_05_2019_lin[0]
    image_quality_conv_AE(model_weights_path, dataset, batch_size=1, n_batches=len(dataset), input_mode="LinSpec")
