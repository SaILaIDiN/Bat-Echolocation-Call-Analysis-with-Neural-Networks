""" Check noise patterns of linear spectrograms """
import os
import pandas as pd
from data.preprocessing.data_separator import clean_noise_recordings_table, noise_file_name_txt
from data.dataset import NoiseDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa.display as ld
import numpy as np


def create_noise_sets_by_height_and_orientation(dir_main, csv_path_main, dir_data_txts, year, attributes, size,
                                                spec_type):
    """ """
    subset_collector = []
    for attr_oh in attributes:
        if year == 2019:
            csv_path = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(arg=csv_path_main, attr=attr_oh)
        elif year == 2020:
            csv_path = r"{arg}\evaluation_{attr}m_04_20-11_20.csv".format(arg=csv_path_main, attr=attr_oh)

        # Create a table from a csv
        recordings_table = pd.read_csv(csv_path)
        # Create a table of noise only from the csv
        noise_table = clean_noise_recordings_table(recordings_table)
        # Check the alignment of labeled noise filenames and existent labels/wav-files
        dir_noise_wl = r"{dir_main}\noise\{attr}\Labels".format(dir_main=dir_main, attr=attr_oh)
        noise_size = noise_file_name_txt(noise_table, dir_data_txts, year, dir_noise_wl, attr_oh, size)
        dir_noise_txt = r"{dir_data_txts}\data_{year}\{attr}\Noise_file_names{noise_size}.txt".format(
            dir_data_txts=dir_data_txts, year=year, attr=attr_oh, noise_size=noise_size)

        noise_subset_tmp = NoiseDataset(dir_main, dir_noise_txt, attr_oh=attr_oh, spec_type=spec_type)
        subset_collector.append(noise_subset_tmp)

    return subset_collector


def visualize_noise_pattern(dataset, batch_size, n_batches, output_path, image_prefix="undefined", plot_or_save="plot",
                            n_images=None):
    """ n_images is the effective number of saved figure that splits the number of images into uniform
         partitions. n_batches should be a multiple of n_images.
        batch_size should stay at 1.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)
    input_image_collector = []  # input image
    image_path_collector = []  # input image path

    for index_batch in range(0, n_batches):
        batch = next(data_iter)
        images = batch["image"].cpu().detach().numpy()
        image_path = batch["image_path"]

        input_image_collector.append(images)
        image_path_collector.append(image_path)

    with open(os.path.join(output_path, f"noise_file_names_from_visuals_{image_prefix}.txt"), "w") as f:
        for image_path in image_path_collector:
            f.write(f"{image_path}\n")

    if n_images and n_batches % n_images == 0:
        fig, axes = plt.subplots(nrows=1, ncols=batch_size*n_images, sharex="all", figsize=(8*6.4, 8*4.8))

        split_pos_old = 0
        for split_pos in range(0, n_batches, n_images):
            for index, images_row in enumerate(zip(input_image_collector[split_pos_old:split_pos], axes)):
                images, row = images_row
                ax = row
                for i, img in enumerate(images):
                    print(img.shape)
                    fontdict = {'fontsize': 40}
                    img = ld.specshow(img, ax=ax, x_axis=None, y_axis=None)  # None for axis needed to customize labels!
                    ax.set_yticks(range(0, 1025, 103))
                    ax.set_yticklabels([str(i) for i in range(0, 150000, 15000)], fontdict=fontdict)
                    ax.set_xticks(range(0, 586, 100))
                    ax.set_xticklabels([str(i) for i in range(0, 586, 100)], fontdict=fontdict)
                    ax.set_xlabel("time step (~0.6ms)", fontdict=fontdict)
                    ax.set_ylabel("Hz", fontdict=fontdict)

                if index == n_images-1:
                    cbar = fig.colorbar(img, ax=ax, format="%+2.f dB")
                    cbar.ax.tick_params(labelsize=40)

                ax.get_xaxis().set_visible(True)
                ax.get_yaxis().set_visible(True)

                if plot_or_save == "save":
                    plt.savefig(os.path.join(output_path, f"{image_prefix}_noise_pattern_sample_{split_pos}.png"))
                    cbar.remove() if index == n_images-1 else None
                else:
                    plt.show()
                    cbar.remove() if index == n_images-1 else None
            split_pos_old = split_pos
        plt.clf()
    else:
        print("Warning: \"n_batches\" is not a multiple of \"n_images\"!")


def visualize_single_sound_pattern(sound_path, output_path, image_prefix="undefined", plot_or_save="plot",
                                   spec_type="LinSpec"):
    sound_spec = np.load(sound_path)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8*6.4, 8*4.8))
    fontdict = {'fontsize': 150}
    img = ld.specshow(sound_spec, ax=ax, x_axis=None, y_axis=None)  # None for axis needed to customize labels!
    if spec_type == "LinSpec":
        ax.set_yticks(range(0, 1025, 206))
        ax.set_yticklabels([str(int(i/1000)) for i in range(0, 150000, 30000)], fontdict=fontdict)
        ax.set_xticks([0, 293, 585])
        ax.set_xticklabels(["0", "0.5", "1"], fontdict=fontdict)
        ax.set_xlabel("sec", fontdict=fontdict)
        ax.set_ylabel("KHz", fontdict=fontdict)
    else:  # MFCC
        ax.set_yticks(range(0, 20, 4))
        ax.set_yticklabels([str(int(i/1000)) for i in range(0, 150000, 30000)], fontdict=fontdict)
        ax.set_xticks([0, 293, 585])
        ax.set_xticklabels(["0", "0.5", "1"], fontdict=fontdict)
        ax.set_xlabel("sec", fontdict=fontdict)
        ax.set_ylabel("KHz", fontdict=fontdict)

    divider = make_axes_locatable(ax)
    cbar_axes = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cbar_axes, format="%+2.f dB")
    cbar.ax.tick_params(labelsize=112)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    if plot_or_save == "save":
        plt.savefig(os.path.join(output_path, f"{image_prefix}_pattern_sample.png"))
    else:
        plt.show()
    return


def create_noise_spec_plots(subset_collector, batch_size, n_batches, attributes, output_path,
                            image_prefix="undefined", plot_or_save="plot", n_images=None):
    """ """
    for subset, attr_oh in zip(subset_collector, attributes):
        visualize_noise_pattern(subset,  batch_size, n_batches, output_path, image_prefix=image_prefix + attr_oh,
                                plot_or_save=plot_or_save, n_images=n_images)
    return


if __name__ == "__main__":
    from data.path_provider import provide_paths, dir_results

    # # # Dataset Year 2019
    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
        local_or_remote="remote", year=2019)
    year = 2019
    size = 1000
    spec_type = "LinSpec"
    attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
    attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
    attribute_2019 = attribute_W_2019 + attribute_E_2019

    noise_subsets_2019 = create_noise_sets_by_height_and_orientation(dir_main, csv_path_main, dir_data_txts, year,
                                                                     attribute_2019, size, spec_type)

    output_path = r"{arg}\noise_patterns\New_Large_Font".format(arg=dir_results)
    create_noise_spec_plots(noise_subsets_2019, batch_size=1, n_batches=32, attributes=attribute_2019,
                            output_path=output_path, image_prefix="year_2019_", plot_or_save="save", n_images=4)

    # # Single visualizations
    noise_path = '{arg}/noise/E_05/LinSpecs/WMM_NW_E_05_2019-05-20_21-01-46_0006874.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path, output_path, image_prefix="fifth_image_noise", plot_or_save="save")

    # E 2019 05m
    noise_path_1 = '{arg}/noise/E_05/LinSpecs/WMM_NW_E_05_2019-05-20_17-37-49_0006813.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_19_05m_sample_1_noise", plot_or_save="save")

    noise_path_2 = '{arg}/noise/E_05/LinSpecs/WMM_NW_E_05_2019-06-26_15-08-52_0009425.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_2, output_path, image_prefix="E_19_05m_sample_2_noise", plot_or_save="save")

    noise_path_3 = '{arg}/noise/E_05/LinSpecs/WMM_NW_E_05_2019-06-01_09-58-46_0007497.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_3, output_path, image_prefix="E_19_05m_sample_3_noise", plot_or_save="save")

    # E 2019 33m
    noise_path_1 = '{arg}/noise/E_33/LinSpecs/WMM_NW_E_33_2019-06-10_16-04-20_0003841.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_19_33m_sample_1_noise", plot_or_save="save")

    # E 2019 65m
    noise_path_1 = '{arg}/noise/E_65/LinSpecs/WMM_NW_E_65_2019-07-05_17-27-34_0003841.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_19_65m_sample_1_noise", plot_or_save="save")

    # E 2019 95m
    noise_path_1 = '{arg}/noise/E_95/LinSpecs/WMM_NW_E_95_2019-05-25_06-55-01_0004062.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_19_95m_sample_1_noise", plot_or_save="save")

    # W 2019 33m
    noise_path_1 = '{arg}/noise/W_33/LinSpecs/WMM_NW_W_33_2019-05-14_23-02-05_0002772.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="W_19_33m_sample_1_noise", plot_or_save="save")

    # W 2019 95m
    noise_path_1 = '{arg}/noise/W_95/LinSpecs/WMM_NW_W_95_2019-07-06_22-17-53_0002417.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="W_19_95m_sample_1_noise", plot_or_save="save")

    # E 2019 65m (Nnoc)
    bats_path_1 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-08-03_19-56-09_0004818.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_1, output_path, image_prefix="E_19_65m_sample_1_Nnoc", plot_or_save="save")

    bats_path_2 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-08-09_00-16-15_0004950.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_2, output_path, image_prefix="E_19_65m_sample_2_Nnoc", plot_or_save="save")

    bats_path_3 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-08-10_03-34-25_0005015.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_3, output_path, image_prefix="E_19_65m_sample_3_Nnoc", plot_or_save="save")

    # E 2019 65m (Nyctaloid)
    bats_path_2 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-08-05_19-55-02_0004855.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_2, output_path, image_prefix="E_19_65m_sample_2_bats", plot_or_save="save")

    # E 2019 65m (Myotis)
    bats_path_1 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_65_2019-06-19_22-16-23_0003228.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_1, output_path, image_prefix="E_19_65m_sample_1_Myotis", plot_or_save="save")

    bats_path_2 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-08-25_20-09-09_0005539.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_2, output_path, image_prefix="E_19_65m_sample_2_Myotis", plot_or_save="save")

    bats_path_3 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-09-02_22-45-54_0006016.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_3, output_path, image_prefix="E_19_65m_sample_3_Myotis", plot_or_save="save")

    # E 2019 65m (Ppip)
    bats_path_1 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-09-02_23-03-45_0006019.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_1, output_path, image_prefix="E_19_65m_sample_1_Ppip", plot_or_save="save")

    bats_path_2 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-09-13_18-58-56_0006196.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_2, output_path, image_prefix="E_19_65m_sample_2_Ppip", plot_or_save="save")

    bats_path_3 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-09-14_23-22-26_0006517.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_3, output_path, image_prefix="E_19_65m_sample_3_Ppip", plot_or_save="save")

    bats_path_4 = "{arg}/bat_calls/E_65/LinSpecs_1/WMM_NW_E_652019-10-25_19-47-03_0007807.wav_Sec_1.npy".format(arg=dir_main)
    visualize_single_sound_pattern(bats_path_4, output_path, image_prefix="E_19_65m_sample_4_Ppip", plot_or_save="save")

    # # # Dataset Year 2020
    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
        local_or_remote="remote", year=2020)
    year = 2020
    size = 1000
    spec_type = "LinSpec"
    attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
    attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
    attribute_2020 = attribute_W_2020 + attribute_E_2020

    noise_subsets_2020 = create_noise_sets_by_height_and_orientation(dir_main, csv_path_main, dir_data_txts, year,
                                                                     attribute_2020, size, spec_type)

    output_path = r"{arg}\noise_patterns\New_Large_Font".format(arg=dir_results)
    create_noise_spec_plots(noise_subsets_2020, batch_size=1, n_batches=32, attributes=attribute_2020,
                            output_path=output_path, image_prefix="year_2020_", plot_or_save="save", n_images=4)

    # # Single visualizations
    # E 2020 10m
    noise_path_1 = '{arg}/noise/E_10/LinSpecs/BATmode2_WMM_NE_2020-09-13_20-56-51_0170263.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_20_10m_sample_1_noise", plot_or_save="save")

    # E 2020 65m
    noise_path_1 = '{arg}/noise/E_65/LinSpecs/BATmode2_WMM_NE_2020-08-18_21-12-06_0002554.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="E_20_65m_sample_1_noise", plot_or_save="save")

    # W 2020 10m
    noise_path_1 = '{arg}/noise/W_10/LinSpecs/BATmode_WMM_NW_2020-04-06_17-06-27_0000136.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="W_20_10m_sample_1_noise", plot_or_save="save")

    # W 2020 95m
    noise_path_1 = '{arg}/noise/W_95/LinSpecs/BATmode_WMM_NW_2020-04-08_20-48-35_0000338.wav.npy'.format(arg=dir_main)
    visualize_single_sound_pattern(noise_path_1, output_path, image_prefix="W_20_95m_sample_1_noise", plot_or_save="save")
