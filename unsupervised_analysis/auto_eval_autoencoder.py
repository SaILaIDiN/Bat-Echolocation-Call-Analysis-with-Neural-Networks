import pandas as pd
from unsupervised_analysis.cluster_analysis_umap import create_UMAP_plot
from unsupervised_analysis.eval_autoencoder import evaluate_classic_AE, evaluate_conv_AE


def cluster_UMAP_auto_bat_noise_post_AE(model_weights_path, output_path, dataset, list_n_neighbors, list_min_dist,
                                        network_type="ClassicAE", image_post_prefix="", input_mode="MFCC"):

    if network_type == "ClassicAE":
        df, feat_cols, _, _ = evaluate_classic_AE(model_weights_path, dataset[0], batch_size=1,
                                                  n_batches=len(dataset[0]), class_mode="binary",
                                                  input_mode=input_mode)
    elif network_type == "ConvAE":
        df, feat_cols, _, _ = evaluate_conv_AE(model_weights_path, dataset, batch_size=1,
                                               n_batches=len(dataset), class_mode="binary",
                                               input_mode=input_mode)
    else:
        print("Network type does not exist!")

    n_epochs = 100
    for n_neighbors in list_n_neighbors:
        for min_dist in list_min_dist:
            image_prefix = f"BC_neighbors_{n_neighbors}_min_dist_{min_dist}_n_epochs_{n_epochs}_{network_type}_" \
                           f"{image_post_prefix}"
            create_UMAP_plot(df, feat_cols, output_path, image_prefix=image_prefix, n_classes=2, alpha=0.5,
                             plot_or_save="save", n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, n_epochs=n_epochs)


def cluster_UMAP_auto_species_post_AE(model_weights_path, output_path, dataset, list_n_neighbors, list_min_dist,
                                      n_classes=3, network_type="ClassicAE", image_post_prefix="", input_mode="MFCC",
                                      class_mode="multi-class", species_or_genus="species"):

    if network_type == "ClassicAE":
        df, feat_cols, _, _ = evaluate_classic_AE(model_weights_path, dataset[0], batch_size=1,
                                                  n_batches=len(dataset[0]), class_mode=class_mode,
                                                  input_mode=input_mode, species_or_genus=species_or_genus)
    elif network_type == "ConvAE":
        df, feat_cols, _, _ = evaluate_conv_AE(model_weights_path, dataset, batch_size=1,
                                               n_batches=len(dataset), class_mode=class_mode,
                                               input_mode=input_mode, species_or_genus=species_or_genus)
    else:
        print("Network type does not exist!")

    n_epochs = 100
    for n_neighbors in list_n_neighbors:
        for min_dist in list_min_dist:
            image_prefix = f"MC_neighbors_{n_neighbors}_min_dist_{min_dist}_n_epochs_{n_epochs}_{network_type}_" \
                           f"{image_post_prefix}"
            create_UMAP_plot(df, feat_cols, output_path, image_prefix=image_prefix, n_classes=n_classes, alpha=0.5,
                             plot_or_save="save", n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, n_epochs=n_epochs)


def cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset, list_n_neighbors, list_min_dist,
                                    n_classes=3, network_type="ClassicAE", image_post_prefix="", input_mode="MFCC",
                                    class_mode="custom", list_text_labels=None):

    df_collector = []
    feat_cols_collector = []
    for dataset, text_label in zip(list_dataset, list_text_labels):
        if network_type == "ClassicAE":
            df, feat_cols, _, _ = evaluate_classic_AE(model_weights_path, dataset[0], batch_size=1,
                                                      n_batches=len(dataset[0]), class_mode=class_mode,
                                                      custom_label=text_label, input_mode=input_mode)
        elif network_type == "ConvAE":
            df, feat_cols, _, _ = evaluate_conv_AE(model_weights_path, dataset[0], batch_size=1,
                                                   n_batches=len(dataset[0]), class_mode=class_mode,
                                                   custom_label=text_label, input_mode=input_mode)
        else:
            print("Network type does not exist!")
        df_collector.append(df)
        feat_cols_collector.extend(feat_cols)

    df = pd.concat(df_collector)
    feat_cols = feat_cols_collector
    n_epochs = 100
    for n_neighbors in list_n_neighbors:
        for min_dist in list_min_dist:
            image_prefix = f"MC_neighbors_{n_neighbors}_min_dist_{min_dist}_n_epochs_{n_epochs}_{network_type}_" \
                           f"{image_post_prefix}"
            create_UMAP_plot(df, feat_cols, output_path, image_prefix=image_prefix, n_classes=n_classes, alpha=0.5,
                             plot_or_save="save", n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, n_epochs=n_epochs)


if __name__ == "__main__":
    from data.path_provider import dir_results, dir_results_remote
    from torch.utils.data import ConcatDataset
    from unsupervised_analysis.datasets_for_clustering import \
        joined_noise_W_2019_lin, joined_noise_E_2019_lin, joined_noise_W_2020_lin, joined_noise_E_2020_lin, \
        joined_noise_W_05_2019_lin, joined_noise_W_33_2019_lin, joined_noise_W_65_2019_lin, joined_noise_W_95_2019_lin,\
        joined_noise_E_05_2019_lin, joined_noise_E_33_2019_lin, joined_noise_E_65_2019_lin, joined_noise_E_95_2019_lin,\
        joined_noise_W_10_2020_lin, joined_noise_W_35_2020_lin, joined_noise_W_65_2020_lin, joined_noise_W_95_2020_lin,\
        joined_noise_E_10_2020_lin, joined_noise_E_35_2020_lin, joined_noise_E_65_2020_lin, joined_noise_E_95_2020_lin,\
        dataset_joined_nyctaloid, dataset_joined_myotis, dataset_joined_plecotus, dataset_joined_psuper, \
        dataset_joined_noise

    list_n_neighbors = [15, 30, 50, 100, 200]
    list_min_dist = [0.0, 0.1, 0.5, 0.8, 1.0]

    # # # Classic Autoencoder
    # # Binary
    # output_path = "results/cluster_analysis/E_2019_balanced/ClassicAE"
    # model_weights_path = r"{arg}\pth_files_AE\AE_classic\Binary\short_AE\test__epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_bat_noise_post_AE(model_weights_path, output_path, dataset_E_2019, list_n_neighbors,
    #                                     list_min_dist, network_type="ClassicAE")
    #
    # # Multi
    # output_path = "results/cluster_analysis/E_2019_balanced/ClassicAE_ppip"
    # model_weights_path = r"{arg}\pth_files_AE\AE_classic\Multi\short_AE_ppip\test__epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_species_post_AE(model_weights_path, output_path, bats_E_2019_ppip, list_n_neighbors,
    #                                   list_min_dist, n_classes=1, network_type="ClassicAE")
    # cluster_UMAP_auto_species_post_AE(model_weights_path, output_path, bats_E_2019_balanced_real, list_n_neighbors,
    #                                   list_min_dist)
    #
    # # # Convolutional Autoencoder
    # # Binary (LinSpec)
    # output_path = "results/cluster_analysis/W_2019_balanced/ConvAE"
    # model_weights_path = r"{arg}\pth_files_AE\AE_conv\Binary\test_lin_epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_bat_noise_post_AE(model_weights_path, output_path, joined_bats_and_noise_W_05_2019_lin,
    #                                     list_n_neighbors, list_min_dist, network_type="ConvAE", input_mode="LinSpec")
    #
    # # Multi (LinSpec)
    # output_path = "results/cluster_analysis/W_2019_balanced/ConvAE"
    # model_weights_path = r"{arg}\pth_files_AE\AE_conv\Multi\test_lin_epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_species_post_AE(model_weights_path, output_path, joined_bats_W_05_2019_lin, list_n_neighbors,
    #                                   list_min_dist, network_type="ConvAE", input_mode="LinSpec")
    #
    # # Binary (MFCC)
    # output_path = "results/cluster_analysis/W_2019_balanced/ConvAE"
    # model_weights_path = r"{arg}\pth_files_AE\AE_conv\Binary\test_mfcc_epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_bat_noise_post_AE(model_weights_path, output_path, joined_bats_and_noise_W_05_2019,
    #                                     list_n_neighbors, list_min_dist,
    #                                     network_type="ConvAE", image_post_prefix="MFCC", input_mode="MFCC")
    #
    # # Multi (MFCC)
    # output_path = "results/cluster_analysis/W_2019_balanced/ConvAE"
    # model_weights_path = r"{arg}\pth_files_AE\AE_conv\Multi\test_mfcc_epoch_7.pth".format(arg=dir_results)
    # cluster_UMAP_auto_species_post_AE(model_weights_path, output_path, joined_bats_W_05_2019,
    #                                   list_n_neighbors, list_min_dist,
    #                                   network_type="ConvAE", image_post_prefix="MFCC", input_mode="MFCC")

    print()
    # # # Convolutional Autoencoder (FINAL)
    # # Noise (LinSpec, W_2019)
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/W_2019"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_W_2019_epoch_8.pth".format(arg=dir_results)
    list_dataset_W_2019 = [joined_noise_W_05_2019_lin, joined_noise_W_33_2019_lin, joined_noise_W_65_2019_lin,
                           joined_noise_W_95_2019_lin]
    attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_W_2019,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_W_2019, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
    # # Noise (LinSpec, E_2019)
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/E_2019"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_E_2019_epoch_8.pth".format(arg=dir_results)
    list_dataset_E_2019 = [joined_noise_E_05_2019_lin, joined_noise_E_33_2019_lin, joined_noise_E_65_2019_lin,
                           joined_noise_E_95_2019_lin]
    attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_E_2019,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_E_2019, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
    # # Noise (LinSpec, W_2020)
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/W_2020"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_W_2020_epoch_7.pth".format(arg=dir_results)
    list_dataset_W_2020 = [joined_noise_W_10_2020_lin, joined_noise_W_35_2020_lin, joined_noise_W_65_2020_lin,
                           joined_noise_W_95_2020_lin]
    attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_W_2020,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_W_2020, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
    # # Noise (LinSpec, E_2020)
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/E_2020"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_E_2020_epoch_8.pth".format(arg=dir_results)
    list_dataset_E_2020 = [joined_noise_E_10_2020_lin, joined_noise_E_35_2020_lin, joined_noise_E_65_2020_lin,
                           joined_noise_E_95_2020_lin]
    attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_E_2020,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_E_2020, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Noise (LinSpec, ALL)
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/All_four"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Noise\test_lin_All_epoch_5.pth".format(arg=dir_results)
    list_dataset_All = [joined_noise_W_2019_lin, joined_noise_E_2019_lin, joined_noise_W_2020_lin,
                        joined_noise_E_2020_lin]
    attribute_All = ["W_2019", "E_2019", "W_2020", "E_2020"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_All,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_All, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Genus-Noise (LinSpec, All)
    output_path = "results/cluster_analysis/Final/ConvAE/Genus-Noise/All_four"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Genus-Noise\test_lin_GN_All_epoch_10.pth".format(arg=dir_results)

    list_dataset_All = [[dataset_joined_nyctaloid], [dataset_joined_myotis], [dataset_joined_plecotus],
                        [dataset_joined_psuper], [dataset_joined_noise]]
    attribute_All = ["Nyctaloid", "Myotis", "Plecotus", "Psuper", "Noise"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_All,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_All, n_classes=5,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Genus (LinSpec, All)
    output_path = "results/cluster_analysis/Final/ConvAE/Genus/All_four"
    model_weights_path = r"{arg}\pth_files_AE\AE_conv\Genus\test_lin_G_All_epoch_10.pth".format(arg=dir_results)

    list_dataset_All = [[dataset_joined_nyctaloid], [dataset_joined_myotis], [dataset_joined_plecotus],
                        [dataset_joined_psuper]]
    attribute_All = ["Nyctaloid", "Myotis", "Plecotus", "Psuper"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_All,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_All, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # # Single correction runs for aesthetics of plots on local system
    # # Noise (LinSpec, W_2019)
    list_n_neighbors = [200]
    list_min_dist = [0.5]
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/W_2019"
    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_W_2019_epoch_8.pth".format(arg=dir_results)
    list_dataset_W_2019 = [joined_noise_W_05_2019_lin, joined_noise_W_33_2019_lin, joined_noise_W_65_2019_lin,
                           joined_noise_W_95_2019_lin]
    attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
    attribute_W_2019_labels = ["W_10", "W_35", "W_65", "W_95"]

    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_W_2019, list_n_neighbors,
                                    list_min_dist, list_text_labels=attribute_W_2019_labels,  n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Noise (LinSpec, E_2019)
    list_n_neighbors = [15]
    list_min_dist = [0.8]
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/E_2019"
    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_E_2019_epoch_8.pth".format(arg=dir_results)
    list_dataset_E_2019 = [joined_noise_E_05_2019_lin, joined_noise_E_33_2019_lin, joined_noise_E_65_2019_lin,
                           joined_noise_E_95_2019_lin]
    attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
    attribute_E_2019_labels = ["E_10", "E_35", "E_65", "E_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_E_2019, list_n_neighbors,
                                    list_min_dist, list_text_labels=attribute_E_2019_labels, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
    # # Noise (LinSpec, W_2020)
    list_n_neighbors = [50]
    list_min_dist = [0.5]
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/W_2020"
    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_W_2020_epoch_7.pth".format(arg=dir_results)
    list_dataset_W_2020 = [joined_noise_W_10_2020_lin, joined_noise_W_35_2020_lin, joined_noise_W_65_2020_lin,
                           joined_noise_W_95_2020_lin]
    attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_W_2020,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_W_2020, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
    # # Noise (LinSpec, E_2020)
    list_n_neighbors = [200]
    list_min_dist = [1.0]
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/E_2020"
    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_E_2020_epoch_8.pth".format(arg=dir_results)
    list_dataset_E_2020 = [joined_noise_E_10_2020_lin, joined_noise_E_35_2020_lin, joined_noise_E_65_2020_lin,
                           joined_noise_E_95_2020_lin]
    attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_E_2020,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_E_2020, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Noise (LinSpec, ALL)
    list_n_neighbors = [50]
    list_min_dist = [0.8]
    output_path = "results/cluster_analysis/Final/ConvAE/Noise/All_four"
    model_weights_path = r"{arg}\AE_conv\Noise\test_lin_All_epoch_5.pth".format(arg=dir_results)
    list_dataset_All = [joined_noise_W_2019_lin, joined_noise_E_2019_lin, joined_noise_W_2020_lin,
                        joined_noise_E_2020_lin]
    attribute_All = ["W_2019", "E_2019", "W_2020", "E_2020"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_All,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_All, n_classes=4,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")

    # # Genus-Noise (LinSpec, All)
    list_n_neighbors = [50]
    list_min_dist = [1.0]
    output_path = "results/cluster_analysis/Final/ConvAE/Genus-Noise/All_four"
    model_weights_path = r"{arg}\AE_conv\Genus-Noise\test_lin_GN_All_epoch_10.pth".format(arg=dir_results)

    list_dataset_All = [[dataset_joined_nyctaloid], [dataset_joined_myotis], [dataset_joined_plecotus],
                        [dataset_joined_psuper], [dataset_joined_noise]]
    attribute_All = ["Nyctaloid", "Myotis", "Plecotus", "Pipistrellus", "Noise"]
    cluster_UMAP_auto_noise_post_AE(model_weights_path, output_path, list_dataset_All,
                                    list_n_neighbors, list_min_dist, list_text_labels=attribute_All, n_classes=5,
                                    network_type="ConvAE", image_post_prefix="LinSpec", input_mode="LinSpec")
