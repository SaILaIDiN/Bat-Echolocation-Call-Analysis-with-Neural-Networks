from unsupervised_analysis.cluster_analysis_umap import create_UMAP_plot, create_dataframe_from_torch_dataset


def cluster_UMAP_auto_bat_noise(dataset, list_n_neighbors, list_min_dist, output_path=""):
    df, feat_cols = create_dataframe_from_torch_dataset(dataset, bat_class_list=None,
                                                        class_mode="binary")
    n_epochs = 100
    for n_neighbors in list_n_neighbors:
        for min_dist in list_min_dist:
            image_prefix = f"BC_neighbors_{n_neighbors}_min_dist_{min_dist}_n_epochs_{n_epochs}"
            create_UMAP_plot(df, feat_cols, output_path, image_prefix=image_prefix, n_classes=2, alpha=0.5,
                             plot_or_save="save", n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, n_epochs=n_epochs)


def cluster_UMAP_auto_species(dataset, bat_class_list, list_n_neighbors, list_min_dist, n_classes=3, output_path=""):
    df, feat_cols = create_dataframe_from_torch_dataset(dataset, bat_class_list,
                                                        class_mode="multi-class")
    n_epochs = 100
    for n_neighbors in list_n_neighbors:
        for min_dist in list_min_dist:
            image_prefix = f"MC_neighbors_{n_neighbors}_min_dist_{min_dist}_n_epochs_{n_epochs}"
            create_UMAP_plot(df, feat_cols, output_path, image_prefix=image_prefix, n_classes=n_classes, alpha=0.5,
                             plot_or_save="save", n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, n_epochs=n_epochs)


