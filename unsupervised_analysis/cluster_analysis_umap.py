from __future__ import print_function
import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


def create_dataframe_from_torch_dataset(dataset, bat_class_list, class_mode="binary"):
    """ dataset: Dataset Instance
        class_mode: "binary" or "multi-class"
        NOTE: dataset and class_mode are not coupled but are semantically connected in the mode/labels!
              i.e. if dataset consists of noise and bats -> class_mode = "binary"
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    X_data = []
    y_data = []
    if class_mode == "multi-class":
        counter = 0
        species_label_before = bat_class_list[0]  # helps with counter
    for i, batch in enumerate(loader, 0):
        image = batch["image"].squeeze(0).detach().cpu().numpy()
        label = batch["label"]  # numpy array [2,]

        if image.shape[1] == 586:
            # print("Yes!")
            X_data.append(image)

            if class_mode == "multi-class":
                print("in multi-class!")
                species_label = batch["text_label"]  # string
                if species_label_before != species_label:
                    counter += 1
                species_label_before = species_label
                y_data.append(species_label)
                print(species_label)
            elif class_mode == "binary":
                if label[0][0] == 0:
                    y_data.append("bat")
                elif label[0][0] == 1:
                    y_data.append("noise")
            else:
                print("Nothing added")
    if class_mode == "multi-class":
        print("COUNTER: ", counter)
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    # print(X_data.shape, y_data.shape)
    # print(X_data[0].shape, y_data[1].shape)
    X_data = np.asarray([i.flatten() for i in X_data])
    y_data = np.asarray(y_data)
    # print(X_data.shape, y_data.shape)

    feat_cols = ["pixel_"+str(i) for i in range(X_data.shape[1])]
    df = pd.DataFrame(X_data, columns=feat_cols)
    df['y'] = y_data
    return df, feat_cols


# # # UMAP
# # UMAP compare E_19 bats and noise

def create_UMAP_plot(df, feat_cols, output_path, image_prefix="undefined", n_classes=2, alpha=0.5, plot_or_save="plot",
                     n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", n_epochs=200):

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                        n_epochs=n_epochs)
    umap_results = reducer.fit_transform(df[feat_cols].values)
    print("Shape Embedding:", umap_results.shape)

    df['UMAP-2d-one'] = umap_results[:, 0]
    df['UMAP-2d-two'] = umap_results[:, 1]
    # plt.figure(figsize=(16, 10))
    plt.figure(figsize=(3*6.4, 3*4.8))
    sns.scatterplot(
        x="UMAP-2d-one", y="UMAP-2d-two",
        hue="y",
        palette=sns.color_palette("husl", n_classes),
        data=df,
        legend="full",
        alpha=alpha,
        s=150
    )
    lgnd = plt.legend(prop={'size': 30})
    for i in range(n_classes):
        lgnd.legendHandles[i]._sizes = [90]
    fontdict = {'fontsize': 38}
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    plt.xlabel("dim 1", fontdict=fontdict)
    plt.ylabel("dim 2", fontdict=fontdict)

    if plot_or_save == "save":
        plt.savefig(os.path.join(output_path, f"{image_prefix}_UMAP-representation.png"))
    else:
        plt.show()
