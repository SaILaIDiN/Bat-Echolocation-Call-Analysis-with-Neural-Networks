import os
import torch
import torch.nn as nn
from optimization_core.metrics import multi_class_label_translator, species_to_genus_translator


def eval_net(model, loader, device, class_mode="binary", species_or_genus="species", bat_class_list=None, step_tot=0,
             subset="undefined", config_name="None", optim_name="SGD", path_prefix="results", test_content="Test",
             remark=""):
    """ Evaluates the given dataset.
        Parameter "set" marks which dataset(-part) is to evaluate.
    """
    model.eval()

    total_error = 0
    os.makedirs(f"{path_prefix}/model_txts/comparisons/{optim_name}/{test_content}/{config_name}{remark}", exist_ok=True)

    wf = open(f"{path_prefix}/model_txts/comparisons/{optim_name}/{test_content}/{config_name}{remark}/"
              f"predictions_after_train_steps_{step_tot}_set_{subset}.txt", mode="w")

    pred_collector = []  # for computation of evaluation metrics
    bin_labels_collector = []  # "-" bat or noise
    species_labels_collector = []  # species names as type str or 'Noise'
    image_path_collector = []  # carries the filenames for more in-depth error analysis
    species_labels_text_collector = []  # secure species labels in text form in any case

    for i, batch in enumerate(loader, 0):
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)  # Shape [N, H, W] -> [N, C, H, W]
        bin_labels = batch["label"]
        species_labels = batch["text_label"]
        image_path = batch["image_path"]

        # secure species labels in text form in any case
        species_labels_text = species_labels
        if class_mode == "binary":
            labels = bin_labels.to(device, dtype=torch.float32)
        else:
            if species_or_genus == "genus":
                # Currently hardcoded for our case if Pipistrellus species
                # bat_class_list = ['Nyctaloid', 'Myotis', 'Psuper']
                bat_class_list = ['Nyctaloid', 'Myotis', 'Plecotus', 'Psuper']
                species_to_combine = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
                genus_name = 'Psuper'
                species_labels = species_to_genus_translator(
                    species_labels, species_to_combine, genus_name)

            # # translate species string labels to one-hot-encoded vectors
            trans_dict = multi_class_label_translator(bat_class_list)  # 'dict'
            species_labels = [trans_dict[species] for species in species_labels]  # 'list' of 'ndarray'
            labels = torch.as_tensor(species_labels)  # 'torch.Tensor' of 'torch.Tensor'
            labels = labels.to(device, dtype=torch.float32)

        if images.shape[3] == 586:
            with torch.no_grad():
                prediction = model(images)
            # # # TEST: Compare raw predictions with one hot encoded labels
            m = nn.Softmax(dim=1)

            pred_detached = m(prediction).cpu().detach().numpy()
            labels_detached = labels.cpu().detach().numpy()

            pred_collector.append(pred_detached)
            bin_labels_collector.append(bin_labels.numpy())
            species_labels_collector.append(species_labels)  # strings for binary, ints for multi
            image_path_collector.append(image_path)
            species_labels_text_collector.append(species_labels_text)

            counter = 0
            if i % 50 == 0:  # for "results/model_txts"
                [wf.write(f"\n{[round(x, 3) for x in pred_detached[i]]}_{labels_detached[i]}_{species_labels[i]}")
                 for i in range(0, len(images))]
                counter += 1
                wf.write(f"\ncounter: {counter}")
            # # # END Test
            criterion = nn.BCEWithLogitsLoss()
            error = criterion(prediction, labels)
            total_error += error

    wf.close()
    model.train()
    return total_error / i, pred_collector, bin_labels_collector, image_path_collector, species_labels_collector, \
        species_labels_text_collector


if __name__ == "__main__":
    from optimization_core.models import YPNetNew
    from torch.utils.data import DataLoader, ConcatDataset
    from data.dataset import BatDataset, NoiseDataset, partition_dataset
    from data.path_provider import dir_results, dir_results_remote, provide_paths

    # Define mode and set up main paths
    system_mode = input("Enter system mode. Choose either 'local_mode' or 'remote_mode': ")
    if system_mode == "local_mode":
        path_prefix = dir_results
        main_paths = "argparse_configs/mp_local_2019.txt"
        local_or_remote = "local"
    elif system_mode == "remote_mode":
        path_prefix = dir_results_remote
        main_paths = "argparse_configs/mp_remote_2019.txt"
        local_or_remote = "remote"
    else:
        path_prefix = "CHANGE_DIR"

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote=local_or_remote, year=2019)

    # Define dataset instance
    full_set_Noise = NoiseDataset(dir_main, dir_noise_txt, "W_05", "MFCC")
    full_set_Bats = BatDataset(dir_main, dir_bats_txt, ["Myotis"], sec=0 + 1, attr_oh="W_05", spec_type="MFCC")
    full_set = ConcatDataset([full_set_Bats, full_set_Noise])
    partitions = partition_dataset(full_set, train_size=0.6, val_size=0.2, train_sub_size=0.05, toy_factor=0.3)
    dataset = partitions[0:3]
    val_loader = DataLoader(dataset[1], batch_size=1, shuffle=True, num_workers=0)

    # Set up pretrained model
    dir_model_weights = input("Enter path to the '.pth' file of your trained network or model: ")
    net = YPNetNew(n_classes=2, n_hidden_out=128)
    net.load_state_dict(torch.load(dir_model_weights))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    val_loss = eval_net(net, val_loader, device)
    print("val_loss: ", val_loss)
