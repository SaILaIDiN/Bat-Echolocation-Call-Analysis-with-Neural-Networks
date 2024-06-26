# Bat Call Classification with CNNs
This project aims to develop a deep learning approach to classify bat echolocation calls. The project structure is as follows:
````
├── .gitignore
├── README.md
├── LICENSE.md
├── argparse_configs
├── data
│   ├── cleaning
│   │   ├── check_duplicates.py
│   │   ├── data_cleaner.py
│   │   ├── data_copy.py
│   │   ├── wav_filename_fix.py
│   ├── preanalysis
│   │   ├── check_data_distribution.py
│   │   ├── check_noise_pattern.py
│   ├── preprocessing
│   │   ├── data_conditioning.py
│   │   ├── data_separator.py
│   │   ├── image_creator.py
│   │   ├── label_creator.py
│   │   ├── recycle_data.py
│   ├── dataset.py
│   ├── path_provider.py
├── optimization_core
│   ├── logger_maker.py
│   ├── metrics.py
│   ├── models.py
│   ├── runbuilder.py
│   ├── train.py
│   ├── validate.py
├── unsupervised_analysis
│   ├── auto_clustering.py
│   ├── auto_eval_autoencoder.py
│   ├── check_image_quality_for_autoencoder.py
│   ├── cluster_analysis_umap.py
│   ├── datasets_for_clustering.py
│   ├── eval_autoencoder.py
│   ├── train_autoencoder.py
├── supervised_analysis
│   ├── auto_eval_check_years_binary.py
│   ├── auto_train_check_species_distinction.py
│   ├── auto_train_check_years_binary.py
│   ├── auto_train_check_years_multi.py
│   ├── check_species_distinction.py
│   ├── check_years_binary.py
│   ├── check_years_multi.py
├── results
│   ├── cluster_analysis
│   ├── logs
│   ├── model_txts
│   ├── noise_patterns
│   ├── pth_files
│   ├── runs


````
## Data
The **`'data'`** directory contains all the data used in this project. The subdirectories are described below:

* **`'cleaning'`**: contains scripts for cleaning the raw data and fixing any naming issues
* **`'preprocessing'`**: contains scripts for preprocessing the data, including conditioning, separating into images and labels, and recycling audio data to full extent.
* **`'preanalysis'`**: contains scripts for a first analysis of the data after preprocessing

## Models
The **`'optimization_core'`** directory contains the core files for building, training and evaluating the models. The subdirectories are described below:

* **`'logger_maker.py'`**: contains a class for logging the model training process
* **`'metrics.py'`**: contains the metrics used to evaluate the model performance
* **`'models.py'`**: contains the deep learning models used in this project
* **`'runbuilder.py'`**: contains a class for generating the model hyperparameters
* **`'train.py'`**: contains the main script for training the models
* **`'validate.py'`**: contains the main script for validating the models

## Results
The **`'results'`** directory contains the output of the model training and evaluation. The subdirectories are described below:

* **`'cluster_analysis'`**: contains plots of the clustered data
* **`'logs'`**: contains the log files for the model training, validation and testing
* **`'model_txts'`**: contains the text files of the trained models
* **`'noise_patterns'`**: contains the noise patterns used in the data analysis
* **`'pth_files'`**: contains the trained model weights in PyTorch format
* **`'runs'`**: contains the TensorBoard logs for each run

## Reproducing the Results 
*(Requires adjustment for different but similar datasets)*

To reproduce the results of the experiments, follow these steps:
1. Choose your data including audio data and labels. Add the path of the labeled data in **`'path_provider.py'`** and in **`'argparse_configs/'`**.
2. Run the data cleaning scripts in the **`'data/cleaning/'`** directory to remove duplicates and fix file naming inconsistencies.
3. Run the data preprocessing scripts in the **`'data/preprocessing/'`** directory to convert the audio files to images and create labels that are interpretable for Pytorch.
4. Run the data preanalysis scripts in the **`'data/preanalysis/'`** directory to check the distribution of the data and get a feeling for noise patterns.
5. Choose between the branches **`'unsupervised_analysis/'`** and **`'supervised_analysis/'`**. 
### Unsupervised Analysis
1. The script **`'datasets_for_clustering.py'`** provides the dataset instances used for training the autoencoder and running the clustering algorithms.
2. The script **`'auto_eval_autoencoder.py'`** combines the training from **`'train_autoencoder.py'`** and the evaluation functionality from **`'eval_autoencoder.py'`**.
3. Use the pretrained model weights obtained after step 2 and run **`'auto_clustering.py'`** which automates the clustering functions from **`'cluster_analysis_umap.py'`**.
### Supervised Analysis
1. Use either **`'check_years_binary.py'`**, **`'check_years_multi.py'`** or **`'check_species_distinction.py'`** to create all relevant dataset instances for bat/noise, bat genus and bat species classification.
2. Call either **`'auto_train_check_years_binary.py'`**, **`'auto_train_check_years_multi.py'`** or **`'auto_train_check_species_distinction.py'`** to automatically train and evaluate the performance of the chosen task.
3. Scripts like **`'auto_eval_check_years_binary.py'`** are further analyses of model outputs but beyond the scope of the recent paper submission.

## General usage of the pipeline (for other datasets)
The directories of **`'supervised_analysis/'`** and **`'unsupervised_analysis/'`** are both hand-tailored for the used dataset. 
The following description helps in understanding the main structure of the code and in the application of the optimization core on other bat echolocation call datasets:
0. Adapt your dataset into a format of spectrograms and corresponding unique label filenames. Choose **`'[0, 1]'`** for bat and **`'[1, 0]'`** for noise for example as labels. 
Filenames of the labels and images should be stored in **`'.txt'`** files within the **`'data/'`** folder.
1. Adjust the paths in **`'argparse_configs/'`** and **`'data/path_provider.py'`** for your dataset and its labels.
2. Create missing directories if required.
3. Train the model by running **`'train.py'`** in the **`'optimization_core/'`** directory. Use the flags to specify the configuration of your training run(s). Consider the argument parser. 
Note: **`'train.py'`** is already including the evaluation function from **`'validate.py'`**.
4. Evaluate the model by running **`'validate.py'`** in the **`'optimization_core/'`** directory. Use the flags to specify the trained model and the data to use for evaluation, respectively.
5. View the results of the experiments in the **`'results/'`** directory. Use TensorBoard to view the logs in the **`'results/runs/'`** directory.

## Meta Data
This project is developed with:
* Python 3.7
* Pytorch 1.9.0
* CUDA 11.3
* librosa 0.8.0
* umap-learn 0.5.3
* numpy 1.21.6
* pandas 1.0.3
* seaborn 0.11.2
* matplotlib 3.4.3

## Citation

If you use this code in your research, please cite the following paper:

```
@article{
  title={An Efficient Neural Network Design Incorporating Autoencoders for the Classification of Bat Echolocation Sounds},
  author={Alipek, S., Maelzer, M., Paumen, Y., Schauer-Weisshahn, H., & Moll, J.},
  journal={Animals},
  volume={13},
  number={16},
  pages={2560},
  year={2023},
  publisher={MDPI},
  doi={https://doi.org/10.3390/ani13162560}
}
```

## License
This project is licensed under the MIT License.
