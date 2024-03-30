# SiMBA: Simplified Mamba-based Architecture for Vision and Multivariate Time series
---------
Implementation of:
- https://arxiv.org/pdf/2403.15360.pdf

This repository contains the code for the SiMBA (Simplified Mamba-based Architecture for Vision and Multivariate Time series), a transformer-based model that can process both visual and time series inputs.

## Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```
python train.py --config config.json
```

where `config.json` is the path to the configuration file.

To evaluate the trained model, run the following command:

```
python evaluate.py --model_path best_model.pth --data_path data/test.json
```

where `best_model.pth` is the path to the trained model and `data/test.json` is the path to the test data.

## Configuration

The configuration file is a JSON file that specifies the hyperparameters for the model. The following options are available:

* `input_size`: The size of the input embedding.
* `hidden_size`: The size of the hidden state in the transformer layers.
* `num_layers`: The number of transformer layers.
* `num_heads`: The number of attention heads in each transformer layer.
* `dropout`: The dropout rate.
* `learning_rate`: The learning rate.
* `batch_size`: The batch size.
* `num_epochs`: The number of epochs to train the model.
* `dataset_type`: The type of dataset to use.
* `dataset_name`: The name of the dataset to use.
* `train_split`: The name of the train split.
* `val_split`: The name of the validation split.
* `local_train_data_path`: The path to the local train data file.
* `local_val_data_path`: The path to the local val data file.
* `num_workers`: The number of workers to use for data loading.
* `weight_decay`: The weight decay coefficient.
* `patience`: The number of epochs to wait before reducing the learning rate.
* `gradient_accumulation_steps`: The number of steps to accumulate gradients before updating the model.
* `early_stopping_patience`: The number of epochs to wait before early stopping.
* `shuffle`: Whether to shuffle the data.

## Data

The model can be trained on either image or time series datasets.

To use an image dataset, set `dataset_type` to `image` and specify the dataset name in `dataset_name`.

To use a time series dataset, set `dataset_type` to `time_series` and specify the dataset name in `dataset_name`.

To use a custom dataset, set `dataset_type` to `local` and specify the paths to the local train and val data files in `local_train_data_path` and `local_val_data_path`, respectively. The data files should be in JSON format and contain the following fields:

* `input`: The input data (image or time series).
* `label`: The label of the input data.

## Evaluation

The model is evaluated on the accuracy of its predictions. The accuracy is calculated as the percentage of correctly predicted labels.

## License

This code is licensed under the MIT License.