# Semi-supervised classification using two benchmark image datasets
Four methods comparison performance with few labeled datasets.

# The aim of the project

Deep learning has been succesfully employed in several domains using large amount of labeled datasets, which are carefully built and for a long time. Obtaining massive annotated datasets is labor-intensive and time-consuming. In many areas, the lack of labeled data limits the use of deep learning as a powerful tool to resolve complex problems in real world applications. However, data with unknown labels is the way vast quantities of data exists in the real world.

The aim of this project is to asses a proposed method that leverages unlabeled data to improve model performance using few labeled examples. The method performance is also compared with other method's.

We use two benchmark datasets (MNIST and FASHION) to test four methods, which are a purely supervised, a semi-supervised layer-wise, a self-training and a self-training layer-wise model, this latter is our proposed method. The models are built with recurrent network architecture LSTM-based.

# Dataset configuration for experiments
The original subset of the datasets are changed to our subsets in order to carry out the experiments. To explore the case when we have a small number of labeled examples we dramatically reduce the number of training labeled examples from 60k to 10k and the rest are made for pre-training or as unlabeled data, which are 50k examples. The test subset remains the same.


| Dataset                          | Original subsets | Our subsets          |
| :------------------------------- | :--------------- | :------------------- |
| MNIST  (70k full)                | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |
| FASHION  (70k full)              | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |


# Experiment desing
 
For MNIST dataset, from the original subset training examples the few labeled examples (10k) represents the 16.67%, which is reduced until 0.33% (200 examples) for different experiments. This smallest percentage of labeled examples is the worst case. For FASHION dataset the same reduction is applied.

| MNIST | FASHION | %      |
| :---- | :------ | :----- |
| 10k   | 10k     | 16\.67 |
| 9k    | 9k      | 15\.00 |
| 8k    | 8k      | 13\.33 |
| 7k    | 7k      | 11\.67 |
| 6k    | 6k      | 10\.00 |
| 5k    | 5k      | 8\.33  |
| 4k    | 4k      | 6\.67  |
| 3k    | 3k      | 5\.00  |
| 2k    | 2k      | 3\.33  |
| 1k    | 1k      | 1\.67  |
| 0\.9k | 0\.9k   | 1\.50  |
| 0\.8k | 0\.8k   | 1\.33  |
| 0\.7k | 0\.7k   | 1\.17  |
| 0\.6k | 0\.6k   | 1\.00  |
| 0\.5k | 0\.5k   | 0\.83  |
| 0\.4k | 0\.4k   | 0\.67  |
| 0\.3k | 0\.3k   | 0\.50  |
| 0\.2k | 0\.2k   | 0\.33  |
