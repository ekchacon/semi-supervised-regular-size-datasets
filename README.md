# Semi-supervised classification using two benchmark image datasets
Four methods comparison performance with few labeled datasets.

# the aim of the project

Deep learning has been succesfully employed in several domains using large amount of labeled datasets, which are carefully built and for a long time. Obtaining massive annotated datasets is labor-intensive and time-consuming. In many areas, the lack of labeled data limits the use of deep learning as a powerful tool to resolve complex problems in real world applications. However, data with unknown labels is the way vast quantities of data exists in the real world.

The aim of this project is to asses a proposed method that leverages unlabeled data to improve model performance using few labeled examples. The method performance is also compared with other method's.

We use two benchmark datasets (MNIST and FASHION) to test four models, which are a purely supervised, a semi-supervised layer-wise, a self-training and a self-training layer-wise model, the latter is our proposed model. The models are built with recurrent network architecture LSTM-based.

# Dataset configuration for experiments
 the original subset of the datasets are changed to our subsets in order to carry out the experiments.


| Dataset                          | Original subsets | Our subsets          |
| :------------------------------- | :--------------- | :------------------- |
| MNIST  (70k full)                | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |
| FASHION  (70k full)              | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |
