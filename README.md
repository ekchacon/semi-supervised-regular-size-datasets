# Semi-supervised classification using two benchmark image datasets

Four methods comparison performance with few labeled datasets.

# Datasets

The next two datasets contain grayscale images of 28 x 28 pixels.

The MNIST is a dataset of handwritten digits (0 - 9) with 10 classes and 60,000 images for training and 10,000 for testing.

The Fashion dataset has 10 classes as well and consist of images of Zolando’s articles from T-Shirt, Trousers, Bags, Ankle boots and others. It has 60,000 and 10,000 images for training and testisng, respectively.

# The aim of the project

Deep learning has been succesfully employed in several domains using large amount of labeled datasets, which are carefully built for a long time. Obtaining massive annotated datasets is labor-intensive and time-consuming. In many areas, the lack of labeled data limits the use of deep learning as a powerful tool to resolve complex problems in real world applications. However, data with unknown labels is the way vast quantities of data exists in the real world.

The aim of this project is to asses a proposed method that leverages unlabeled data to improve model performance using few labeled examples. The method performance is also compared with other method's.

We use two benchmark datasets (MNIST and FASHION) to test four methods, which are a purely supervised, a semi-supervised layer-wise, a self-training and a self-training layer-wise model, this latter is our proposed method. The models are built with recurrent network architecture LSTM-based.

# The proposed method

We introduce an approach where the self-training method is employed to fine tune a layer-wise pre-trained network and we called it the self-training layer-wise method. The self-training method has shown good results with deep models when labeled examples are scarcely available (Li, et al., [2019](https://doi.org/10.1007/s12083-018-0702-9) and Xie, et al., [2020](https://doi.org/10.48550/arXiv.1911.04252)). The layer-wise procedure is utilised to learn hidden features from large amount of unlabeled data and it acts as a regulariser of the deep neural network weights (Xu, et al., [2018](https://doi.org/10.1109/ICMEW.2018.8551584)).

<img width="500" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/1101442f-e1a8-4a38-a413-01c982a431bc">

The greedy layer-wise strategy and self-training are both semi-supervised learning and they manage unlabaled data in different ways it is possible to join these two methods to exploit unlabeled data twice. The proposed method works in 2 phases.

1. The model is pre-trained with the greedy layer-wise procedure.
2. And it is fine tuned with self-training.

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


# Experiment design
 
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

# Results for MNIST dataset

The self-training layer-wise method can benefit itself twice with unlabeled data in a joined capacity of layer-wise procedure and self-training to have a better performance. Specifically, the figure below demonstrates that the self-training layer-wise method performs superior in the range of 0.33%-3.33% except in the 1.67% and then slightly better in 5.00%-16.67% than the rest of methods with MNIST dataset. We can also closely observe that our method achieves much better accuracies where the percentages of training examples are equal or less than 3.33%.

<img width="765" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/715a9c89-2819-4c5a-a59b-8fddf3fbcb3e">

# Results for FASHION dataset

Looking at the FASHION dataset chart, there is a slightly improvement of the self-training layer-wise performance over the other methods, which happen in the percentages of labeled examples of 0.67%-0.83%, 1.17%-1.50% and 5.00%-15.00%. Although not better than with the MNIST dataset, our method can still reach good accuracies than the other methods in percentages smaller than 3.33%.

<img width="770" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/791929d8-c2b8-45be-aa16-7807a5b50e24">
