# Semi-supervised classification using two benchmark image datasets

Semi-supervised classification with TensorFlow employing 4 methods:

- Supervised
- Semi-supervised layer-wise
- Self-training
- Our proposed method: Self-training layer-wise.

# Datasets
<!--
The next two datasets contain grayscale images of 28 x 28 pixels.
The MNIST is a dataset of handwritten digits (0 - 9) with 10 classes and 60,000 images for training and 10,000 for testing.
The Fashion dataset has 10 classes as well and consist of images of Zolando’s articles from T-Shirt, Trousers, Bags, Ankle boots and others. It has 60,000 and 10,000 images for training and testisng, respectively.
-->

The subsequent datasets comprise 28 x 28 pixel grayscale images. The MNIST dataset encompasses handwritten digits (0 - 9) categorized into 10 classes, with 60,000 images designated for training and 10,000 for testing. The Fashion dataset also features 10 classes, comprising images of Zolando’s articles such as T-Shirts, Trousers, Bags, Ankle boots, and others. It includes 60,000 and 10,000 images for training and testing, respectively.

# The aim of the project
<!--
Deep learning has been succesfully employed in several domains using large amount of labeled datasets, which are carefully built for a long time. Obtaining massive annotated datasets is labor-intensive and time-consuming. In many areas, the lack of labeled data limits the use of deep learning as a powerful tool to resolve complex problems in real world applications. However, data with unknown labels is the way vast quantities of data exists in the real world.
-->
Deep learning has demonstrated efficacy across diverse domains through the utilization of extensive labeled datasets, meticulously curated over prolonged periods. The acquisition of voluminous annotated datasets is characterized by labor-intensive and time-consuming processes. Within numerous domains, the dearth of labeled data constrains the applicability of deep learning as a potent tool for addressing intricate challenges in real-world scenarios. Nevertheless, it is noteworthy that a significant portion of real-world data exists in an unlabeled state.

<!--
The aim of this project is to asses a proposed method that leverages unlabeled data to improve model performance using few labeled examples. The method performance is also compared with other method's.
-->
This project seeks to evaluate a proposed methodology that harnesses unlabeled data to enhance model performance with a limited number of labeled examples. Additionally, the performance of the proposed method is systematically compared with that of alternative methodologies.

<!--
We use two benchmark datasets (MNIST and FASHION) to test four methods, which are a purely supervised, a semi-supervised layer-wise, a self-training and a self-training layer-wise model, this latter is our proposed method. The models are built with recurrent network architecture LSTM-based.
-->
Two benchmark datasets, MNIST and FASHION, serve as the testing grounds for four distinct methodologies. These methodologies encompass a purely supervised approach, a semi-supervised layer-wise strategy, a self-training model, and a self-training layer-wise model—the latter being the proposed method. The models are constructed employing a recurrent network architecture based on Long Short-Term Memory (LSTM).

# The proposed method

<!--
We introduce an approach where the self-training method is employed to fine tune a layer-wise pre-trained network and we called it the self-training layer-wise method. The self-training method has shown good results with deep models when labeled examples are scarcely available (Li, et al., [2019](https://doi.org/10.1007/s12083-018-0702-9) and Xie, et al., [2020](https://doi.org/10.48550/arXiv.1911.04252)). The layer-wise procedure is utilised to learn hidden features from large amount of unlabeled data and it acts as a regulariser of the deep neural network weights (Xu, et al., [2018](https://doi.org/10.1109/ICMEW.2018.8551584)).
-->
We propose a methodology wherein the self-training technique is applied to refine a layer-wise pre-trained network, denoted as the self-training layer-wise method. The self-training approach has demonstrated efficacy in scenarios with limited labeled examples, particularly with deep models (Li, et al., [2019](https://doi.org/10.1007/s12083-018-0702-9) and Xie, et al., [2020](https://doi.org/10.48550/arXiv.1911.04252)). The layer-wise process is employed to extract latent features from abundant unlabeled data, concurrently serving as a regularizer for the weights of deep neural networks (Xu, et al., [2018](https://doi.org/10.1109/ICMEW.2018.8551584)).

<img width="500" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/1101442f-e1a8-4a38-a413-01c982a431bc">

<!--
The greedy layer-wise strategy and self-training are both semi-supervised learning and they manage unlabaled data in different ways it is possible to join these two methods to exploit unlabeled data twice. The proposed method works in 2 phases.
-->
The strategy of greedy layer-wise learning and self-training, both falling under the umbrella of semi-supervised learning, exhibit distinct approaches in handling unlabeled data. Integration of these two methods is conceivable to leverage unlabeled data in a dual capacity. The proposed methodology operates in two distinct phases.

1. The model is pre-trained with the greedy layer-wise procedure.
2. And it is fine tuned in self-training manner.

# Dataset configuration for experiments

<!--
The original subset of the datasets are changed to our subsets in order to carry out the experiments. To explore the case when we have a small number of labeled examples we dramatically reduce the number of training labeled examples from 60k to 10k and the rest are made for pre-training or as unlabeled data, which are 50k examples. The test subset remains the same.
-->
The initial subsets of the datasets are modified to facilitate experimentation in this study. To investigate scenarios involving a limited number of labeled examples, a substantial reduction is implemented, diminishing the training labeled examples from 60,000 to 10,000, with the remaining 50,000 serving as pre-training or unlabeled data. The test subset remains unaltered throughout the experiments.

| Dataset                          | Original subsets | Our subsets          |
| :------------------------------- | :--------------- | :------------------- |
| MNIST  (70k full)                | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |
| FASHION  (70k full)              | 60k Training     | 50k Pre-training     |
|                                  |                  | 10k Training         |
|                                  | 10k Test         | 10k Test             |


# Experiment design

<!--
For MNIST dataset, from the original subset training examples the few labeled examples (10k) represents the 16.67%, which is reduced until 0.33% (200 examples) for different experiments. This smallest percentage of labeled examples is the worst case. For FASHION dataset the same reduction is applied.
-->
In the MNIST dataset, the initial subset of training examples, comprising a few labeled instances (10,000), corresponds to 16.67%. This percentage is systematically diminished to 0.33% (200 examples) across various experiments, representing the scenario with the smallest proportion of labeled examples, indicative of the most challenging case. The identical reduction procedure is applied to the FASHION dataset.

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

<!--
The self-training layer-wise method can benefit itself twice with unlabeled data in a joined capacity of layer-wise procedure and self-training to have a better performance. Specifically, the figure below demonstrates that the self-training layer-wise method performs superior in the range of 0.33%-3.33% except in the 1.67% and then slightly better in 5.00%-16.67% than the rest of methods with MNIST dataset. We can also closely observe that our method achieves much better accuracies where the percentages of training examples are equal or less than 3.33%.
-->
The self-training layer-wise approach capitalizes on unlabeled data by combining layer-wise procedures and self-training, resulting in enhanced performance. Notably, the provided figure illustrates the superior performance of the self-training layer-wise method within the 0.33%-3.33% range, except at 1.67%, and marginally outperforms other methods in the 5.00%-16.67% range with the MNIST dataset. It is noteworthy that our method consistently attains significantly improved accuracies when the percentage of training examples equals or falls below 3.33%.

<img width="765" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/715a9c89-2819-4c5a-a59b-8fddf3fbcb3e">

# Results for FASHION dataset
<!--
Looking at the FASHION dataset chart, there is a slightly improvement of the self-training layer-wise performance over the other methods, which happen in the percentages of labeled examples of 0.67%-0.83%, 1.17%-1.50% and 5.00%-15.00%. Although not better than with the MNIST dataset, our method can still reach good accuracies than the other methods in percentages smaller than 3.33%.
-->
Examining the chart depicting the FASHION dataset, a modest enhancement is discerned in the performance of the self-training layer-wise method relative to other methodologies. This improvement is notable in the ranges of labeled examples percentages spanning 0.67%-0.83%, 1.17%-1.50%, and 5.00%-15.00%. While not surpassing the performance observed with the MNIST dataset, our method consistently achieves noteworthy accuracies superior to alternative approaches when the percentage of labeled examples is below 3.33%.

<img width="770" alt="image" src="https://github.com/ekchacon/semi-supervised-regular-size-datasets/assets/46211304/791929d8-c2b8-45be-aa16-7807a5b50e24">

# Discussion

<!--
The application of supervised learning methods in scenarios with limited labeled examples significantly impacts model performance across various applications
(Tiago, et al., [2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9893790), Glaser, et al., [2021](https://www.scitepress.org/Papers/2021/102680/102680.pdf), Pratama, et al., [2022](https://recil.ensinolusofona.pt/bitstream/10437/9972/1/Television%20reshaped%20by%20big%20data.pdf)). The scarcity of labeled examples, without the incorporation of pre-training methods or alternative techniques, inevitably results in a reduction in model accuracy. This phenomenon is evident in the performance of both the DeepHeart model (Ballinger, et al., [2018](https://ojs.aaai.org/index.php/AAAI/article/view/11891)) and the EfficientNet-B7 architecture (Zoph, et al., [2020](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf)).
-->
The utilization of supervised learning methodologies in contexts characterized by a paucity of labeled examples markedly influences model performance across diverse applications (Tiago, et al., [2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9893790), Glaser, et al., [2021](https://www.scitepress.org/Papers/2021/102680/102680.pdf), Pratama, et al., [2022](https://recil.ensinolusofona.pt/bitstream/10437/9972/1/Television%20reshaped%20by%20big%20data.pdf)). The constrained availability of labeled examples, in the absence of pre-training methods or alternative techniques, inevitably leads to a diminution in model accuracy. This observation is evident in the performance of the DeepHeart model (Ballinger, et al., [2018](https://ojs.aaai.org/index.php/AAAI/article/view/11891)) and the EfficientNet-B7 architecture (Zoph, et al., [2020](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf)).


We introduce a novel methodology designed for the optimal utilization of unlabeled data. This method underwent testing in scenarios with scarce labeled data, specifically on the MNIST and FASHION datasets. In the case of MNIST, our method demonstrated superior accuracy compared to alternative methods, particularly in instances where training examples were below 1.67%. However, the observed accuracy improvement of our method over others with the FASHION dataset seemed consistent across all dataset sizes, with less pronounced distinctions in cases of severe data scarcity, specifically when the percentage was less than 1.67%.

On the other hand, beyond the critical scenarios or beyond the 3.33% of training examples, the outcomes of alternative methods closely approximate those of the proposed method for the MNIST and FASHION datasets. This trend is attributed to the growing number of training examples. Thus, harnessing unlabeled data through semi-supervised methods proves most valuable in situations characterized by a severe shortage of labeled data.
