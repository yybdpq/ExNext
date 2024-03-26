# ExNext
we focus on implementing two key aspects of trust-worthiness in recommender systems – accuracy and explainability – and present **Self-explainable Next POI Recommendation**, a novel ante-hoc self-explainable frame-work for next POI recommendation, jointly learns to predict and
explain the recommendation in an ante-hoc manner (i.e. learning to explain during training itself, as opposed to post-hoc explainability methods popularly used today).



## Installation
Install  dependencies in `requirements.txt`:
```shell
pip install -r requirements.txt
```



## Environment

This is the environment when we trained our model.

+ SYSTEM: Ubuntu 22.04.3 LTS
+ GPU: NVIDIA RTX 4090
+ CPU: 13th Gen Intel(R) Core(TM) i7-13700KF

\```

python==3.10.12

torch==2.1.1

tqdm==4.66.1

pyyaml==6.0.1

pandas==2.1.3

numpy==1.26.2

tensorboard==2.15.1

scikit-learn==1.3.2 

shapely==2.0.2

\```

For our explainable model, We configure the learning rate as 1𝑒 −4 and the epoch of 20, while keeping 𝛽 constant at 1𝑒 −2 across all three datasets.



## Dataset

### Preprocess
The training and test datasets of the system are stored in the `data/raw` folder, including three raw datasets NYC, TKY and CA.

If we first train the model with 'main.py', it will generate a processed folder `data/processed` store the processed datasets. After that, the model will be trained to use the processed files directly and skip the data preprocessing step.

In particular, for the CA dataset, we generated the raw dataset using the following command

```shell
 python pre/generate_ca.py
```

### Original Link

Here we introduce where and how our data comes from. Our dataset was sourced from [STHGCN](https://github.com/ant-research/Spatio-Temporal-Hypergraph-Model),

thanks for all the data providers.

+ NYC:
  + Preprocessed: https://github.com/songyangme/GETNext/blob/master/dataset/NYC.zip
+ TKY: 
  + Raw: http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
+ CA: 
  + Raw: http://snap.stanford.edu/data/loc-gowalla.html; 
  + Category information: https://www.yongliu.org/datasets.html



## Train

Train the model using python `main.py`. All hyper-parameters are defined in `conf`.

We can reproduce the  performance of our model with the script below. Please choose 'nyc', 'tky', 

or 'ca' for **{dataset_name}**.

```shell
python main.py -f {dataset_name}.yml
```

 
