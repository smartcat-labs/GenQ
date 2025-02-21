<!--- BADGES: START --->
<p align="center">
    <a href="https://smartcat.io/">
        <picture>
            <!-- Image for dark theme -->
            <source media="(prefers-color-scheme: dark)" srcset="https://github.com/smartcat-labs/product2query/blob/feat/readme/images/logo_smartcat_white.svg">
            <!-- Image for light theme -->
            <source media="(prefers-color-scheme: light)" srcset="https://github.com/smartcat-labs/product2query/blob/feat/readme/images/logo_smartcat_black.svg">
            <!-- Fallback image -->
            <img alt="Company" src="https://smartcat.io/wp-content/uploads/2023/07/logo.svg">
        </picture>
    </a>
</p>
<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <!--CHECK THIS OUT AFTER CHANGING REPO TO PUBLIC -->
    <a href="https://github.com/smartcat-labs/product2query/blob/dev/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/smartcat-labs/product2query.svg?logo=github&style=flat&color=green">
    </a>
    <!--CHANGE THE LINK TO THE NEW MODELS -->
    <a href="https://huggingface.co/collections/smartcat/product2query-6783f6786b250284f060918d">
        <img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow">
    </a>
    <a href="https://huggingface.co/datasets/smartcat/Amazon-2023-GenQ">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue">
    </a>
    <!-- <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a> -->
    <a href="https://github.com/smartcat-labs">
        <img alt="Open_Source" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<!--- BADGES: END --->  
# GenQ

🤖 ✨ 🔍 Generate precise, realistic user-focused search queries from product text 🛒 🚀 📊

## Introduction
This project contains the scripts used to creaate the fine-tuned model GenQ. 
This model has been specifically designed to generate high-quality queries for e-commerce products, achieving improved performance compared to the base model.

This repository serves as a comprehensive resource for:

**Data Preprocessing**: Scripts and utilities for preparing the dataset used in fine-tuning, ensuring a robust and effective training process.  

**Model Fine-Tuning**: Code and configurations for fine-tuning the base model on the customized dataset.  

**Performance Insights**: Configurations and examples showcasing the model's performance improvements and applications.  

By leveraging the GenQ model, e-commerce platforms can enhance search quality and generate more relevant queries tailored to their products.  
Whether you're looking to understand the data preparation process, fine-tune your own model, or integrate this solution into your workflow, this repository has you covered.  

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Training](#training)
  - [Training Parameters](#training-parameters)
  - [Metrics Used](#metrics-used)
- [Setup](#setup)  
  - [Installation](#installation)
- [Usage and Examples](#usage)
  - [Intended Use](#intended-use)
  - [Examples of Use](#examples-of-use)
  - [Examples](#examples)
  - [How to Use](#how-to-use)

## Model Details

<strong>Model Name:</strong> Fine-Tuned Query-Generation Model <br>
<strong>Model Type:</strong> Text-to-Text Transformer <br>
<strong>Architecture:</strong> Based on a pre-trained transformer model: [BeIR/query-gen-msmarco-t5-base-v1](https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1) <br>
<strong>Primary Use Case:</strong> Generating accurate and relevant search queries from product descriptions<br>
<strong>Dataset:</strong> [smartcat/Amazon-2023-GenQ](https://huggingface.co/datasets/smartcat/Amazon-2023-GenQ)<br>
   
<br>

There are three models in our collection that are trained differently:   

**T5-GenQ-T-v1**: Trained on only the product titles  
**T5-GenQ-TD-v1**: Trained on titles + descriptions of the products  
**T5-GenQ-TDE-v1**: Trained on titles + descriptions of the products and a set of products with titles only (2x of the dataset)


## Training

### Training parameters:
<ul>
    <li><strong>max_input_length:</strong> 512</li>
    <li><strong>max_target_length:</strong> 30</li>
    <li><strong>batch_size:</strong> 48</li>
    <li><strong>num_train_epochs:</strong> 8</li>
    <li><strong>evaluation_strategy:</strong> epoch</li>
    <li><strong>save_strategy:</strong> epoch</li>
    <li><strong>learning_rate:</strong> 5.6e-05</li>
    <li><strong>weight_decay:</strong> 0.01 </li>
    <li><strong>predict_with_generate:</strong> true</li>
    <li><strong>load_best_model_at_end:</strong> true</li>
    <li><strong>metric_for_best_model:</strong> eval_rougeL</li>
    <li><strong>greater_is_better:</strong> true</li>
    <li><strong>logging_startegy:</strong> epoch</li>
</ul>

### Metrics Used:
**[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))**, or **R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation, is a set of metrics used for evaluating automatic summarization and machine translation in NLP. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation. ROUGE metrics range between 0 and 1, with higher scores indicating higher similarity between the automatically produced summary and the reference.  

In our evaluation, ROUGE scores are scaled to resemble percentages for better interpretability. The metric used in the training was ROUGE-L.

The results of our model variations are:

<table border="1" class="dataframe">
  <tr style="text-align: center;">
    <th>Model</th>
    <th>Epoch</th>
    <th>Step</th>
    <th>ROUGE-1</th>
    <th>ROUGE-2</th>
    <th>ROUGE-L</th>
    <th>ROUGE-Lsum</th>
  </tr>
  <tr>
    <td><b>T5-GenQ-T-v1</b></td>
    <td>7.0</td>
    <td>29995</td>
    <td>75.2151</td>
    <td>54.8735</td>
    <td><b>74.5142</b></td>
    <td>74.5262</td>
  </tr>
  <tr>
    <td><b>T5-GenQ-TD-v1</b></td>
    <td>8.0</td>
    <td>34280</td>
    <td>78.2570</td>
    <td>58.9586</td>
    <td><b>77.5308</b></td>
    <td>77.5466</td>
  </tr>
  <tr>
    <td><b>T5-GenQ-TDE-v1</b></td>
    <td>8.0</td>
    <td>68552</td>
    <td>76.9075</td>
    <td>57.0980</td>
    <td><b>76.1464</b></td>
    <td>76.1502</td>
  </tr>
</table>

## Setup

### Installation
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/smartcat-labs/product2query.git
```
For installing and setting up poetry:
[Poetry Documentation](https://python-poetry.org/docs/)

After installing and setting up poetry, run   
```python
poetry install --no-root
```  
to install all necessary dependencies
<!--Set up the links to files-->
### Run training
For running the training, prepare the **config.yaml** file. If you don't want to modify it you can simply run the training with:
```bash
python train/train.py \  
  -c config/config.yaml \  
  -o 'finetuned-amazon-product2query' \  
  --log_level INFO
```
If you want to test out the training on a sample of the dataset, set the ```dev``` flag to ```True``` in the **config.yaml** or simply run the training with:
```bash
python train/train.py \  
  -c config/test_config.yaml \  
  -o 'finetuned-amazon-product2query' \  
  --log_level INFO
```
The best three models will be saved to train\runs\date_time\ by default.  
The checkpoint with the highest ROUGE-L score, which you can check in **evaluation_metrics.csv**, should be your best performing model.  
To check out each checkpoint, you can run the evaluation.

### Evaluation
The evaluation consists of generating queries with two models and calculating the results of each ROUGE metric. In our case, we ran the evaluation with the pre-trained model and our fine-tuned model.  

For running the evaluation, prepare the **eval_config.yaml** file. You must set the ```model_paths``` in the file to your checkpoint path to test out your model. If you don't want to modify the file you can simply run the evaluation with:
```bash
python eval/checkpoint_eval.py config/eval_config.yaml
```
This will run the evaluation with our fine-tuned model by default.   

After it's finished, you can look at the results in the **generated_results.csv** saved to eval\runs\date_time\ by default.   
For further analysis use the **results_analysis.ipynb** notebook with your model to see specific cases where your model had better/worse results.

## Usage

### Intended use
This model is designed to improve e-commerce search functionality by generating user-friendly search queries based on product descriptions. It is particularly suited for applications where product descriptions are the primary input, and the goal is to create concise, descriptive queries that align with user search intent.
### Examples of Use:
<li>Generating search queries for product indexing.</li>
<li>Enhancing product discoverability in e-commerce search engines.</li>
<li>Automating query generation for catalog management.</li>
   
### Examples

<table border="1" text-align: center>
  <thead>
    <tr>
      <th>Target Query</th>
      <th>Before Fine-tuning</th>
      <th>After Fine-tuning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>flannel pajama set women's</td>
      <td>what to wear with a pajama set</td>
      <td>women's plaid pajama set</td>
    </tr>
    <tr>
      <td>custom name necklace</td>
      <td>what is casey name necklace</td>
      <td>personalized name necklace</td>
    </tr>
    <tr>
      <td>Large Satin Sleep Cap</td>
      <td>what is the size of a silk bonnet</td>
      <td>satin sleep cap</td>
    </tr>
  </tbody>
</table>


### How to use
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("smartcat/T5-product2query-finetune-v1")
tokenizer = AutoTokenizer.from_pretrained("smartcat/T5-product2query-finetune-v1")

description = "Silver-colored cuff with embossed braid pattern. Made of brass, flexible to fit wrist."

inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
generated_ids = model.generate(inputs["input_ids"], max_length=30, num_beams=4, early_stopping=True)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) 

```

Graph showing RougeL metric during each epoch

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/rougeL.png?raw=true)


Graph of comparison between models for each rouge metric

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/output.png?raw=true)

Graph of comparison between rouge score distributions

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/output2.png?raw=true)

Graph showing average score for each metric for base and fine-tuned model

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/avg_score.png?raw=true)

