---
datasets:
- smartcat/Amazon_Sample_Metadata_2023
language:
- en
metrics:
- rouge
base_model:
- BeIR/query-gen-msmarco-t5-base-v1
library_name: transformers
---
<!--- BADGES: START --->
<p align="center">
    <a href="https://smartcat.io/">
        <img alt="Company" src="https://smartcat.io/wp-content/uploads/2023/07/logo.svg">
    </a>
</p>
<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/smartcat-labs/product2query/blob/dev/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/smartcat-labs/product2query.svg?color=green">
    </a>
    <a href="https://huggingface.co/collections/smartcat/product2query-6783f6786b250284f060918d">
        <img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow">
    </a>
    <a href="https://huggingface.co/datasets/smartcat/Amazon_Sample_Metadata_2023/viewer/products2query">
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
# Product to Query
Generate realistic user queries from product text.

## Introduction
This project contains the fine-tuned model product2query, which is based on the pre-trained model [BeIR/query-gen-msmarco-t5-base-v1](https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1). 
The fine-tuned model has been specifically designed to generate high-quality queries for e-commerce products, achieving improved performance compared to the base model.

This repository serves as a comprehensive resource for:

Data Preprocessing: Scripts and utilities for preparing the dataset used in fine-tuning, ensuring a robust and effective training process.
Model Fine-Tuning: Code and configurations for fine-tuning the base model on the customized dataset.
Performance Insights: Configurations and examples showcasing the model's performance improvements and applications.
By leveraging the fine-tuned product2query model, e-commerce platforms can enhance search quality and generate more relevant queries tailored to their products. Whether you're looking to understand the data preparation process, fine-tune your own model, or integrate this solution into your workflow, this repository has you covered.

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Training Parameters](#training-parameters)
- [Intended Use](#intended-use)
- [Examples of Use](#examples-of-use)
- [Metrics Used](#metrics-used)
- [Examples](#examples)
- [How to Use](#how-to-use)

### Model Details

<strong>Model Name:</strong> Fine-Tuned Query-Generation Model <br>
<strong>Model Type:</strong> Text-to-Text Transformer <br>
<strong>Architecture:</strong> Based on a pre-trained transformer model: [BeIR/query-gen-msmarco-t5-base-v1](https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1) <br>
<strong>Primary Use Case:</strong> Generating accurate and relevant search queries from product descriptions for e-commerce applications.<br>
<strong>Dataset:</strong> [smartcat/Amazon_Sample_Metadata_2023](https://huggingface.co/datasets/smartcat/Amazon_Sample_Metadata_2023)<br>

### Training parameters:
<ul>
    <li><strong>max_input_length:</strong> 512</li>
    <li><strong>max_target_length:</strong> 30</li>
    <li><strong>batch_size:</strong> 16</li>
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

### Intended use
This model is designed to improve e-commerce search functionality by generating user-friendly search queries based on product descriptions. It is particularly suited for applications where product descriptions are the primary input, and the goal is to create concise, descriptive queries that align with user search intent.
### Examples of Use:
<li>Generating search queries for product indexing.</li>
<li>Enhancing product discoverability in e-commerce search engines.</li>
<li>Automating query generation for catalog management.</li>

### Metrics Used:
<strong>ROUGE-L:</strong> Measures the longest common subsequence between the target query and the generated query to evaluate alignment and relevance.

Based on the results below, the best model was achived in the 7th epoch and it represents the model that is chosen for the final model.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th>epoch</th>
      <th>step</th>
      <th>loss</th>
      <th>grad_norm</th>
      <th>learning_rate</th>
      <th>eval_loss</th>
      <th>eval_rouge1</th>
      <th>eval_rouge2</th>
      <th>eval_rougeL</th>
      <th>eval_rougeLsum</th>
      <th>eval_runtime</th>
      <th>eval_samples_per_second</th>
      <th>eval_steps_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.0</td>
      <td>14265</td>
      <td>1.9226</td>
      <td>11.211930</td>
      <td>0.000049</td>
      <td>1.681115</td>
      <td>56.6365</td>
      <td>34.2513</td>
      <td>56.1039</td>
      <td>56.0981</td>
      <td>712.4442</td>
      <td>35.594</td>
      <td>2.225</td>
    </tr>
    <tr>
      <td>2.0</td>
      <td>28530</td>
      <td>1.6894</td>
      <td>5.820090</td>
      <td>0.000042</td>
      <td>1.606815</td>
      <td>57.6006</td>
      <td>35.2668</td>
      <td>57.0582</td>
      <td>57.0577</td>
      <td>725.4298</td>
      <td>34.957</td>
      <td>2.185</td>
    </tr>
    <tr>
      <td>3.0</td>
      <td>42795</td>
      <td>1.5767</td>
      <td>5.419910</td>
      <td>0.000035</td>
      <td>1.572992</td>
      <td>58.1900</td>
      <td>36.0225</td>
      <td>57.6002</td>
      <td>57.6122</td>
      <td>729.0902</td>
      <td>34.782</td>
      <td>2.174</td>
    </tr>
    <tr>
      <td>4.0</td>
      <td>57060</td>
      <td>1.4953</td>
      <td>7.541308</td>
      <td>0.000028</td>
      <td>1.551169</td>
      <td>58.6074</td>
      <td>36.3093</td>
      <td>58.0192</td>
      <td>58.0383</td>
      <td>724.4636</td>
      <td>35.004</td>
      <td>2.188</td>
    </tr>
    <tr>
      <td>5.0</td>
      <td>71325</td>
      <td>1.4338</td>
      <td>4.890110</td>
      <td>0.000021</td>
      <td>1.540994</td>
      <td>58.5639</td>
      <td>36.4092</td>
      <td>57.9490</td>
      <td>57.9669</td>
      <td>726.4090</td>
      <td>34.910</td>
      <td>2.182</td>
    </tr>
    <tr>
      <td>6.0</td>
      <td>85590</td>
      <td>1.3857</td>
      <td>7.108903</td>
      <td>0.000014</td>
      <td>1.536733</td>
      <td>58.6788</td>
      <td>36.4511</td>
      <td>58.0754</td>
      <td>58.0852</td>
      <td>725.5978</td>
      <td>34.949</td>
      <td>2.184</td>
    </tr>
    <tr style="text-decoration: underline;">
      <td>7.0</td>
      <td>99855</td>
      <td>1.3504</td>
      <td>5.902568</td>
      <td>0.000007</td>
      <td>1.536227</td>
      <td>58.8399</td>
      <td>36.6104</td>
      <td>58.2366</td>
      <td>58.2533</td>
      <td>722.5710</td>
      <td>35.096</td>
      <td>2.194</td>
    </tr>
    <tr>
      <td>8.0</td>
      <td>114120</td>
      <td>1.3269</td>
      <td>6.621144</td>
      <td>0.000000</td>
      <td>1.540336</td>
      <td>58.8344</td>
      <td>36.5932</td>
      <td>58.2187</td>
      <td>58.2316</td>
      <td>723.1026</td>
      <td>35.070</td>
      <td>2.192</td>
    </tr>
  </tbody>
</table>

Graph showing RougeL metric during each epoch

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/rougeL.png?raw=true)


Graph of comparison between models for each rouge metric

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/output.png?raw=true)

Graph of comparison between rouge score distributions

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/output2.png?raw=true)

Graph showing average score for each metric for base and fine-tuned model

![Alt text](https://github.com/smartcat-labs/product2query/blob/main/images/avg_score.png?raw=true)

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
