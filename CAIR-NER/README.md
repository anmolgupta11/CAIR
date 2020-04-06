# **Efficient Deep Learning Flair Architecture for Chemical Named Entity Recognition**

**Chemical named entity recognition (NER)** is an active field of research in biomedical natural language processing (NLP). My model, based on state of the art flair architecture, along-with the usage of Word2Vec model of word tokenization, achieves the best performance on the CHEMDNER dataset out of all the existing models.

**[Flair](https://github.com/flairNLP/flair)** proposes a new approach to address core natural language processing tasks such as part-of-speech (PoS) tagging, named entity recognition (NER), sense disambiguation and text classification. This architecture leverages character-level neural language modeling to learn powerful, contextualized representations of human language from large corpora.

## **Dataset**

Training and Evaluation have been performed on **[CHEMDNER corpus](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)**. The corpus contains ten thousand abstracts from eleven chemistry-related fields of science with over 84k manually annotated chemical entities (20k unique) of **eight types**:

```
1. ABBREVIATION (15.55%)
2. FAMILY (14.15%)
3. FORMULA (14.26%)
4. IDENTIFIER (2.16%)
5. MULTIPLE (0.70%)
6. SYSTEMATIC (22.69%)
7. TRIVIAL (30.36%)
8. NO CLASS (0.13%)
```


The CHEMDNER corpus comprises **three parts**: *training (3.5k)*, *development (3.5k)* and *testing(3k)* datasets.
Each of the three parts consists of 2 types of files — annotations file and the abstracts file. The abstracts file consists of sentences.
*The annotations file consists of 6 columns:*

```
1. Article identifier (PMID)
2. Type of text from which the annotation was derived (T: Title, A: Abstract)
3. Start offset
4. End offset
5. Text string of the entity mention or Chemical name
6. Type of chemical entity mention (ABBREVIATION, FAMILY ,FORMULA, IDENTIFIERS, MULTIPLE, SYSTEMATIC, TRIVIAL) or TAGS
```

## **Model**

A very simple, state of the art NLP framework **FLAIR** model is used for embedding and training our data. Flair is a very powerful NLP library which allows us to use powerful models like Name Entity Recognition (NER), POS tagging, sense disambiguation and classification.
I have used NER tagging for my dataset. Word Embeddings and Flair Embeddings of Flair architecture are also used for better results.

## **Results**

Training of model is done, without optmisation due to lack of resources like GPU. Also, the training has been done on 40% of the dataset.
Batch size is taken as 16, and max epochs = 150. Learning rate of 0.1 is taken. 

**Micro F1 score** on overall test dataset is **79.75%** with **Precision = 82.47%** and **recall = 77.2%**.

**For TAGS , the best F1 score came out to be 86.18% for TRIVIAL , with precision = 86.44% and recall = 85.93%. The lowest F1 score came out to be 34.71% for MULTIPLE tags, having precision = 63.64% and recall = 23.68%.**

## **Further Improvements**

As it can be seen, that there is no optimisation done and still a pretty high accuracy is reached. I believe that this model will surpass all the existing models on our dataset.
**Further Improvements are much appreciated.**

