#This is the final training step which is followed by implementation. Model gives an accuracy of ~83%.
#Further improvements required.


#!/usr/bin/env python
# coding: utf-8

# # Word and Flair embeddings

import torch

pip install flair==0.4.2

from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1:'ner'}
# this is the folder in which train, test and dev files reside
data_folder = 'preprocessed data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='training.newtagsf.txt',
                              test_file='evaluation.newtagsf.txt',
                              dev_file ='development.newtagsf.txt'
                              ).downsample(0.4)
    

print(corpus)




from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
# modeltrainer.train()
# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove.gensim'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward-0.4.1.pt'),
    FlairEmbeddings('news-backward-0.4.1.pt'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)


# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources16/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources16/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources16/taggers/example-ner/weights.txt')



# load the model you trained
# my model can be found at "https://drive.google.com/open?id=15vndpPliCRWfUDNgbO8Th4STLg9XsgPt"
model = SequenceTagger.load('resources16/taggers/example-ner/final-model.pt')

# create example sentence
sentence = sentence('this is nitric oxide and on combining with water it forms acid.')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())


# # PubMed+PMC+wikipedia-Word2Vec



pip install gensim




from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from gensim.models import Word2Vec

tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
# modeltrainer.train()
# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    #WordEmbeddings('glove.gensim'),
    Word2Vec('wikipedia-pubmed-and-PMC-w2v.bin'),
    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward-0.4.1.pt'),
    FlairEmbeddings('news-backward-0.4.1.pt'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
    
    