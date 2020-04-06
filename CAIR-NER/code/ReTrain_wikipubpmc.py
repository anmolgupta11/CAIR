# # Further training on wiki+pubmed+pmc Word2Vec


#!/usr/bin/env python
# coding: utf-8


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
from gensim.models import Word2Vec

tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
# modeltrainer.train()
# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    #WordEmbeddings('glove.gensim'),
    WordEmbeddings('wikipedia-pubmed-and-PMC-w2v.bin'),
    #WordEmbeddings('bio_embedding_intrinsic')
    
    #WordEmbeddings('bio_embedding_extrinsic')
    

]
    
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load('resources16/taggers/example-ner/final-model.pt')


# ReTrained old model from resources16


# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#trainer: model = SequenceTagger.load('resources16/taggers/example-ner/final-model.pt')
# 7. start training
trainer.train('resourcespub/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resourcespub/taggers/example-ner/loss.tsv')
plotter.plot_weights('resourcespub/taggers/example-ner/weights.txt')






