```python
## Importing Packages
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from gensim.corpora.textcorpus import TextCorpus
from gensim.test.utils import datapath
from gensim import utils

import numpy as np
import pandas as pd
import nltk
import glob
import os
import re

import sklearn
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200
%matplotlib inline
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-5-66d7b1d4d6df> in <module>
          1 ## Importing Packages
    ----> 2 from gensim.test.utils import common_texts, get_tmpfile
          3 from gensim.models import Word2Vec
          4 
          5 from gensim.corpora.textcorpus import TextCorpus
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\__init__.py in <module>
          3 """
          4 
    ----> 5 from gensim import parsing, corpora, matutils, interfaces, models, similarities, summarization, utils  # noqa:F401
          6 import logging
          7 
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\corpora\__init__.py in <module>
          4 
          5 # bring corpus classes directly into package namespace, to save some typing
    ----> 6 from .indexedcorpus import IndexedCorpus  # noqa:F401 must appear before the other classes
          7 
          8 from .mmcorpus import MmCorpus  # noqa:F401
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\corpora\indexedcorpus.py in <module>
         13 import numpy
         14 
    ---> 15 from gensim import interfaces, utils
         16 
         17 logger = logging.getLogger(__name__)
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\interfaces.py in <module>
         19 import logging
         20 
    ---> 21 from gensim import utils, matutils
         22 from six.moves import range
         23 
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\matutils.py in <module>
       1102 try:
       1103     # try to load fast, cythonized code if possible
    -> 1104     from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation
       1105 
       1106 except ImportError:
    

    ~\Anaconda3\envs\Word2Vec\lib\site-packages\gensim\_matutils.cp37-win_amd64.pyd in init gensim._matutils()
    

    AttributeError: type object 'gensim._matutils.array' has no attribute '__reduce_cython__'



```python
# import NLTK corpus
#nltk.download('brown')
from nltk.corpus import brown
sentences = brown.sents()
```


```python
# open text file
path = "C:\\Users\\alwer\\Desktop\\corpus.txt"
file = open(path, encoding = "utf8", errors='ignore')
text = file.read()
```


```python
# Tokenizing Corpus
sents = nltk.sent_tokenize(text)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-b6b34563d2f4> in <module>
          1 # Tokenizing Corpus
    ----> 2 sents = nltk.sent_tokenize(text)
    

    NameError: name 'nltk' is not defined



```python
# Building Your Model - CBOW
model_cbow = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4, iter=2)
model_cbow.save("word2vec_cbow.model")
```


```python
# Building Your Model - Skip-gram
model_skip = Word2Vec(sentences, size=100, window=10, min_count=1, workers=4, iter=2,sg=1)
model_skip.save("word2vec_skip.model")
```


```python
# Training Pre-Loaded Model Further
model_skip_train = Word2Vec.load("word2vec_skip.model")
model_skip_train.train(sents, total_examples=1, epochs=10)
```


```python
## Exploring Your Model
vector2 = model_cbow.wv['web']
print(vector2)
vector1 = model_skip.wv['web']
print(vector1)
vector = model_skip_train.wv['web']
print(vector)
```


```python
# Similar Words
print(model_skip.similarity('computer', 'human'))
print(model_skip.similarity('dog', 'human'))
```


```python
# Similar words
similar_words = {search_term: [item[0] for item in model_skip.wv.most_similar([search_term], topn=5)]
                  for search_term in ['god', 'table', 'computer', 'human', 'dog', 'plant', 'love','hate']}
similar_words
```


```python
model_cbow.doesnt_match("woman ovarian brain".split())
```


```python
model_cbow.most_similar(positive=["author"])
```


```python
# https://github.com/laurenfklein/emory-qtm340/blob/master/notebooks/class12-word-vectors-complete.ipynb

# similarity b/t two words

print(ccp_model.wv.similarity(w1="freedom",w2="justice"))
print(ccp_model.wv.similarity(w1="freedom",w2="dinner"))
```


```python
# analogies
# format is: "man is to king as woman is to ???"

ccp_model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
```


```python
# Visualize Similar Words
wvs = model_skip.wv[words]
words = sum([[k] + v for k, v in similar_words.items()], [])

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)

labels = words

plt.figure(figsize=(14, 8))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
```


```python
# Tensorboard Projections

from tensorboard.plugins import projector

weights     = model_skip.wv.vectors
index_words = model_skip.wv.index2word

vocab_size    = weights.shape[0]
embedding_dim = weights.shape[1]

print('Shape of weights:', weights.shape)
print('Vocabulary size: %i' % vocab_size)
print('Embedding size: %i'  % embedding_dim)

with open(os.path.join(MODEL_DIR,'metadata.tsv'), 'w') as f:
    f.writelines("\n".join(index_words))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embeddings'
embedding.metadata_path = './metadata.tsv'
projector.visualize_embeddings(MODEL_DIR, config)

tensor_embeddings = tf.Variable(model.wv.vectors, name='embeddings')

checkpoint = tf.compat.v1.train.Saver([tensor_embeddings])
checkpoint_path = checkpoint.save(sess=None, global_step=None, save_path=os.path.join(MODEL_DIR, "model.ckpt"))
```


```python
# Import Documents into Corpus
file_list = glob.glob(os.path.join(os.getcwd(),"C:\\Users\\alwer\\Desktop\\test", "*.txt"))

corpus = []

for file_path in file_list:
    with open(file_path, encoding = "utf8", errors='ignore') as f_input:
        corpus.append(f_input.read())
        
print(corpus[:2])  
```


```python
# Change File List to Title - to work on
my_dir = "C:\\Users\\alwer\\Desktop\\test"
filelist = []
filesList = []
os.chdir( my_dir )

# Step 2: Build up list of files:
for files in glob.glob("*.txt"):
    fileName, fileExtension = os.path.splitext(files)
    filelist.append(fileName) #filename without extension
```


```python
# Import Documents into Dataframe
corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Title': filelist, 'Text': corpus})
corpus_df = corpus_df[['Title', 'Text']]
corpus_df
```


```python

```


```python
## Word Embeddings of Multiple Documents
```


```python
# Set values for various parameters
feature_size = 100    # Word vector dimensionality  
window_context = 10          # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words

w2v_model = Word2Vec(corpus, size=feature_size, 
                          window=window_context, min_count=min_word_count,
                          sample=sample, iter=50, sg=1)
```


```python
# Visualize Document Corpus Embeddings
from sklearn.manifold import TSNE

words = w2v_model.wv.index2word
wvs = w2v_model.wv[words]

tsne = TSNE(n_components=2, random_state=0, n_iter=250, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
```


```python
def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector
    
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


# get document level embeddings
w2v_feature_array = averaged_word_vectorizer(corpus=corpus, model=w2v_model,
                                             num_features=feature_size)
pd.DataFrame(w2v_feature_array)
```


```python
from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation()
ap.fit(w2v_feature_array)
cluster_labels = ap.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)
```


```python

```


```python
# Import GLoVe

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('test_glove.txt')
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file
```


```python
## KeyedVectors

from gensim.models import KeyedVectors
path = get_tmpfile("wordvectors.kv")
model.wv.save(path)
model.wv.save("model.wv")
```


```python
filename = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```


```python
model = KeyedVectors.load_word2vec_format(filename, binary=True)
```


```python
wv = KeyedVectors.load("model.wv", mmap='r')
```


```python
vector = wv['computer']
```


```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = "glove.txt"
word2vec_output_file = "word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)
m = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
```


```python
# Further Cleaning 
import numpy as np

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(sents):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

sents_clean = np.vectorize(normalize_document)
```


```python

```


```python
# Doc2Vec
```


```python
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_df)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
```


```python
from gensim.test.utils import get_tmpfile

fname = get_tmpfile("my_doc2vec_model")

model.save(fname)
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
```


```python
vector = model.infer_vector(["system", "response"])
print(vector)
```


```python
##https://praveenbezawada.com/2018/01/25/document-similarity-using-gensim-dec2vec/

class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])
 
docLabels = list(corpus_df['Title'])
data = list(corpus_df['Text'])
sentences = TaggedDocumentIterator(data, docLabels)
```


```python
model = Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)
```


```python
# Store the model to mmap-able files
model.save('/tmp/model_docsimilarity.doc2vec')
# Load the model
model = Doc2Vec.load('/tmp/model_docsimilarity.doc2vec')
```


```python
# Open Text File
import io
with io.open("corpus.txt",'r',encoding='utf8',errors='ignore') as f:
   text = f.read()
```


```python
def test_predict():
    #Select a random document for the document dataset
    rand_int = np.random.randint(0, corpus_df.shape[0])
    print ('Random int {}'.format(rand_int))
    test_corpus_df = corpus_df.iloc[rand_int]['info']
    label = corpus_df.iloc[rand_int, corpus_df.columns.get_loc('problemReportId')]
 
    #Clean the document using the utility functions used in train phase
    test_corpus_df = default_clean(test_corpus_df)
    test_corpus_df = stop_and_stem(test_corpus_df, stem=False)
 
    #Convert the corpus_df document into a list and use the infer_vector method to get a vector representation for it
    new_doc_words = test_corpus_df.split()
    new_doc_vec = model.infer_vector(new_doc_words, steps=50, alpha=0.25)
 
    #use the most_similar utility to find the most similar documents.
    similars = model.docvecs.most_similar(positive=[new_doc_vec])
test_predict()
```
