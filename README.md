# ml-project-2-zml_proj2
# Mining Effective Words For Climate Change Communication


## Team
The project is accomplished by team **ZML** with members:

Lazar Milikic: @Lemmy00

Yurui Zhu: @ruiarui

Marko Lisicic: @mrfox99

## Project Outline

  In order to garner more effective attention on Twitter for the topic of climate change, our project aims to design and implement interpretable models to predict tweet engagement, and then to identify words, phrases, and visual appeals (e.g., image, video, hashtag, etc.) that could increase engagement by interpreting the learned parameters. 
  
We implenment three model: **BTM-init** for training only tweet text embedding, **BTM-meta** for training tweet text embedding together with meta-data binary labels, and **BTM-latent**, where the latent author vectors are taking into account.
  
 The accuracy of our models was approximately 60\%, and we were able to identify patterns in the most engaging words and phrases, which we believe can be used as a strategy when composing new tweets that aim to draw attention to climate change.


## Guideline

To get all the data needed to run the code, please contact Aswin Suresh(aswin.suresh@epfl.ch) for access. You can copy the entire data folder and put it the same level as src folder. The data folder contains following files:

```
├── src
├── data
│   ├── authors_weights.pickle: the latent author vector that are used for BTM-latent model training and interpretation
│   ├── bigram.pkl.bz2: all the bigrams and corresponding embeddings, for interpretation
│   ├── dictionary.pkl.bz2: all the words and corresponding embeddings, for interpretation
│   ├── embeddings_difference_meta.pickle: calculated difference of embeddings bweteen pairs, with meta data labels
│   ├── pairs10%.pkl.bz2: pairing result
│   ├── tweets_embd.pkl.bz2: tweets id and its corresponding embeddings
│   └── tweets.pkl.bz2: full dataset with raw 

```


The necessary libaries for running our code and notebooks are:

 - PyTorch : To train the model
 - Pandas 
 - NumPy
 - pickle : To save and load data 
 - nltk : To preprocess and tokenize the text
 - Emoji and emoji_translate : For translate emojis
 - fasttext : To calculate word embeddings
 - scikit-learn : For anlaysis needed like PCA, T-SNE
 - Seaborn : For visualzation
 - tqdm : For showing processing info
 - Other basic python libraries such as `re`,`json` etc.
 

 


Then you can use following command to run the code for training and the training with GPU take around 30-45 min:
```bash
cd src
# train BTM-init model 
python run.py Init 
# train btm-meta model
python run.py Meta
# train btm-latent model
python run.py Latnet 
# Model finetuning 
python fine-tunning.py [Init, Meta, Latnet] # depending on which model
```

To run model, 


## Project Structure

The code structure of our project is shown as followed:

```
├── Data analysis and feature extraction
│   ├── Analysis.ipynb: Exploratory data analysis
│   ├── Author vector.ipynb: generate author vector
│   └── data_loader.py: Read data from original json file, and inital save it into pickle that can be quickly read by python
├── Interpretation.ipynb : interpretation with single words
├── Interpretation_bigram.ipynb : interpretation with bigrams
├── Interpretation_tsne.ipynb : t-sne grouping and Visualization result
├── Models : trained model result for further analysis and interpretation
│   ├── TSNEresult.csv
│   ├── btm-inital-time10%.pth
│   ├── btm-latent-time10%.pth
│   └── btm-meta-time10%.pth
├── Word embedding.ipynb : create word embeddings and tweet embeddings 
├── data : this folder contains all the data we need and intermediate results
├── src
│   ├── data cleaning and feature extraction
│   │   └── create_pairs.py : generate tweets pairs 
│   ├── dataset.py : load data with pyTorch dataloader
│   ├── fine-tunning.py : model fintuning
│   ├── models.py : model defination
│   ├── run.py : run the experiments
│   └── training.py : define training process
├── CS_433_Project_2.pdf: a 4-pages report of the complete solution.
└── README.md
```




