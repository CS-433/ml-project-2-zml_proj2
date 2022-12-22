# ml-project-2-zml_proj2
# Mining Effective Words For Climate Change Communication


## Team
The project is accomplished by team **ZML** with members:

Lazar Milikic: @Lemmy00

Yurui Zhu: @ruiarui

Marko Lisicic: @mrfox99

## Project Outline

  In order to garner more effective attention on Twitter for the topic of climate change, our project aims to design and implement interpretable models to predict tweet engagement, and then to identify words, phrases, and visual appeals (e.g., image, video, hashtag, etc.) that could increase engagement by interpreting the learned parameters. The accuracy of our models was approximately 60\%, and we were able to identify patterns in the most engaging words and phrases, which we believe can be used as a strategy when composing new tweets that aim to draw attention to climate change.


## Guideline

To get all the data needed to run the code, please contact Aswin Suresh(aswin.suresh@epfl.ch) for access. You can copy the entire data folder and put it the same level as src folder.



To run model, 


## Code Structure
```
├── implementaions.py: Implementations of 6 ML function
├── run.py: Python file for regenerating our final/best prediction.
├── preprocessing.py: Preprocessing pipeline, including filling missing values, build sin/cos and polynomial feature expansion
├── utils.py: All utils functions such as MSE and cross-entropy, gradient calculation function, K-cross validation etc.
├── helper.py: Functions to help load the data, generate mini-batch iterator and create *.csv* files for submission.
├── cross_vaildation.py: Python file for running cross validaion to find the best hyper-parameter.
├── Exploratory.ipynb: Data exploratory analysis, where it contains plots and conclusions to support our preprocessing and model selection.
├── fine_tuning.ipynb: fine tuning the model and plot the result
├── learning_curves.ipynb: plot the learning curves for Rigde and logistic regression.
├── CS_433_Project_2.pdf: a 4-pages report of the complete solution.
├── README.md
└── Data
```


