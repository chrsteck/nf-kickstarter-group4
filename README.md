# nf-kickstarter-group4
Analysis of the Kickstarter dataset for the neuefische bootcamp can 21-1

by: Kay Delventhal, Christoph Michel and Christian Steck

This repo contains an Analysis of the Kickstarter dataset as well as two models to predicting the success of kickstarter campaigns. The first model predicts the success for general campaigns by their (sub-)categories and the goals. The second predicts the success of games campaigns in the us using a bag-of-words approach with the campaign descriptions. The following files are in the repo:

* **kickstarter_analysis.ipynb** - a notebook containing the analysis
* **model.py** - a script training the general model and saving the model as pickle and its summary as a text file or predicting campaign succes and calculating precision
* **savingthemodel_game.py** - a script training the gaming model and saving the model as pickle and its summary as a text file
* **MakeFile*** A make-file to install the requirements for the virtual environment
* **requirements.txt** - python virtual environment requirements
* **model** - a folder with the models and their summaries