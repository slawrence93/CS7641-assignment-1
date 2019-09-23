# Instructions for running code

1. clone the repo
2. make sure you have the most recent version of python 3 installed
3. install necessary libraries from requirements.txt: `pip install requirements.txt`
4. install most recent non-stable version of sklearn: `pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn` (this step is important because my code uses methods that are not available in latest stable build)
5. run supervised learning experiments: `python experiments.py -all`. this command will run experiments for all of the classifiers and show the various plots as well as print out the final accuracy scores for each classifier and dataset.

if you only want to run an experiment for one classifier pass the following arguments:

`-t`: Decision Tree
`nn`: Neural Network
`svm`: Support Vector Machines
`knn`: k-Nearest Neighbor
`b`: Boosting