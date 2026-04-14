# FOCaL

### Functional Outcome Causal Learning

The repo contains the code to perform function-on-scalar doubly-robust regression. The reader can find scripts for the following use:

1) **FOCaL**, a class object that allows to call the following functions:
    * *fit*: train the nuisance functions and compute pseudo-outcomes
    * *predict*: predict the functional outcomes for new data
    * *predict_Y*: retrieve per-observation pseudo outcomes under factual treatment A
    * *get_FATE*: get the estimated functional average treatment effect

2) **Simple example**, to get used to the estimator, with particular emphasis on fitting and predicting commands

3) **Simulations**, which provides the script to run the experiments that show the properties of the estimator, with particular emphasis on the double-robustness property

4) **Covid** and **Share** Data. Applications of the estimator on real data

--- 

The implemented methodology is described in the following paper:

- [A Doubly Robust Machine Learning Approach for Disentangling Treatment Effect Heterogeneity with Functional Outcomes](https://arxiv.org/abs/2602.11118)

--- 

### FILES DESCRIPTION

        
- *FOCaL.py*:
  file to run Functional Outcome Causal Learning 
  
- *simple_example.py*:
  file to run FOCaL on simple synthetic data (fit and predict)

- *simulations.py*:
  file to run FOCaL on synthetic data to evaluate model properties

- *app_covid.py*:
  file to run FOCaL on COVID data (epidemiological)

- *app_share.py*:
  file to run FOCaL on SHARE data (health, life quality)

- *data*:
  folder with data for COVID and SHARE applications

