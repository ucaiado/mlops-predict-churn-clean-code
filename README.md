Predict Customer Churn with Clean Code
================


This project is part of the [ ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)
 program from Udacity. I developed a model to identify credit card customers
 that are most likely to churn, following coding (PEP8) and engineering best
 practices for implementing software (modular, documented, and tested) and the
 package also have the flexibility of being run interactively or from the
 command-line interface (CLI).


### Install
To set up your environment to run the code in this repository, start by
 installing Docker in your machine. Then, start Docker Desktop and run.

```shell
$ make docker-build
```


### Run
In a terminal or command window, navigate to the top-level project directory
 `mlops-predict-churn-clean-code/` (that contains this README) and run the following
 command.

```shell
$ make tests
```

It will generate EDA plots in the `images/eda/` folder, store some results also
 as images in the `images/results/` directory, save the models trained in
 `models/`, and write the test logs in `logs/churn_library.log` file.


### License
The contents of this repository are covered under the [MIT License](LICENSE).

