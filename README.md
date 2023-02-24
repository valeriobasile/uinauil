# UINAUIL

**UINAUIL** (*Unified Interactive Natural Understanding of the Italian Language*) is a Python module for downloading language resources in order to train, evaluate, and analyze natural language understanding systems for the Italian Laguage, inspired by the [GLUE](https://gluebenchmark.com/) and [SuperGLUE](https://super.gluebenchmark.com/) benchmarks. It is based on the [European Language Grid](https://www.european-language-grid.eu/) (**ELG**) platform, that provides access to Language Technology services and resources from all over Europe. 


## Installation

The `uinauil` module can be installed via pip / PyPI with:

    pip install uinauil

Otherwise, you can directly download the [source file](src/uinauil.py) and save it on your working folder.

### Dependencies
The project dependencies are:

* python 3.6.7
* elg 0.4.22
* scikit_learn 1.2.1

You can install the project requirements with:

    pip install -r requirement.txt

## Quick start

Here the **basic functionalities** of the `uinauil` package: 
1. select a task;
2. download training e test set of the task;
3. evaluate the predictions of a model on the standard metrics of the task.

**Note**: the `uinauil` package **does not contain models**, but only datasets for *training* your models on common tasks.

    import uinauil as ul

    # load a task, for example 'facta'
    task = ul.Task('facta')

    # get training and test set of the task
    train = task.data.training_set   # training set
    test = task.data.test_set        # test set

    # train your model on the training set and make prediction on test set
    ...
    predctions = <make predictions on test set>

    # evaluate your model on the standard metrics of the task
    scores = task.evaluate(predictions)
    print(scores)

You can see an example of this code on the [Quick Start](examples/01_Quick_start.ipynb) notebook.


## Usage

Here we introduce all the functionalities of the `uinauil` package.

First of all, import the package with:

    import uinauil as ul

For getting the **list of available tasks** use:

    ul.tasks

`tasks` contains a dictionary of the available tasks ([here](#Details-on-tasks) the list with a brief presentation). Each **key** is the name of the task, while the **value** contains its *identifier* on the [ELG platform](https://www.european-language-grid.eu/) and the *type* of task.

### Select a task

You can select a task (for example `'facta'`) using the `Task` method:

    task = ul.Task('facta')

This instruction downloads the task data on your local computer into the `./data` folder. In order to download the resources you need to use your [ELG](https://www.european-language-grid.eu/) account.

The `task` object contains 4 instance variables:
* `task_name`: name of the task (the same used as key in `ul.tasks`);
* `desc`: description of the task from the [ELG platform](https://www.european-language-grid.eu/);
* `link`: link to the original resource on the [ELG platform](https://www.european-language-grid.eu/);
* `data`: dataset of the task, the main content described in the next section.

### The `data` variable

The `data` variable of Task object contains the **dataset**, already divided into training and test sets:

    train = task.data.training_set   # training set
    test = task.data.test_set        # test set

Both training and test sets are retrieved in **JSON format** (a dictionary in Python). Each task uses a different set of keys, then the `data` variable is also provided with several variables containing useful metadata:
* `data.keys`: list of all the keys in the dictionary;
* `data.feature_keys`: list of keys of the features;
* `data.target_key`: key of the target;
* `data.feature_dim`: list of dimensionality of each feature (single or multiple), in the order of `data.feature_keys`;
* `data.target_dim`: dimensionality of target (single or multiple);
* `data.feature_info`: list of brief descriptions of the meaning of each feature, in the order of `data.feature_keys`;
* `data.target_values`: list of possible values of the target;
* `data.target_desc`: list of brief descriptions of the meaning of each possible value of the target, in the order of `data.target_values`.

You can see a sample of all the available metadata or each task on the [Metadata](examples/02_Metadata.ipynb) notebook.

### Train a model

Sorry, `uinaiul` does not contain any pre-trained model. You have to use an **external package** for *building* a model, then you can *train* it using the training set of a task.

### Evaluate a model

The `evaluate` method of the task object takes a list of predictions on the test set and calculates the standard performance metrics on the task.

    scores = task.evaluate(predictions)
    print(scores)

The metrics have been chosen according to the **type** of task (described [here](#Details-on-tasks)):
* for `sequence` tasks, the method returns the *accuracy*, as the ratio of hits over all the tokens;
* for `classification` tasks, the method returns *accuracy* on all classes, and *precision*, *recall* and *F1* for each single class and their macro average;
* for `pairs` tasks, the method returns the [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) from [scikit-learn](https://scikit-learn.org/stable/index.html).

## Details on tasks

The `uinauil` module contains **6 tasks**, divides into the following **3 types**:

* `sequence` type, where the features set is composed by a *list of tokens*. It consists of the following **2 tasks**:
    * `facta`: the [FACTA](https://live.european-language-grid.eu/catalogue/corpus/8045) dataset consists of 169 news stories selected from the Ita-TimeBank, 120 Wikinews articles, and 301 tweets, annotated with event **factuality information**. 
    * `eventi`: the [EVENTI](https://live.european-language-grid.eu/catalogue/corpus/7376) corpus collects news articles and stories annotated with **temporal information** at different levels (i.e. events, temporal expressions, signals and temporal relations) following the [It-TimeML Annotation Guidelines](https://sites.google.com/site/eventievalita2014/file-cabinet).
* `classification` type, where the features set is composed by a *single text*. It consists of the following **3 tasks**:
    * `haspeede`: the [HaSpeeDe2](https://live.european-language-grid.eu/catalogue/corpus/7498) dataset collects 8,012 tweets and 500 news headlines annotated for the presence of **hate speech**, stereotypes and nominal utterance.
    * `ironita`: the [IronITA](https://live.european-language-grid.eu/catalogue/corpus/7372) dataset collects 4,849 tweets annotated for **irony and sarcasm**.
    * `sentipolc`: The [SENTIPOLC 2016](https://live.european-language-grid.eu/catalogue/corpus/7479) dataset contains 9410 tweets annotated for **subjectivity**, overall and literal polarity, and irony.
* `pairs` type, where the features set is composed by a *pair of texts*. It consists of the following **task**:
    * `textualentailment`: the [Textual Entailment](https://live.european-language-grid.eu/catalogue/corpus/8121) dataset contains 800 pairs of Italian sentences, extracted from Wikipedia, and annotated for the presence of **textual entailment**. 


## Working examples

The `examples` folder in this project contains several examples of use of the `uinauil` module:
* [**Quick Start**](examples/01_Quick_start.ipynb): presents the main feature of `uinauil`;
* [**Metadata**](examples/02_Metadata.ipynb): shows the values of the metadata for each task.


## Leaderboard

As a baseline we used the training and test sets of each task to train and evaluate **common ML models**. The results of the best models are summarized in the following table:

| Rank | Creator    | Model Name           | AVG  | facta | eventi | haspeede | ironita | sentipolc | textualentailment |
| ---- | --------   | -------------------- | ---  | ------| ------ | -------- | ------- | --------- | ----------------- |
| 1    |            | Italian BERT XXL     |.769 | .908 | .936 | .791 | .765 | .675 | .541 | 
| 2    |            | Italian BERT         | .755 | .907 | .916 | .785 | .736 | .646 | .538 | 
| 3    |            | ALBERTO              | .744 | .909 | .925 | .741 | .742 | .621 | .529 | 
| 4    |            | Multilingual BERT    |.731 | .909 | .925 | .739 | .709 | .559 | .544 | 

The leaderboards of the models on each sigle task are available on the [Leaderboards](leaderboards.md) file.


### Include a new model

If you want to add your model to the ranking, please contact [Valerio Basile](https://valeriobasile.github.io/).

## Authors

The `uinauil` package has been created by [Valerio Basile](https://valeriobasile.github.io/), with the collaboration of [Alessio Bosca](https://it.linkedin.com/in/alessio-bosca) and [Livio Bioglio](https://www.studium.unito.it/do/docenti.pl/Show?_id=lbioglio#tab-profilo).


## Citation

If you use `uinauil` in a scientific publication, we would appreciate citations of our paper:
```bibtex
@inproceedings{???,
    title = "???",
    author = "???",
    booktitle = "???",
    year = "202x",
    publisher = "???",
    pages = "???",
    url = "???"
}
```


## License

[Apache License 2.0](LICENSE)
