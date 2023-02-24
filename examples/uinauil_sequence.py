from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
import json
import os
import uinauil
import sys


benchmark = uinauil.Task("facta")

# TODO move the pandas logic into the library

# Shape the data like simpletransformers wants it
train_data = []
for item in benchmark.data.training_set:
    for token, label in zip(item["tokens"], item["labels"]):
        train_data.append([
            item["id"],
            token,
            label
        ])
train_df = pd.DataFrame(train_data)
train_df.columns = ["sentence_id", "words", "labels"]

# Model configuration
model_args = NERArgs()
model_args.num_train_epochs=1
model_args.labels_list = list(train_df.labels.unique())
model_args.train_batch_size = 16
model_args.overwrite_output_dir = True
model_args.lr = 1e-4
model_args.use_multiprocessing=False
model_args.use_multiprocessing_for_evaluation=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a ClassificationModel
model = NERModel(
#    "bert", "dbmdz/bert-base-italian-xxl-cased", args=model_args
    "bert", "dbmdz/bert-base-italian-cased", args=model_args, use_cuda=False
#    "bert", "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", args=model_args
)

# Train the model
model.train_model(train_df)

# Shape the test data like simpletransformers wants it
test_data = []
for item in benchmark.data.test_set:
    for token, label in zip(item["tokens"], item["labels"]):
        test_data.append([
            item["id"],
            token,
            label
        ])
test_df = pd.DataFrame(test_data)
test_df.columns = ["sentence_id", "words", "labels"]

# Make predictions with the model
predictions, raw_outputs = model.predict([" ".join(item["tokens"]) for item in benchmark.data.test_set])

original_stdout = sys.stdout
with open('./facta_predictions.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(json.dumps(predictions))
    sys.stdout = original_stdout

scores = benchmark.evaluate(predictions)
print(scores)
