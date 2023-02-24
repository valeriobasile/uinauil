#!/usr/bin/env python

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import os
import uinauil
import sys

benchmark = uinauil.Task("textualentailment")

# Shape the data like simpletransformers wants it
train_df = pd.DataFrame([[item["text1"], item["text2"], item["label"]] for item in benchmark.data.training_set])
train_df.columns = ["text_a", "text_b", "labels"]

# Model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = [0, 1]
model_args.train_batch_size = 16
model_args.lr = 1e-5
model_args.overwrite_output_dir = True
model_args.use_multiprocessing=False
model_args.use_multiprocessing_for_evaluation=False
model_args.use_cuda = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a ClassificationModel
model = ClassificationModel(
#    "bert", "dbmdz/bert-base-italian-xxl-cased", args=model_args
    "bert", "dbmdz/bert-base-italian-cased", args=model_args, use_cuda=False
#    "bert", "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", args=model_args
)

# Train the model
model.train_model(train_df)

# Make predictions with the model
predictions, raw_outputs = model.predict([[item["text1"], item["text2"]] for item in benchmark.data.test_set])

original_stdout = sys.stdout
with open('../predictions/textualentailment_predictions.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(predictions)
    sys.stdout = original_stdout


scores = benchmark.evaluate(predictions)
print(scores)
