#!/usr/bin/env python

from simpletransformers.classification import ClassificationArgs, ClassificationModel
import pandas as pd
import os
import uinauil
import sys
import torch

print ("*"*20)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

config = {'task': "sentipolc", 'epochs': 1, 'learning_rate': 1e-4, 'model': "dbmdz/bert-base-italian-cased"}

benchmark = uinauil.Task(config['task'])

# Shape the data like simpletransformers wants it
train_df = pd.DataFrame([[item["text"], item["label"]] for item in benchmark.data.training_set])
train_df.columns = ["text", "labels"]
    
# Model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=config['epochs']
model_args.labels_list = list(train_df.labels.unique())
model_args.train_batch_size = 16
model_args.lr = config['learning_rate']
model_args.overwrite_output_dir = True
model_args.fp16 = True
model_args.use_cuda = True
model_args.use_multiprocessing=False
model_args.use_multiprocessing_for_evaluation=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a ClassificationModel
model = ClassificationModel(
    "bert", 
    config['model'],
    args=model_args, 
    num_labels=len(model_args.labels_list),
    use_cuda=True
)

# Train the model
model.train_model(train_df)

# Make predictions with the model
predictions, raw_outputs = model.predict([item["text"] for item in benchmark.data.test_set])

'''original_stdout = sys.stdout
with open('../predictions/sentipolc_predictions.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(predictions)
    sys.stdout = original_stdout
'''
scores = benchmark.evaluate(predictions)
print(scores)
