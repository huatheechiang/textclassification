from datasets import Dataset, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
import json

from google.colab import drive
drive.mount('content-of-drive')
train_file_path = "path-to-train-file"
test_file_path = "path-to-test-file"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True) 
 
def json_to_pandas(path):
  dataset = []
  with open(path) as F:
    for l in F.readlines():
      l = json.loads(l)
      newL = dict()
      tmp = l['subreddit']
      if tmp == 'MarioMaker':
        newL['labels'] = 0
      elif tmp == 'Nerf':
        newL['labels'] = 1
      elif tmp == 'ukulele':
        newL['labels'] = 2
      else:
        newL['labels'] = 3
      newL['text'] = l['body']
      dataset.append(newL)

  dataset = pd.DataFrame.from_dict(dataset)
  print(dataset)
  return Dataset.from_pandas(dataset)

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    return result

global tokenized_dataset_test
global tokenized_dataset_train

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 4)
training_args = TrainingArguments(
  output_dir="output-directory",
  evaluation_strategy="steps", 
  num_train_epochs = 10, #set it to 1 for testing
  save_steps = 10000,
  learning_rate = 5e-6, #defaults to 5e-5
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
)

def classifySubreddit_train(train_file_path): 
  dataset_train = json_to_pandas(train_file_path)
  tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True)
  train_test_split = tokenized_dataset_train.train_test_split(test_size=0.1)
  train = train_test_split['train']
  eval = train_test_split['test']

  global trainer
  model.cuda()
  trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = eval,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
  )
  trainer.train()

def classifySubreddit_test(test_file_path):
  dataset_test = json_to_pandas(test_file_path)
  tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True)
  output = trainer.predict(tokenized_dataset_test)
  
  x = output[0]
  numList = []
  for i in x:
    out = np.argmax(i)
    numList.append(out)
  
  subList = []
  for i in numList:
    if i == 0:
      subList.append("MarioMaker")
    elif i == 1:
      subList.append("Nerf")
    elif i == 2:
      subList.append("ukulele")
    else:
      subList.append("newToTheNavy")

  return subList

