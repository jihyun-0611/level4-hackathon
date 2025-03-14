from tqdm import tqdm

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from utils import *
from dataset import KOLDDataset

import random
import numpy as np
import os

def train(run_name, device, data_path, model_name, output_dir, pretrained=None, SEED=42):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

  train_df, val_df = load_data(data_path)
  train_dataset = KOLDDataset(train_df, tokenizer)
  val_dataset = KOLDDataset(val_df, tokenizer)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  training_args = TrainingArguments(
      output_dir=output_dir,
      report_to=["wandb"],  # wandb 사용 설정
      run_name=run_name,
      overwrite_output_dir=True,
      do_train=True,
      do_eval=True,
      do_predict=True,
      logging_strategy='steps',
      eval_strategy='steps',
      save_strategy='steps',
      logging_steps=100,
      eval_steps=100,
      save_steps=100,
      save_total_limit=2,
      learning_rate=2e-05,
      adam_beta1=0.9,
      adam_beta2=0.999,
      adam_epsilon=1e-08,
      weight_decay=0.01,
      lr_scheduler_type='linear',
      per_device_train_batch_size=32,
      per_device_eval_batch_size=32,
      num_train_epochs=2,
      load_best_model_at_end=True,
      metric_for_best_model='eval_f1',
      greater_is_better=True,
      seed=SEED
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      data_collator=data_collator,
      compute_metrics=compute_metrics
  )

  trainer.train()

def main():
  # set seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  base_dir = ''
  data_path = os.path.join(base_dir, 'dataset/kold_data.json')
  output_dir = os.path.join(base_dir, 'output')

  os.makedirs(output_dir, exist_ok=True)

  model_name = 'klue/bert-base'
  run_name = 'model/' + model_name + '/data-base'

  train(run_name, device, data_path, model_name, output_dir)

if __name__ == "__main__":
  main()



