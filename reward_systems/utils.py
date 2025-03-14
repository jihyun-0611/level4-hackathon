import json
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
import torch


def load_data(data_path):
  with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  df = pd.DataFrame(data)

  df['target'] = df['OFF'].astype(int)
  df = df.rename(columns={'comment': 'text'})
  df = df[['text', 'target']]
  df = df.reset_index().rename(columns={'index': 'idx'})

  train_df, val_df = train_test_split(
      df,
      test_size=0.2,
      random_state=42,
      stratify=df['target']
      # 계층적 데이터 추출 옵션 (분류 모델에서 추천!): 여러 층으로 분할후 각 층별로 렌덤 데이터 추출, 원래 데이터의 분포와 유사하게 데이터 추출
  )
  train_df = train_df.reset_index(drop=True)
  val_df = val_df.reset_index(drop=True)
  return train_df, val_df

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)

  f1 = evaluate.load('f1')
  accuracy = evaluate.load('accuracy')
  precision = evaluate.load('precision')
  recall = evaluate.load('recall')

  return {
      'accuracy': accuracy.compute(predictions=predictions, references=labels)["accuracy"],
      # macro : 각 클래스별로 점수를 계산한 후 산술 평균을 구함
      'f1': f1.compute(predictions=predictions, references=labels, average='macro')["f1"],
      'precision': precision.compute(predictions=predictions, references=labels, average='macro')["precision"],
      'recall': recall.compute(predictions=predictions, references=labels, average='macro')["recall"]
  }

def get_predictions_and_confidences(trainer, dataset):
  predictions = trainer.predict(dataset)
  logits = predictions.predictions
  labels = predictions.label_ids
  probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

  confs, preds = torch.max(probs, dim=-1)

  texts = dataset['text']
  idxs = dataset['idx']
  return texts, preds.numpy(), confs.numpy(), labels, idxs