# %%
import random
import time
from typing import Optional, Dict, List, Tuple, Iterable, Sized
from copy import deepcopy
import gc
import re

from transformers import AutoTokenizer

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torch.utils.data import Dataset, Subset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Зафиксируем зерна
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# %% [markdown]
# # 1. PyTorch Lightning
# 
# 

# %% [markdown]
# Pytorch Lightning - это высокоуровневое API для PyTorch, упрощающее разработку и распределённое обучение моделей.
# 
# Есть 2 основных паттерна использования этого API:
# 1. Через базовые классы LightningModule, LightningDataModule и Trainer;
# 2. Через фабрику lightinig.Fabric;
# 
# Рассмотрим оба варианта.

# %% [markdown]
# | Свойство           | Classic Lightning | Lightning Fabric |
# | ----------------- | --- | --- |
# | **Цикл обучения** | Пользователь определяет `training_step` и передает необходимые callbacks в `Trainer` | Полностью определяется пользоваетелем |
# | **Отличия в коде**  | Требует наследования класса модели от `LightningModule` и реализации всех необходимых методов | Минимальны - нужно только обернуть основные объекты |
# | **Гибкость**   | Меньше гибкости, `Trainer` управляет всем | Такая же, как в PyTorch |
# | **Когда использовать?**  | Быстрое прототипирование, стандартные задачи | Когда нужна гибкость или нужно масштабировать обучение моделей в существующем проекте |

# %%
from typing import Any
import lightning as L

class SimpleFFN_Classifier(nn.Module):

    def __init__(self, input_dim, n_classes, hidden_dim, n_hidden_layers=1, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim)] * n_hidden_layers, 
        )
        self.output = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.activation(self.input(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        probs = self.softmax(self.output(x))

        return probs

class WrappedModel(L.LightningModule):

    def __init__(self, input_dim, n_classes, hidden_dim, n_hidden_layers=1, activation=nn.ReLU(), lr=1e-3):

        self.save_hyperparameters()
        self.classifier = SimpleFFN_Classifier(input_dim, n_classes, hidden_dim, n_hidden_layers=1, activation=nn.ReLU())
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def training_step(self, batch, batch_idx, loader_idx=0):
        x, labels = batch
        probs = self.classifier(x)

        loss = self.loss_fn(probs, labels)

        return loss
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.hparams.get('lr', 3e-4))
    
model = WrappedModel(10, 10, 16, 4)

trainer = L.Trainer(
    fast_dev_run=1,
    max_time={},
)

trainer.fit(
    model, 
    train_dataloaders=None,
    val_dataloaders=None,
    datamodule=None,
    ckpt_path=None,
    )

# %%
# train on 4 GPUs
trainer = L.Trainer(
    devices=4,
    accelerator="gpu",
)

# 20+ helpful flags for rapid idea iteration
trainer = L.Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
)

# access the state of the art techniques
from lightning.pytorch.callbacks import StochasticWeightAveraging
trainer = L.Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=3e-2)])

# %%
class DataModule(L.LightningDataModule):

    def prepare_data(self):
        # called once on 1 device
        df = pd.read_hdf('datasets/example.h5')
        self.data = Dataset()

    def setup(self, stage: str):
        # called on each device, 
        self.train, self.val, self.test = \
            torch.utils.data.random_split(self.data, [0.8, 0.1, 0.1])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test)

    def on_exception(self, exception: BaseException) -> None:
        pass

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)


# %%
dataset = Dataset()
fabric = L.Fabric()

model = SimpleFFN_Classifier()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model, optimizer = fabric.setup(model, optimizer)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        print(loss.data)


# %% [markdown]
# - **`Fabric(...)`**: Конструктор. Здесь вы настраиваете оборудование и стратегию.
#   - `accelerator`: 'cpu', 'cuda', 'mps', 'tpu', 'auto'.
#   - `devices`: Количество устройств, список ID устройств или 'auto'.
#   - `strategy`: 'ddp', 'deepspeed' и т.д.
#   - `precision`: '64', '32', '16-mixed', 'bf16-mixed'.
#   
# - **`fabric.setup(model, optimizer)`**: Оборачивает модель и оптимизатор. Должен быть вызван перед началом обучения.
#   
# - **`fabric.setup_dataloaders(...)`**: Оборачивает загрузчики данных для обработки распределенной выборки и автоматического размещения на устройствах.
#   
# - **`fabric.backward(loss)`**: Замена `loss.backward()`. Обрабатывает масштабирование градиентов для смешанной точности и синхронизацию для распределенного обучения.
#   
# - **`fabric.print(...)`**: Замена `print()`. Обеспечивает вывод сообщения только из основного процесса (ранг 0).
#   
# - **`fabric.launch(function, *args)`**: Опциональный способ запуска функции обучения. Обеспечивает правильный контекст распределенного выполнения кода.
#   
# - **`fabric.save(path, state)`**: Утилита для сохранения контрольных точек, правильно работающая в распределенной среде выполнения.
#   
# - **`fabric.all_gather(tensor)`**: Сбор тензора со всех процессов. Полезно для агрегирования метрик.

# %%
# Run on 2 GPUs with DDP strategy
fabric run main_fabric.py --accelerator=cuda --devices=2

# Run with 16-bit mixed precision
fabric run main_fabric.py --accelerator=cuda --devices=1 --precision=16-mixed

# Run on CPU
fabric run main_fabric.py --accelerator=cpu


# %% [markdown]
# # 2. Реализуем baseline обучения трасформера

# %%
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("google/mobilebert-uncased")
embeddings = deepcopy(model.embeddings)
# https://huggingface.co/docs/transformers/main_classes/tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
VOCAB_SIZE = tokenizer.vocab_size
print(f'Using tokenizer of')
print(f'Vocabulary size: {VOCAB_SIZE}')
del model

# %%
class RecipeDataset(Dataset):

    def __init__(
            self, 
            path: str,
            tokenizer: AutoTokenizer,
            source_columns: List[str],
            target_columns: List[str] = None,
            nrows = None,
            max_len = 256,
            padding_type = "max_length",
            ):

        self.source_columns = source_columns
        self.target_columns = target_columns if target_columns else source_columns

        columns = source_columns
        if target_columns:
            columns += target_columns
        
        data = pd.read_csv(path, usecols=list(set(columns)), nrows=nrows)

        self.src_ = []
        self.tgt_ = []
        self.tokenizer = tokenizer
        self.CLS_id = tokenizer.cls_token_id
        self.SEP_id = tokenizer.sep_token_id
        self.PAD_idx = tokenizer.pad_token_id
        self.max_len = max_len
        self.padding_type = padding_type

        for i in tqdm.trange(len(data)):
            row = data.iloc[i]
            src_text = self.process_row(row, source_columns)
            self.src_.append(src_text)
            
            if target_columns is not None:
                tgt_text = self.process_row(row, target_columns) 
                self.tgt_.append(tgt_text)
        
        if target_columns is None:
            self.tgt_ = self.src_
        
        self.size = len(self.src_)

    def process_row(self, row: pd.Series, columns: List[str]):
        """Processes a single recipe row from the DataFrame into a clean string."""
        entry_parts = []
        for col in columns:
            if pd.notna(row[col]):
                content = str(row[col])
                if content.startswith('[') and content.endswith(']'):
                    content = re.sub(r'["\\$$\\\\$$]', '', content)
                entry_parts.append(f'{col.replace("_", " ")}: {content}')
                entry_parts.append('\n')
        return ''.join(entry_parts[:-1])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> Tuple[str, str]:
        source = self.src_[idx]
        target = self.tgt_[idx]
        return source, target

    def single(self, idx):
        return self.collate_fn([(self.src_[idx], self.tgt_[idx])])

    def collate_fn(self, batch: List[Tuple[str, str]]) -> Dict[str, List]:
        
        sources = [f[0] for f in batch]
        targets = [f[1] for f in batch]
    
        source_enc = self.tokenizer(list(sources), max_length=self.max_len, 
                                    padding=self.padding_type, truncation=True, 
                                    return_tensors="pt").to(device=self.device)

        target_enc = self.tokenizer(list(targets), max_length=self.max_len,
                                    padding=self.padding_type, truncation=True, 
                                    return_tensors="pt").to(device=self.device)
        return {
            'input_ids': source_enc['input_ids'], 
            'input_mask': source_enc['attention_mask'],
            'input_lengths': source_enc['attention_mask'].sum(dim=-1),
            'labels': target_enc['input_ids'],
            'labels_mask': target_enc['attention_mask'],
            'labels_lengths': target_enc['attention_mask'].sum(dim=-1),   
            }

    def get_loaders(
            self,
            names: Optional[List[str]] = ['train', 'val'],
            ratios: Optional[List[float]] = [0.9, 0.1],
            shuffle: Optional[List[bool]] = [True, False],
            batch_size: int = 8,
            load_ratio: int = 1.0,
            **kwargs,
        ) -> Dict[str, DataLoader]:
        """
        Fetches several dataloaders from this dataset
        """ 

        indices = list(range(len(self)))
        i0 = 0
        dataloaders: Dict[str, DataLoader] = {}
        
        for name, part, shuff in zip(names, ratios, shuffle):
            part_len = int(len(indices) * part * load_ratio )
            subset = Subset(self, indices[i0: i0 + part_len])
            dataloaders[name] = DataLoader(subset, batch_size, shuff, collate_fn=self.collate_fn, **kwargs)
            i0 += part_len        
            
        return dataloaders

# %%
DATA_PATH = '../data/1_Recipe_csv.csv'
NSAMPLES = 48000

copy_dataset = RecipeDataset(DATA_PATH, tokenizer, ['description', 'ingredients'], None,\
                     nrows=NSAMPLES, padding_type="longest", device='cpu')

ingredients_dataset = RecipeDataset(DATA_PATH, tokenizer, ['recipe_title', 'description'], ['ingredients'],\
                     nrows=NSAMPLES, padding_type="longest", device='cpu')

recipe_dataset = RecipeDataset(DATA_PATH, tokenizer, ['recipe_title', 'description'], ['ingredients'],\
                     nrows=NSAMPLES, padding_type="longest", device='cpu')

# %%
# Реализуйте класс модели

class RecipeTransformer(L.LightningModule): ...

# %%
# Запустите обучение с помощью L.Trainer()

torch.set_float32_matmul_precision('medium')



