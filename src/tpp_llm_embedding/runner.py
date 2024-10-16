"""
TPP-LLM Embedding Runner
"""
from typing import Dict, Tuple, Union

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from tpp_llm_embedding.evaluation import evaluate_similarities


class TPPLLMEmbeddingRunner(object):
    """
    TPP-LLM Embedding Model Runner
    """

    def __init__(
        self, model: nn.Module, loss_fn: nn.Module, learning_rate: float, num_epochs: int, warmup_ratio: float,
        device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the TPP-LLM embedding model runner

        :param model: TPP-LLM embedding model
        :param loss_fn: loss function
        :param learning_rate: learning rate
        :param num_epochs: number of epochs
        :param warmup_ratio: ratio for warm-up steps
        :param device: device
        """
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device(device)

        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.global_step = 0
        self.scheduler = None

    def run_batch(self, batch: Dict[str, list], phase: str) -> Tuple[Tensor, Tensor]:
        """
        Run a batch of event sequences

        :param batch: a batch of event sequences
        :param phase: running phase (train or eval)
        :return: description embeddings, event sequence embeddings
        """
        # Move batch data to the device
        batch = {
            'time_since_start': [_seq.to(self.device) for _seq in batch['time_since_start']],
            'time_since_last_event': [_seq.to(self.device) for _seq in batch['time_since_last_event']],
            'type_event': [_seq.to(self.device) for _seq in batch['type_event']],
            'type_text': batch['type_text'],
            'description': batch['description'],
        }

        if phase == 'train':
            self.model.train()

            # Get the loss term
            desc_embeddings = self.model(inputs=batch['description'], input_type='text')
            seq_embeddings = self.model(inputs=(batch['time_since_start'], batch['type_text']), input_type='event')
            loss = self.loss_fn(desc_embeddings=desc_embeddings, seq_embeddings=seq_embeddings)
            metrics = {
                'loss': loss.float().cpu().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.global_step / self.num_training_steps * self.num_epochs,
            }
            print(metrics)

            # Optimize the model parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the learning rate
            self.scheduler.step()
            self.global_step += 1

            return desc_embeddings, seq_embeddings

        elif phase == 'eval':
            self.model.eval()

            # Get embeddings
            with torch.no_grad():
                desc_embeddings = self.model(inputs=batch['description'], input_type='text')
                seq_embeddings = self.model(inputs=(batch['time_since_start'], batch['type_text']), input_type='event')

            return desc_embeddings, seq_embeddings

        else:
            raise KeyError(f'Unknown phase: {phase}.')

    def run_epoch(self, dataloader: DataLoader, phase: str) -> dict:
        """
        Run an epoch with batches of event sequences

        :param dataloader: data loader
        :param phase: running phase (train or eval)
        :return: metrics
        """
        desc_embeddings = []
        seq_embeddings = []

        for batch in tqdm(dataloader):
            batch_desc_embeddings, batch_seq_embeddings = self.run_batch(batch=batch, phase=phase)
            desc_embeddings.append(batch_desc_embeddings)
            seq_embeddings.append(batch_seq_embeddings)

        desc_embeddings = torch.cat(desc_embeddings, dim=0)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)

        if phase == 'eval':
            metrics = evaluate_similarities(desc_embeddings=desc_embeddings, seq_embeddings=seq_embeddings)
        else:
            metrics = {}

        return metrics

    def run(
        self, dataloader_train: DataLoader = None, dataloader_val: DataLoader = None,
        dataloader_test: DataLoader = None) -> None:
        """
        Run the training, validation, and testing for the model

        :param dataloader_train: data loader of the training set
        :param dataloader_val: data loader of the validation set
        :param dataloader_test: data loader of the testing set
        """
        # Calculate the number of training steps and the number of warmup steps
        self.num_training_steps = self.num_epochs * len(dataloader_train)
        self.num_warmup_steps = int(self.warmup_ratio * self.num_training_steps)
        self.global_step = 0

        # Initialize the learning rate scheduler with warm-up
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=0.5,
        )

        # Initial validation
        metrics_val_best = None

        if dataloader_val:
            metrics_val = self.run_epoch(dataloader_val, phase='eval')
            print(f'validation results: {metrics_val}')
            metrics_val_best = metrics_val

        if dataloader_test:
            metrics_test = self.run_epoch(dataloader_test, phase='eval')
            print(f'test results: {metrics_test}')

        # Start training loop
        for epoch in range(self.num_epochs):
            print(f'epoch: {epoch}')

            if dataloader_train:
                metrics_train = self.run_epoch(dataloader_train, phase='train')
                print(f'train results of epoch {epoch}: {metrics_train}')

            if dataloader_val:
                metrics_val = self.run_epoch(dataloader_val, phase='eval')
                print(f'validation results of epoch {epoch}: {metrics_val}')

                if metrics_val['cosine_mrr'] > metrics_val_best['cosine_mrr']:
                    metrics_val_best = metrics_val
                    print(f'new best validation results')

            if dataloader_test:
                metrics_test = self.run_epoch(dataloader_test, phase='eval')
                print(f'test results of epoch {epoch}: {metrics_test}')
