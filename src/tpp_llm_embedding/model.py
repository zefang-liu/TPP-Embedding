"""
TPP-LLM Embedding Model
"""
from typing import List, Union

import torch
import torch.nn as nn
import transformers
from peft import get_peft_model, PeftConfig
from sentence_transformers import models
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from tpp_llm_embedding.layers import TimePositionalEncoding


class TPPLLMEmbeddingModel(nn.Module):
    """
    TPP-LLM Embedding Model
    """

    def __init__(
        self, model_name: str, temporal_emb_type: str, temporal_emb_first: bool = False, embedding_mode: str = 'both',
        hidden_state_mode: str = 'all', pooling_mode: str = 'mean', bnb_config: BitsAndBytesConfig = None,
        peft_config: PeftConfig = None, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the TPP-LLM Embedding model

        :param model_name: LLM name
        :param temporal_emb_type: temporal embedding type
        :param temporal_emb_first: temporal embedding first (before the text embedding) for each event
        :param embedding_mode: embedding model ("time" embedding, "type" embedding, or "both" embeddings)
        :param hidden_state_mode: hidden state mode ("last" embedding, "both" embeddings, or "all" embeddings)
        :param pooling_mode: pooling mode for the last hidden layer
        :param bnb_config: bits and bytes configuration
        :param peft_config: PEFT configuration
        :param device: device
        """
        super().__init__()

        # Load the LLM
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            device_map=self.device,
        )

        # Apply the PEFT config
        if peft_config is not None:
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.train()
            self.llm.print_trainable_parameters()
        else:
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False

        # Set the model parameters
        self.hidden_size = self.llm.config.hidden_size
        self.llm_embedder = self.llm.get_input_embeddings()
        self.embedding_dim = self.llm_embedder.embedding_dim
        self.embedding_mode = embedding_mode
        self.hidden_state_mode = hidden_state_mode
        self.pooling_mode = pooling_mode
        self.temporal_emb_type = temporal_emb_type
        self.temporal_emb_first = temporal_emb_first
        self.dtype = self.llm.dtype

        # Load the temporal embedding
        if self.temporal_emb_type == 'linear':
            self.temporal_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
        elif self.temporal_emb_type == 'positional':
            self.temporal_embedder = TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
        else:
            raise KeyError(f'Temporal embedding type {self.temporal_emb_type} not implemented.')

        # Create the pooling layer
        self.pooling = models.Pooling(self.hidden_size, pooling_mode=self.pooling_mode)

    def embed_texts(self, batch_texts: List[str]) -> Tensor:
        """
        Embed a batch of texts into vectors

        :param batch_texts: batch of texts
        :return: batch of text embeddings, (batch_size, hidden_size)
        """
        batch_tokens = self.tokenizer(
            batch_texts, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True,
            return_attention_mask=True,
        ).to(self.device)
        batch_hidden_states = self.llm(
            input_ids=batch_tokens['input_ids'], attention_mask=batch_tokens['attention_mask'],
        ).last_hidden_state
        batch_embeddings = self.pooling({
            'token_embeddings': batch_hidden_states, 'attention_mask': batch_tokens['attention_mask'],
        })['sentence_embedding']
        return batch_embeddings

    @torch.no_grad()
    def embed_event_text(self, event_texts: List[str], add_special_tokens: bool = False) -> List[Tensor]:
        """
        Embed the event texts

        :param event_texts: event texts
        :param add_special_tokens: add special tokens or not
        :return: token embeddings of event texts, [(text_token_len, embedding_dim), ...]
        """
        event_tokens = self.tokenizer(
            event_texts, return_tensors='pt', add_special_tokens=add_special_tokens,
            padding=True, truncation=False)
        nums_tokens = event_tokens['attention_mask'].to(self.device).sum(dim=-1)
        event_embeddings_padded = self.llm_embedder(event_tokens['input_ids'].to(self.device))
        event_embeddings = [
            event_embedding_padded[:num_tokens]
            for num_tokens, event_embedding_padded in zip(nums_tokens, event_embeddings_padded)]
        return event_embeddings

    def embed_event_sequences(self, batch_event_times: List[Tensor], batch_event_texts: List[List[str]]) -> Tensor:
        """
        Embed event sequences into vectors

        :param batch_event_times: batch of event times in sequences, [(seq_len,), ...]
        :param batch_event_texts: batch of event texts in sequences, [(seq_len,), ...]
        :return: batch of event sequence embeddings, (batch_size, hidden_size)
        """
        batch_sequence_embeddings = []
        batch_attention_masks = []
        batch_event_emb_indices = []

        # Process each event sequence in the batch
        for event_times, event_texts in zip(batch_event_times, batch_event_texts):
            sequence_embeddings = []
            sequence_attention_mask = []
            event_emb_indices = []

            # Get the temporal embeddings for event times
            temporal_embeddings = self.temporal_embedder(event_times.unsqueeze(-1))  # (seq_len, embedding_dim)

            # Get token embeddings for the event texts
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]

            # Process each event with event times and texts
            for temporal_embedding, event_token_embeddings in zip(temporal_embeddings, event_text_embeddings):
                if self.temporal_emb_first:
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.embedding_mode in ['time', 'both']:
                        sequence_embeddings.append(temporal_embedding)
                        sequence_attention_mask.append(1)
                        if self.hidden_state_mode == 'all' or self.hidden_state_mode == 'both':
                            event_emb_indices.append(len(sequence_embeddings) - 1)

                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.embedding_mode in ['type', 'both']:
                        for event_token_embedding in event_token_embeddings:
                            sequence_embeddings.append(event_token_embedding)
                            sequence_attention_mask.append(1)
                            if self.hidden_state_mode == 'all':
                                event_emb_indices.append(len(sequence_embeddings) - 1)

                        if self.hidden_state_mode == 'both':
                            event_emb_indices.append(len(sequence_embeddings) - 1)

                else:
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.embedding_mode in ['type', 'both']:
                        for event_token_embedding in event_token_embeddings:
                            sequence_embeddings.append(event_token_embedding)
                            sequence_attention_mask.append(1)
                            if self.hidden_state_mode == 'all':
                                event_emb_indices.append(len(sequence_embeddings) - 1)

                        if self.hidden_state_mode == 'both':
                            event_emb_indices.append(len(sequence_embeddings) - 1)

                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.embedding_mode in ['time', 'both']:
                        sequence_embeddings.append(temporal_embedding)
                        sequence_attention_mask.append(1)
                        if self.hidden_state_mode == 'all' or self.hidden_state_mode == 'both':
                            event_emb_indices.append(len(sequence_embeddings) - 1)

                # Record the index of the last embedding of this event
                if self.hidden_state_mode == 'last':
                    event_emb_indices.append(len(sequence_embeddings) - 1)

            # Convert sequence embeddings to tensors
            batch_sequence_embeddings.append(torch.stack(sequence_embeddings))  # [(seq_emb_len, embedding_dim), ...]
            batch_attention_masks.append(torch.tensor(sequence_attention_mask).to(self.device))  # [(seq_emb_len,), ...]
            batch_event_emb_indices.append(torch.tensor(event_emb_indices).to(self.device))  # [(seq_len,), ...]

        # Pad the sequence embeddings in the batch
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_sequence_embeddings, batch_first=True)  # (num_seqs, max_seq_emb_len, embedding_dim)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, batch_first=True)  # (num_seqs, max_seq_emb_len)

        # Pass the padded embeddings through the LLM
        llm_output = self.llm(
            inputs_embeds=padded_embeddings, attention_mask=padded_attention_masks,
        ).last_hidden_state  # (batch_size, max_seq_emb_len, hidden_size)

        # Collect the hidden states at the last event embedding positions
        batch_hidden_states = [
            llm_output[i, batch_event_emb_indices[i], :]
            for i in range(len(batch_event_emb_indices))
        ]  # [(seq_len, hidden_size), ...]

        # Pool hidden states
        batch_embeddings = torch.stack([
            self.pooling({'token_embeddings': hidden_states.unsqueeze(0)})['sentence_embedding'].squeeze()
            for hidden_states in batch_hidden_states
        ])  # (batch_size, hidden_size)

        return batch_embeddings

    def forward(self, inputs, input_type: str) -> Tensor:
        """
        Forward function to get embeddings of event sequences or texts (descriptions)

        :param inputs: batch of inputs (texts or event times with event texts)
        :param input_type: input type (text or event)
        :return: batch of embeddings
        """
        if input_type == 'text':
            outputs = self.embed_texts(batch_texts=inputs)
        elif input_type == 'event':
            outputs = self.embed_event_sequences(
                batch_event_times=inputs[0], batch_event_texts=inputs[1])
        else:
            raise KeyError(f'Input type {input_type} not supported.')
        return outputs
