"""
Train the TPP-LLM Embedding Model
"""
import argparse

import torch
import transformers
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

from tpp_llm_embedding.data import TPPLLMDataset, collate_fn
from tpp_llm_embedding.losses import TPPLLMMultipleNegativesRankingLoss
from tpp_llm_embedding.model import TPPLLMEmbeddingModel
from tpp_llm_embedding.runner import TPPLLMEmbeddingRunner

if __name__ == '__main__':
    # Set teh argument parser
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and evaluate the TPP-LLM embedding model with event sequences.')
    parser.add_argument(
        '--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm path')
    parser.add_argument(
        '--data_path', type=str, required=True, help='data path')
    parser.add_argument(
        '--temporal_emb_type', type=str, default='positional', choices=['positional', 'linear'],
        help='temporal embedding type')
    parser.add_argument(
        '--temporal_emb_first', action='store_true', help='temporal embedding first or not')
    parser.add_argument(
        '--embedding_mode', type=str, default='all', help='embedding mode for event sequence embeddings')
    parser.add_argument(
        '--pooling_mode', type=str, default='mean', help='pooling mode for the last hidden layer')
    parser.add_argument(
        '--quant_type', type=str, default=None, choices=['4bit', '8bit', None], help='quantization type')
    parser.add_argument(
        '--peft_type', type=str, default=None, choices=['lora', None], help='peft type')
    parser.add_argument(
        '--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument(
        '--lora_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        help='lora target modules')
    parser.add_argument(
        '--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument(
        '--eval_batch_size', type=int, default=16, help='batch size for evaluation')
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4, help='larning rate')
    parser.add_argument(
        '--warmup_ratio', type=float, default=0, help='warmup ratio')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument(
        '--device', type=str, default='cpu', help='cpu or cuda device')
    parser.add_argument(
        '--seed', type=int, default=0, help='seed for reproducibility')

    # Parse arguments
    args = parser.parse_args()
    print(f'arguments: {args}')
    transformers.set_seed(args.seed)

    # Get the quantization config
    if args.quant_type == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quant_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Get the PEFT config
    if args.peft_type == 'lora':
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        peft_config = None

    # Load the model
    model = TPPLLMEmbeddingModel(
        model_name=args.model_path,
        temporal_emb_type=args.temporal_emb_type,
        temporal_emb_first=args.temporal_emb_first,
        bnb_config=bnb_config,
        peft_config=peft_config,
        embedding_mode=args.embedding_mode,
        pooling_mode=args.pooling_mode,
        device=args.device,
    )
    print(f'model: {model}')

    # Load the dataset
    dataset_train = TPPLLMDataset(f'{args.data_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.data_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.data_path}/test.json')
    dataloader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Train and test the model
    runner = TPPLLMEmbeddingRunner(
        model=model,
        loss_fn=TPPLLMMultipleNegativesRankingLoss(),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
    )
    runner.run(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_test=dataloader_test,
    )
