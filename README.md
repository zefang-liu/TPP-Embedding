# \[NAACL'25 KnowledgeNLP\] TPP-Embedding: Temporal Event Sequence Retrieval

This repository contains the code and resources for **TPP-Embedding**, a model designed for retrieving temporal event sequences from textual descriptions. It integrates **[TPP-LLM](https://github.com/zefang-liu/TPP-LLM)** to generate shared embeddings for event sequences and descriptions, enabling efficient retrieval based on semantic and temporal information. For more details, please refer to our [paper](https://arxiv.org/abs/2410.14043).

## Features

- **Event Time and Type Embeddings:** Representing temporal and semantic information for each event in the sequence.
- **Shared Embedding Space:** Embedding both event sequences and descriptions using the same model.
- **Pooling Strategies:** Various pooling techniques to aggregate hidden states into a sequence-level representation.
- **QLoRA Quantization:** Efficient low-rank fine-tuning for large language models using 4-bit precision.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/zefang-liu/TPP-Embedding.git
cd TPP-Embedding
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Add the source code to your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:<path-to-your-folder>/src
```

## Usage

To train the model on your dataset, first preprocess the event sequences and descriptions in `data/`, then run:

```bash
python scripts/train_embedding.py @configs/your_data.config
```

## Datasets

The model has been evaluated on multiple real-world datasets, including:

- Stack Overflow
- Chicago Crime
- NYC Taxi Trip
- U.S. Earthquake
- Amazon Reviews

All above datasets are available on [Hugging Face](https://huggingface.co/tppllm).

## Citation

If you find TPP-Embedding useful in your research, please cite our [papers](https://arxiv.org/abs/2410.14043):

```bibtex
@article{liu2024tppllmm,
  title={TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models},
  author={Liu, Zefang and Quan, Yinzhu},
  journal={arXiv preprint arXiv:2410.02062},
  year={2024}
}

@article{liu2024efficient,
  title={Retrieval of Temporal Event Sequences from Textual Descriptions},
  author={Liu, Zefang and Quan, Yinzhu},
  journal={arXiv preprint arXiv:2410.14043},
  year={2024}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
