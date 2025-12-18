# CS336 Spring 2025 Assignment 1: Basics

## Project Overview
This project is the first assignment for the CS336 course (Spring 2025), focusing on the implementation of a Transformer Language Model and a Byte Pair Encoding (BPE) Tokenizer from scratch.

The goal is to build the fundamental building blocks of Large Language Models (LLMs), including:
-   **Transformer Architecture:** Linear layers, Embeddings, RMSNorm, SwiGLU Feed-Forward Networks, and Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE).
-   **Tokenization:** A BPE tokenizer to process text into tokens.
-   **Optimization:** Implementations for Cross-Entropy Loss, Gradient Clipping, and AdamW optimizer.

## Architecture & Structure
The codebase is organized as follows:
-   **`cs336_basics/`**: The main package containing the implementation.
    -   **`transformer/`**: Core transformer components (`Linear.py`, `Embedding.py`, `FFN.py`, `transformer.py`, `RMSNorm.py`, `rope.py`).
    -   **`bpe.py`**: BPE Tokenizer implementation.
    -   **`util/`**: Utility functions (e.g., cross-entropy loss).
-   **`tests/`**: Unit tests to verify the correctness of the implementation.
    -   **`adapters.py`**: A crucial file that acts as a bridge between the test suite and your implementation. You need to implement the functions here to expose your code to the tests.

## Building and Running

### Environment Setup
This project uses `uv` for dependency management.
To install `uv`:
```sh
# Recommended
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via pip
pip install uv
```

### Running Code
You can run any Python script in the repository using `uv run`. This automatically handles environment creation and dependency resolution.
```sh
uv run <python_file_path>
```

### Running Tests
To run the test suite:
```sh
uv run pytest
```
Note: Tests will initially fail. You need to complete the implementation in `cs336_basics` and connect it via `tests/adapters.py`.

### Data Setup
To download the required datasets (TinyStories and OpenWebText sample), run the following commands:
```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Development Conventions

1.  **Implementation First:** Focus on implementing the core classes in `cs336_basics/`.
2.  **Adapter Pattern:** The tests utilize `tests/adapters.py` to call your code. Ensure the functions in `adapters.py` correctly instantiate and use your classes.
    -   Example: `run_linear` in `adapters.py` should create your `Linear` module, load weights, and run the forward pass.
3.  **Type Hinting:** The codebase uses `jaxtyping` for tensor shape validation. Adhere to these type hints.
4.  **Conventions:** Follow standard PyTorch practices.
    -   Modules should inherit from `torch.nn.Module`.
    -   Use `torch.Tensor` for data.
    -   Respect the shapes specified in the docstrings.

## Key Files to Watch
-   `cs336_basics/transformer/transformer.py`: Main Transformer block and LM implementation.
-   `tests/adapters.py`: The interface for tests.
-   `pyproject.toml`: Project dependencies and configuration.
