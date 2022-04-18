# PyTorch-Vision-Transformer
## Description
This is an implementation of the vision transformer in PyTorch. You can find the accompanying blog [here](https://borna-ahz.medium.com/coding-the-vision-transformer-in-pytorch-part-1-birds-eye-view-1c0a79d8732e).
## Usage
The ```VisionTransformer``` class from ```model.py``` is very flexible and can be utilized for fetching vision transformers of various settings. Arguments
that are to be passed are the token dimension, patch size, image size, depth of transformer, dimension of query/key/value vectors,
number of heads for multi-head self-attention, the hidden dimension of the transformer's multilayer perceptrons, the rate of dropout, 
and finally the number of output classes. 

For instance, a ViT-Base may be constructed via,
```python
from model import VisionTransformer


# ViT-Base with a patch size of 16, an input size of 256, and 1000 classes
vit_base = VisionTransformer(
  token_dim=768, # Token dimension
  patch_size=16, # Patch size
  image_size=256, # Image size
  n_layers=12, # Depth of transformer
  multihead_attention_head_dim=64, # Dimension of query/key/value vectors
  multihead_attention_n_heads=12, # Number of heads for multi-head self-attention
  multilayer_perceptron_hidden_dim=3072, # The hidden dimension of the transformer's multilayer perceptrons
  dropout_p=0.1, # The rate of dropout
  n_classes=1000,
  )
```

The implemented modules may also be used out-of-the-box. They include,

* ```utils.py```:
  * ```MultilayerPerceptron```: Multilayer perceptron with one hidden layer
    * ```__init__```: Sets up the modules
      * Args:
        * ```in_dim (int)```: Dimension of the input
        * ```hidden_dim (int)```: Dimension of the output
        * ```out_dim (int)```: Dimension of the hidden layer
        * ```dropout_p (float)```: Probability for dropouts applied after the hidden layer and second linear layer
    * ```forward```: Runs the input through the multilayer perceptron
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): Output of the multilayer perceptron
  * ```Tokenizer```: Tokenizes images
    * ```__init__```: Sets up the modules
      * Args:
        * ```token_dim (int)```: Dimension of each token
        * ```patch_size (int)```: Height/width of each patch
    * ```forward```: Tokenizes the input
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): Resultant tokens (one-dimensional)
  * ```ClassTokenConcatenator```: Concatenates a class token to a set of tokens
    * ```__init__```: Sets up the modules
      * Args:
        * ```token_dim (int)```: Dimension of each token
    * ```forward```: Concatenates the class token to the input
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): The input, with the class token concatenated to it
  * ```PositionEmbeddingAdder```: Adds learnable parameters to tokens for position embedding
    * ```__init__```: Sets up the modules
      * Args:
        * ```n_tokens (int)```: Number of tokens
        * ```token_dim (int)```: Dimension of each token
    * ```forward```: Adds learnable parameters to the input tokens
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): The input, with the learnable parameters added
* ```attention.py```:
  * ```QueriesKeysValuesExtractor```: Gets queries, keys, and values for multi-head self-attention
    * ```__init__```: Sets up the modules
      * Args:
        * ```token_dim (int)```: Dimension of each input token
        * ```head_dim (int)```: Dimension of the queries/keys/values per head
        * ```n_heads (int)```: Number of heads
    * ```forward```: Gets queries, keys, and values from the input
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tuple[Tensor, Tensor, Tensor]```): Queries, keys, and values
  * ```get_attention```: Calculates multi-head self-attention from queries, keys, and values
    * Args:
      * ```queries (Tensor)```: Queries
      * ```keys (Tensor)```: Keys
      * ```values (Tensor)```: Values
    * Returns (```Tensor```): Multi-head self-attention calculated using the provided queries, keys, and values
  * ```MultiHeadSelfAttention```: Multi-head self-attention
    * ```__init__```: Sets up the modules
      * Args:
        * ```token_dim (int)```: Dimension of each input token
        * ```head_dim (int)```: Dimension of the queries/keys/values per head
        * ```n_heads (int)```: Number of heads
        * ```dropout_p (float)```: Probability for dropout applied on the output
    * ```forward```: Applies multi-head self-attention to the input
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): Result of multi-head self-attention
* ```model.py```:
  * ```TransformerBlock```: Transformer block
    * ```__init__```: Sets up the modules
      * Args:
        * ```token_dim (int)```: Dimension of each input token
        * ```multihead_attention_head_dim (int)```: Dimension of the queries/keys/values per head for multi-head self-attention
        * ```multihead_attention_n_heads (int)```: Number of heads for multi-head self-attention
        * ```multilayer_perceptron_hidden_dim (int)```: Dimension of the hidden layer for the multilayer perceptrons
        * ```dropout_p (float)```: Probability for dropout for multi-head self-attention and the multilayer perceptrons
    * ```forward```: Runs the input through the transformer block
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): Output of the transformer block
  * ```Transformer```: Transformer
    * ```__init__```: Sets up the modules
      * Args:
        * ```n_layers (int)```: Depth of the transformer
        * ```token_dim (int)```: Dimension of each input token
        * ```multihead_attention_head_dim (int)```: Dimension of the queries/keys/values per head for multi-head self-attention
        * ```multihead_attention_n_heads (int)```: Number of heads for multi-head self-attention
        * ```multilayer_perceptron_hidden_dim (int)```: Dimension of the hidden layer for the multilayer perceptrons
        * ```dropout_p (float)```: Probability for dropout for multi-head self-attention and the multilayer perceptrons
    * ```forward```: Runs the input through the transformer
      * Args:
        * ```input (Tensor)```: Input
      * Returns (```Tensor```): Output of the transformer 

