from typing import Tuple

from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import softmax


class QueriesKeysValuesExtractor(Module):
	"""
	Gets queries, keys, and values for multi-head self-attention
	"""
	def __init__(
		self,
		token_dim: int,
		head_dim: int,
		n_heads: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			head_dim (int): Dimension of the queries/keys/values per head
			n_heads (int): Number of heads
		"""
		super().__init__()

		self.head_dim = head_dim
		self.n_heads = n_heads

		queries_keys_values_dim = 3 * self.head_dim * self.n_heads
		self.input_to_queries_keys_values = Linear(
			in_features=token_dim,
			out_features=queries_keys_values_dim,
			bias=False,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tuple[Tensor, Tensor, Tensor]:
		"""
		Gets queries, keys, and values from the input

		Args:
			input (Tensor): Input
		
		Returns (Tuple[Tensor, Tensor, Tensor]): Queries, keys, and values
		"""
		batch_size, n_tokens, token_dim = input.shape
		queries_keys_values = self.input_to_queries_keys_values(input)
		queries_keys_values = queries_keys_values.reshape(
			batch_size,
			3,
			self.n_heads,
			n_tokens,
			self.head_dim,
			)
		return queries_keys_values.unbind(dim=1)


def get_attention(
	queries: Tensor,
	keys: Tensor,
	values: Tensor,
	) -> Tensor:
	"""
	Calculates attention

	Args:
		queries (Tensor): Queries
		keys (Tensor): Keys
		values (Tensor): Values
	
	Returns (Tensor): Attention calculated using the provided queries, keys,
	and values
	"""
	scale = queries.shape[-1] ** -0.5
	attention_scores = (queries @ keys.transpose(-2, -1)) * scale
	attention_probabilities = softmax(attention_scores, dim=-1)
	return attention_probabilities @ values


class MultiHeadSelfAttention(Module):
	"""
	Multi-head self-attention
	"""
	def __init__(
		self,
		token_dim: int,
		head_dim: int,
		n_heads: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			head_dim (int): Dimension of the queries/keys/values per head
			n_heads (int): Number of heads
			dropout_p (float): Probability for dropout applied on the output
		"""
		super().__init__()

		self.query_keys_values_extractor = QueriesKeysValuesExtractor(
			token_dim=token_dim,
			head_dim=head_dim,
			n_heads=n_heads,
			)

		self.concatenated_heads_dim = n_heads*head_dim
		self.attention_to_output = Linear(
			in_features=self.concatenated_heads_dim,
			out_features=token_dim,
			)

		self.output_dropout = Dropout(
			p=dropout_p,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Calculates attention from the input

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Attention
		"""
		batch_size, n_tokens, token_dim = input.shape

		queries, keys, values = self.query_keys_values_extractor(input)
		attention = get_attention(
			queries=queries,
			keys=keys,
			values=values,
			)
		
		output = attention.transpose(1, 2).reshape(
			batch_size,
			n_tokens,
			self.concatenated_heads_dim,
			)
		output = self.attention_to_output(output)
		return self.output_dropout(output)
