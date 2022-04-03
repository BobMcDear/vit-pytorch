from torch import cat, Tensor, zeros
from torch.nn import Conv2d, Dropout, GELU, Linear, Module, Parameter


class MultilayerPerceptron(Module):
	"""
	Multilayer perceptron with one hidden layer
	"""
	def __init__(
		self, 
		in_dim: int,
		hidden_dim: int,
		out_dim: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Dimension of the input
			hidden_dim (int): Dimension of the hidden layer
			out_dim (int): Dimension of the output
			dropout_p (float): Probability for dropouts applied after the
			hidden layer and second linear layer
		"""
		super().__init__()

		self.lin_1 = Linear(
			in_features=in_dim,
			out_features=hidden_dim,
			)
		self.act_1 = GELU()
		self.dropout_1 = Dropout(p=dropout_p)
		self.lin_2 = Linear(
			in_features=hidden_dim,
			out_features=out_dim,
			)
		self.dropout_2 = Dropout(p=dropout_p)
	
	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the multilayer perceptron

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Output of the multilayer perceptron
		"""
		output = self.lin_1(input)
		output = self.act_1(output)
		output = self.dropout_1(output)
		output = self.lin_2(output)
		output = self.dropout_2(output)
		return output


class Tokenizer(Module):
	"""
	Tokenizes an image
	"""
	def __init__(
		self,
		token_dim: int,
		patch_size: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each token
			patch_size (int): Height/width of each patch
		"""
		super().__init__()

		self.input_to_tokens = Conv2d(
			in_channels=3,
			out_channels=token_dim,
			kernel_size=patch_size,
			stride=patch_size,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Tokenizes the input with patch embeddings

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Tokens, in the shape of 
		(batch_size, n_token, token_dim)
		"""
		output = self.input_to_tokens(input)
		output = output.flatten(start_dim=-2, end_dim=-1)
		output = output.transpose(-2, -1)
		return output


class ClassTokenConcatenator(Module):
	"""
	Concatenates a class token to a set of tokens
	"""
	def __init__(
		self,
		token_dim: int,
		):
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each token
		"""
		super().__init__()

		class_token = zeros(token_dim)
		self.class_token = Parameter(class_token)
	

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Concatenates the class token to the input

		Args:
			input (Tensor): Input
		
		Returns (Tensor): The input, with the class token concatenated to
		it
		"""
		class_token = self.class_token.expand(len(input), 1, -1)
		output = cat((input, class_token), dim=1)
		return output


class PositionEmbeddingAdder(Module):
	"""
	Adds learnable parameters to tokens for position embedding
	"""
	def __init__(
		self,
		n_tokens: int,
		token_dim: int,
		):
		"""
		Sets up the modules

		Args:
			n_tokens (int): Number of tokens
			token_dim (int): Dimension of each token
		"""
		super().__init__()

		position_embedding = zeros(n_tokens, token_dim)
		self.position_embedding = Parameter(position_embedding)
	
	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Adds the position embeddings to the input

		Args:
			input (Tensor): Input
		
		Returns (Tensor): The input, with the position embeddings added
		"""
		output = input+self.position_embedding
		return output
