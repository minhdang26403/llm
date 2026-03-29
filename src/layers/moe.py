import torch
import torch.nn as nn


class MoERouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # The gating network: simply maps hidden_dim to num_experts
        # We usually disable bias in the router to keep the routing symmetric
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            routing_weights: [batch_size * seq_len, top_k]
            selected_experts: [batch_size * seq_len, top_k]
        """
        # 1. Flatten the batch and sequence dimensions.
        # In MoE, every single token is routed completely independently.
        flat_hidden_states = hidden_states.view(-1, self.hidden_dim)

        # 2. Get the raw routing logits
        # Shape: [batch_size * seq_len, num_experts]
        logits: torch.Tensor = self.gate(flat_hidden_states)

        # 3. Convert to probabilities
        routing_probs = logits.softmax(dim=-1)

        # 4. Select the Top-K experts
        # routing_weights shape: [batch_size * seq_len, top_k]
        # selected_experts shape: [batch_size * seq_len, top_k]
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k)

        # 5. Re-normalize the weights so they sum to 1.0
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts


class MoETokenDispatcher:
    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k

        # State variables to hold the memory maps between dispatch and combine
        self.sort_map = torch.empty(0)
        self.batch_size = 0
        self.seq_len = 0
        self.hidden_dim = 0

    def dispatch(
        self, hidden_states: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sorts tokens into contiguous memory blocks for each expert.

        Args:
            hidden_states: The raw token embeddings.
                           Shape: [batch_size, seq_len, hidden_dim]
            selected_experts: The expert indices chosen by the router.
                              Shape: [batch_size * seq_len, top_k]

        Returns:
            sorted_tokens: Tokens duplicated and grouped by expert.
                           Shape: [batch_size * seq_len * top_k, hidden_dim]
            sorted_indices: The expert ID for each sorted token.
                            Shape: [batch_size * seq_len * top_k]
        """
        self.batch_size, self.seq_len, self.hidden_dim = hidden_states.shape

        # 1. Flatten the inputs
        flat_hidden_states = hidden_states.view(-1, self.hidden_dim)
        flat_expert_indices = selected_experts.view(-1)

        # 2. Duplicate tokens for Top-K routing
        # If k=2, we need two physical copies of the token in memory
        duplicated_tokens = (
            flat_hidden_states.unsqueeze(1)
            .expand(-1, self.top_k, -1)
            .reshape(-1, self.hidden_dim)
        )

        # 3. The Permutation
        # For pure PyTorch, argsort is the cleanest representation of the memory mapping
        self.sort_map = torch.argsort(flat_expert_indices)

        # 4. Group the tokens
        sorted_tokens = duplicated_tokens[self.sort_map]
        sorted_expert_indices = flat_expert_indices[self.sort_map]

        # We return the sorted tokens to be fed into the MLPs,
        # and the indices so we know which token belongs to which expert.
        return sorted_tokens, sorted_expert_indices

    def combine(self, expert_outputs: torch.Tensor, routing_weights: torch.Tensor):
        """
        Un-sorts the processed tokens, scales them by router weights, and restores the
            sequence.

        Args:
            expert_outputs: The tokens after passing through the MLPs.
                            Shape: [batch_size * seq_len * top_k, hidden_dim]
            routing_weights: The softmax probabilities from the router.
                             Shape: [batch_size * seq_len, top_k]

        Returns:
            final_output: The combined tokens, restored to their original shape.
                          Shape: [batch_size, seq_len, hidden_dim]
        """
        # 1. Reverse the sort map to figure out where tokens originally came from
        reversed_map = torch.argsort(self.sort_map)

        # 2. Un-sort the expert outputs back to their original sequence order
        unsorted_outputs = expert_outputs[reversed_map]

        # 3. Reshape back to [batch_size * seq_len, top_k, hidden_dim]
        unsorted_outputs = unsorted_outputs.view(-1, self.top_k, self.hidden_dim)

        # 4. Multiply by the routing weights calculated by the MoERouter
        # We unsqueeze the weights to broadcast across the hidden dimension
        weighted_outputs = unsorted_outputs * routing_weights.unsqueeze(-1)

        # 5. Sum the Top-K expert outputs together for each token
        final_output = weighted_outputs.sum(dim=1)

        # 6. Restore the sacred sequence dimension!
        return final_output.view(self.batch_size, self.seq_len, self.hidden_dim)


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()

        # Standard two-layer MLP for an expert
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class SparseMoEBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_experts: int, top_k: int, intermediate_dim: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. The Scanner
        self.router = MoERouter(hidden_dim, num_experts, top_k)

        # 2. The Conveyor Belt
        self.dispatcher = MoETokenDispatcher(num_experts, top_k)

        # 3. The Assembly Lines
        # We use an nn.ModuleList so PyTorch properly registers the weights
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        # --- STEP 1: TRIAGE ---
        # Get the tickets and the probabilities
        routing_weights, selected_experts = self.router(hidden_states)

        # --- STEP 2: DISPATCH ---
        # Physically group the memory in VRAM
        sorted_tokens, sorted_expert_indices = self.dispatcher.dispatch(
            hidden_states, selected_experts
        )

        # --- STEP 3: EXECUTE EXPERTS ---
        # A. Count exactly how many tokens are going to each expert
        # minlength ensures we get an array of size `num_experts` even if an expert gets
        # zero tokens.
        expert_counts = torch.bincount(
            sorted_expert_indices, minlength=self.num_experts
        ).tolist()

        # B. Slice the massive sorted_tokens tensor into smaller contiguous chunks
        token_chunks = torch.split(sorted_tokens, expert_counts, dim=0)

        expert_output_chunks = []

        # C. Run the math (only loops N times, not batch_size * seq_len times!)
        for i, chunk in enumerate(token_chunks):
            if chunk.shape[0] > 0:
                # If the expert was assigned tokens, run the forward pass
                out = self.experts[i](chunk)
            else:
                # If the expert was starved (0 tokens), return an empty tensor of the
                # right shape
                out = torch.empty(
                    (0, self.hidden_dim), device=chunk.device, dtype=chunk.dtype
                )

            expert_output_chunks.append(out)

        # D. Stitch the processed chunks back into a single tensor
        # Shape: [batch_size  * seq_len * top_k, hidden_dim]
        expert_outputs = torch.cat(expert_output_chunks, dim=0)

        # --- STEP 4: COMBINE ---
        # Un-sort, scale by probabilities, and restore the sequence shape
        final_output = self.dispatcher.combine(expert_outputs, routing_weights)

        return final_output
