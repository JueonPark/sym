"""DLRM/GraphSAGE kernels from PR #162 (cdc44e2), plus offload compute regions.

Only benchmark code: the high-priority optimization series did not merge
PR #162's DLRM/GraphSAGE examples. Keep their original model mathematics here.
"""
import torch
F = torch.nn.functional


def decoder_layer(x, past, w, heads=16):
    d, head_dim = x.shape[1], x.shape[1] // heads
    q, k, v = F.linear(F.layer_norm(x, (d,)), w[0]).split(d, dim=1)
    q, k, v = (t.view(-1, heads, head_dim).transpose(0, 1) for t in (q, k, v))
    if past is not None:
        k, v = torch.cat([past[0], k], dim=1), torch.cat([past[1], v], dim=1)
    k, v = k.contiguous(), v.contiguous()
    total, n = k.shape[1], q.shape[1]
    mask = torch.ones(n, total, dtype=torch.bool, device=x.device).tril(total - n)
    scores = (q @ k.transpose(1, 2) / head_dim ** 0.5).masked_fill(~mask, float('-inf'))
    attention = scores.softmax(dim=-1) @ v
    x = x + F.linear(attention.transpose(0, 1).reshape(n, d), w[1])
    return x + F.linear(F.gelu(F.linear(F.layer_norm(x, (d,)), w[2])), w[3]), (k, v)


def embed(token, position, tokens, start):
    return token[tokens] + position[start:start + len(tokens)]


def logits(x, token):
    return F.layer_norm(x, (x.shape[1],)) @ token.t()


def route(x, router):
    h = F.layer_norm(x, (x.shape[1],))
    values, experts = (h @ router).topk(2, dim=1)
    return h, experts, values.softmax(dim=1)


def expert(h, tokens, w):
    return F.linear(F.gelu(F.linear(h[tokens], w[0])), w[1])


def combine(out, gates, tokens, slots, y):
    return out.index_add_(0, tokens, gates[tokens, slots].unsqueeze(1) * y)

DENSE_FEATURES = 13
EMBEDDING_DIM = 64


def linear_stack(sizes, generator, device):
    """(weight, zero bias) pairs of an MLP with these layer widths, on ``device``."""
    layers = []
    for fan_in, fan_out in zip(sizes, sizes[1:]):
        weight = torch.randn(fan_out, fan_in, generator=generator) / fan_in ** 0.5
        layers.append((weight.to(device), torch.zeros(fan_out, device=device)))
    return layers


def run_stack(layers, x):
    """ReLU between layers, none after the last."""
    for index, (weight, bias) in enumerate(layers):
        x = F.linear(x, weight, bias)
        if index < len(layers) - 1:
            x = torch.relu(x)
    return x


class DLRM:
    """Sum-pooled embedding tables on the CPU; bottom MLP, dot interaction and top MLP on the GPU."""

    def __init__(self, tables, rows, device, generator):
        self.tables = [
            torch.nn.EmbeddingBag.from_pretrained(torch.randn(rows, EMBEDDING_DIM, generator=generator) * 0.05,
                                                  freeze=True, mode="sum")
            for _ in range(tables)
        ]
        self.bottom = linear_stack([DENSE_FEATURES, 512, 256, EMBEDDING_DIM], generator, device)
        self.top = linear_stack([EMBEDDING_DIM + (tables + 1) * tables // 2, 512, 256, 1], generator, device)
        self.pairs = torch.triu_indices(tables + 1, tables + 1, offset=1, device=device)

    def pooled(self, indices, offsets):
        """CPU, PyTorch: every table's sum-pooled bags -> [T, B, 64] float32."""
        return torch.stack([table(i, o) for table, i, o in zip(self.tables, indices, offsets)])

    def predict(self, dense, embeddings):
        """GPU: dense [B, 13] float32 and embeddings [B, T, 64] float16 -> click probabilities [B]."""
        bottom = run_stack(self.bottom, dense)
        vectors = torch.cat([bottom.unsqueeze(1), embeddings.float()], dim=1)       # [B, T + 1, 64]
        dots = torch.bmm(vectors, vectors.transpose(1, 2))
        interactions = dots[:, self.pairs[0], self.pairs[1]]                      # [B, (T + 1) T / 2]
        return torch.sigmoid(run_stack(self.top, torch.cat([bottom, interactions], dim=1))).squeeze(1)


def make_batch(batch, tables, rows, generator):
    """Dense features and, per table, one bag of 1-4 uniform random rows per sample."""
    dense = torch.randn(batch, DENSE_FEATURES, generator=generator)
    indices, offsets = [], []
    for _ in range(tables):
        bags = torch.randint(1, 5, (batch,), generator=generator)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), bags.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, rows, (int(bags.sum()),), generator=generator))
    return dense, indices, offsets



FEATURES = 128
HIDDEN = 128
CLASSES = 16


def random_graph(nodes, avg_degree, generator):
    """CSR adjacency: node v has randint(1, 2 * avg_degree) uniform random neighbours."""
    degrees = torch.randint(1, 2 * avg_degree, (nodes,), generator=generator)
    indptr = torch.zeros(nodes + 1, dtype=torch.long)
    indptr[1:] = degrees.cumsum(0)
    return indptr, torch.randint(0, nodes, (int(indptr[-1]),), generator=generator)


def sample(graph, nodes, fanout, generator):
    """``fanout`` neighbours of every node, uniformly with replacement -> [len(nodes), fanout]."""
    indptr, indices = graph
    start = indptr[nodes]
    degree = indptr[nodes + 1] - start
    picks = (torch.rand(len(nodes), fanout, generator=generator) * degree.unsqueeze(1)).long()
    return indices[start.unsqueeze(1) + picks]


def minibatch(graph, seeds, fanouts, generator):
    """Two-hop sample around sorted unique ``seeds`` (CPU, PyTorch).

    ``nodes`` are the sorted global ids whose features the batch needs. Layer 1
    reads rows of ``nodes`` for the one-hop set ``frontier`` and its sampled
    neighbours; layer 2 reads rows of the layer-1 output for the seeds and theirs."""
    hop0 = sample(graph, seeds, fanouts[0], generator)
    frontier = torch.unique(torch.cat([seeds, hop0.flatten()]))
    hop1 = sample(graph, frontier, fanouts[1], generator)
    nodes = torch.unique(torch.cat([frontier, hop1.flatten()]))
    return {
        "nodes": nodes,
        "layer1": (torch.searchsorted(nodes, frontier), torch.searchsorted(nodes, hop1)),
        "layer2": (torch.searchsorted(frontier, seeds), torch.searchsorted(frontier, hop0)),
    }


class GraphSAGE:
    """Two SAGE-mean layers with a fixed fanout: aggregation is a gather plus a mean (no atomics)."""

    def __init__(self, generator, device):
        def weight(fan_in, fan_out):
            return (torch.randn(fan_out, fan_in, generator=generator) / fan_in ** 0.5).to(device)

        self.layer1 = (weight(FEATURES, HIDDEN), weight(FEATURES, HIDDEN))
        self.layer2 = (weight(HIDDEN, CLASSES), weight(HIDDEN, CLASSES))

    def forward(self, features, layer1, layer2):
        """features [n, 128] float16 on the GPU and the batch's local indices -> seed logits [seeds, 16]."""
        x = features.float()
        own, neighbours = layer1
        h = torch.relu(F.linear(x[own], self.layer1[0]) + F.linear(x[neighbours].mean(dim=1), self.layer1[1]))
        own, neighbours = layer2
        return F.linear(h[own], self.layer2[0]) + F.linear(h[neighbours].mean(dim=1), self.layer2[1])


