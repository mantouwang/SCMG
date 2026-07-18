import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import GCNConv

try:
    from torch_geometric.utils import dropout_edge
except ImportError:  # pragma: no cover
    dropout_edge = None
    from torch_geometric.utils import dropout_adj


def _drop_edges(edge_index, probability, training):
    if not training or probability <= 0.0:
        return edge_index
    if dropout_edge is not None:
        return dropout_edge(
            edge_index,
            p=float(probability),
            force_undirected=True,
            training=True,
        )[0]
    return dropout_adj(
        edge_index,
        p=float(probability),
        force_undirected=True,
        training=True,
    )[0]


def _consume_linear_initialization(in_features, out_features):
    weight = torch.empty(out_features, in_features)
    bias = torch.empty(out_features)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


class LocalGraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, local_residual=True, adaptive_neighbor=False):
        super().__init__()
        self.graph = GCNConv(in_dim, out_dim, add_self_loops=False)
        self.local = nn.Linear(in_dim, out_dim) if local_residual else None
        self.neighbor_scale = nn.Parameter(torch.zeros(())) if adaptive_neighbor else None
        self.disagreement_gate = None
        self.disagreement_alpha = None

    def enable_disagreement(self, hidden_dim=4):
        out_dim = self.graph.out_channels
        self.disagreement_gate = nn.Sequential(
            nn.Linear(out_dim * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), out_dim),
        )
        self.disagreement_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index):
        graph = self.graph(x, edge_index)
        if self.neighbor_scale is not None:
            graph = torch.exp(self.neighbor_scale) * graph
        if self.local is None:
            return graph
        local = F.relu(self.local(x))
        output = local + graph
        if self.disagreement_gate is not None:
            difference = local - graph
            gate_input = torch.cat([local, graph, torch.abs(difference)], dim=1)
            correction = torch.tanh(self.disagreement_gate(gate_input)) * graph
            output = output + torch.tanh(self.disagreement_alpha) * correction
        return output


class VariantClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, adaptive_neighbor=True):
        super().__init__()
        self.graph1 = GCNConv(in_dim, hidden_dim, add_self_loops=False)
        self.graph2 = GCNConv(hidden_dim, 1, add_self_loops=False)
        self.local1 = nn.Linear(in_dim, hidden_dim)
        self.local2 = nn.Linear(hidden_dim, 1)
        self.graph_scale1 = nn.Parameter(torch.zeros(())) if adaptive_neighbor else None
        self.graph_scale2 = nn.Parameter(torch.zeros(())) if adaptive_neighbor else None
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        graph1 = self.graph1(x, edge_index)
        if self.graph_scale1 is not None:
            graph1 = torch.exp(self.graph_scale1) * graph1
        hidden = F.relu(graph1) + F.relu(self.local1(x))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = self.graph2(hidden, edge_index)
        if self.graph_scale2 is not None:
            output = torch.exp(self.graph_scale2) * output
        return output + self.local2(hidden)


class DetachedRankReadouts(nn.Module):
    def __init__(self, in_dim=100, ranks=(6, 10), alpha_init=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, rank),
                    nn.GELU(),
                    nn.Linear(rank, 1),
                )
                for rank in ranks
            ]
        )
        self.alphas = nn.Parameter(torch.full((len(ranks),), float(alpha_init)))

    def forward(self, x):
        source = x.detach()
        return sum(
            torch.tanh(alpha) * head(source)
            for alpha, head in zip(self.alphas, self.heads)
        )


class AdvancedOmicsBulkBranch(nn.Module):
    """Bulk encoder for three 16-dimensional omics views."""

    def __init__(self):
        super().__init__()
        hidden_dim = 64
        classifier_hidden = 200
        self.dropout = 0.55
        self.edge_dropout = 0.55
        self.input_dropout = 0.35

        self.public_encoder = LocalGraphBlock(16, hidden_dim, True, True)
        self.private_encoders = nn.ModuleList(
            [LocalGraphBlock(16, hidden_dim, True, True) for _ in range(3)]
        )
        self.public_alpha = nn.Parameter(torch.full((3,), -3.0))

        self.private_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(16, 4),
                    nn.GELU(),
                    nn.Linear(4, hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.private_adapter_alpha = nn.Parameter(torch.full((3,), 0.0))

        self.view_identity = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.view_identity, std=0.02)

        self.directed_pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
        )

        self.h2_blocks = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False) for _ in range(3)]
        )
        self.h2_alpha = nn.Parameter(torch.tensor(-3.0))
        self.shared_down = nn.Linear(hidden_dim, 16)
        self.shared_up = nn.Linear(16, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.view_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])
        self.relation_head = nn.Linear(hidden_dim, 1)
        self.fused_head = nn.Linear(hidden_dim, 1)
        self.dispersion_head = nn.Linear(hidden_dim * 2, 1)
        self.raw_norms = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.raw_view_heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(3)])
        self.expert_alphas = nn.Parameter(
            torch.tensor([-4.0, -4.0, -4.0, -3.0], dtype=torch.float32)
        )
        self.extra_expert_alphas = nn.Parameter(
            torch.tensor([-5.0, -5.0, -5.0, -4.5, -4.0], dtype=torch.float32)
        )

        self.project = LocalGraphBlock(hidden_dim, 100, True, True)
        self.project.enable_disagreement(hidden_dim=4)
        self.classifier = VariantClassifier(100, classifier_hidden, self.dropout, True)
        self.rank_readouts = DetachedRankReadouts(100, (6, 10), 0.0)

    def _tokens(self, x_bulk, edge_index):
        raw = x_bulk[:, :48]
        dropped = F.dropout(raw, p=self.input_dropout, training=self.training)
        blocks = [dropped[:, 0:16], dropped[:, 16:32], dropped[:, 32:48]]
        tokens = []
        for index, block in enumerate(blocks):
            public = self.public_encoder(block, edge_index)
            private = self.private_encoders[index](block, edge_index)
            token = private + torch.sigmoid(self.public_alpha[index]) * public
            token = token + torch.tanh(
                self.private_adapter_alpha[index]
            ) * self.private_adapters[index](block)
            tokens.append(token + self.view_identity[index])
        return torch.stack(tokens, dim=1)

    def _relation_token(self, tokens):
        return self.directed_pair_encoder(
            (
                tokens[:, 0] - tokens[:, 1]
                + tokens[:, 0] - tokens[:, 2]
                + tokens[:, 1] - tokens[:, 2]
            )
            / 3.0
        )

    def forward(self, x_bulk, edge_index):
        edges = _drop_edges(edge_index, self.edge_dropout, self.training)
        tokens = self._tokens(x_bulk, edges)
        fusion_relation = self._relation_token(tokens)
        signed_relation = torch.cat(
            [
                tokens[:, 0] - tokens[:, 1],
                tokens[:, 0] - tokens[:, 2],
                tokens[:, 1] - tokens[:, 2],
            ],
            dim=1,
        )
        fusion_input = torch.cat(
            [tokens.flatten(start_dim=1), signed_relation, fusion_relation], dim=1
        )
        fused = self.fusion(fusion_input)
        z_base = self.project(fused, edges)
        z = F.dropout(z_base, p=self.dropout, training=self.training)
        logit = self.classifier(z, edges) + self.rank_readouts(z_base)

        view_logits = [head(tokens[:, i]) for i, head in enumerate(self.view_heads)]
        expert_relation = self._relation_token(tokens)
        expert_logits = view_logits + [self.relation_head(expert_relation)]
        logit = logit + sum(
            torch.tanh(alpha) * expert
            for alpha, expert in zip(self.expert_alphas, expert_logits)
        )

        raw = F.dropout(x_bulk[:, :48], p=self.input_dropout, training=self.training)
        raw_blocks = [raw[:, 0:16], raw[:, 16:32], raw[:, 32:48]]
        extra_logits = [
            head(norm(block))
            for head, norm, block in zip(self.raw_view_heads, self.raw_norms, raw_blocks)
        ]
        dispersion = torch.cat(
            [
                tokens.std(dim=1, unbiased=False),
                tokens.max(dim=1).values - tokens.min(dim=1).values,
            ],
            dim=1,
        )
        extra_logits.append(self.dispersion_head(dispersion))
        logit = logit + sum(
            torch.tanh(alpha) * expert
            for alpha, expert in zip(self.extra_expert_alphas, extra_logits)
        )
        return z_base, logit.view(-1)


class BulkBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.bulk_branch = AdvancedOmicsBulkBranch()

    def forward(self, x, edge_index):
        return self.bulk_branch(x, edge_index)


def make_mlp(in_dim, hidden_dim, out_dim, layers=2, dropout=0.2):
    modules = []
    dimension = in_dim
    for _ in range(max(layers - 1, 0)):
        modules.extend(
            [
                nn.Linear(dimension, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        dimension = hidden_dim
    modules.append(nn.Linear(dimension, out_dim))
    return nn.Sequential(*modules)


class StageSourceTransformerEncoder(nn.Module):
    def __init__(self, token_dim=64, layers=2, heads=2, dropout=0.35):
        super().__init__()
        self.token = nn.Linear(4, token_dim)
        self.stage_emb = nn.Parameter(torch.zeros(3, token_dim))
        self.source_emb = nn.Parameter(torch.zeros(3, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))
        nn.init.normal_(self.stage_emb, std=0.02)
        nn.init.normal_(self.source_emb, std=0.02)

    def forward(self, batch):
        x = batch["pb_token"]
        n = x.size(0)
        hidden = self.token(x)
        hidden = (
            hidden
            + self.stage_emb.view(1, 3, 1, -1)
            + self.source_emb.view(1, 1, 3, -1)
        )
        hidden = self.encoder(hidden.reshape(n, 9, -1))
        return self.out(hidden.mean(dim=1))


class ProgressionAttentionEncoder(nn.Module):
    def __init__(self, token_dim=64, layers=2, heads=2, dropout=0.35):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(4, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.source_score = nn.Linear(token_dim, 1)
        self.prog_emb = nn.Parameter(torch.zeros(6, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.query = nn.Parameter(torch.zeros(token_dim))
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))
        nn.init.normal_(self.prog_emb, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, batch):
        source_tokens = self.token(batch["pb_token"])
        source_weights = torch.softmax(self.source_score(source_tokens).squeeze(-1), dim=2)
        stage = torch.sum(source_tokens * source_weights.unsqueeze(-1), dim=2)
        normal, precancer, cancer = stage[:, 0], stage[:, 1], stage[:, 2]
        progression = torch.stack(
            [
                normal,
                precancer,
                cancer,
                precancer - normal,
                cancer - precancer,
                cancer - normal,
            ],
            dim=1,
        )
        hidden = self.encoder(progression + self.prog_emb.unsqueeze(0))
        attention = torch.softmax(torch.matmul(hidden, self.query), dim=1)
        return self.out(torch.sum(hidden * attention.unsqueeze(-1), dim=1))


class HybridProgressionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = StageSourceTransformerEncoder()
        self.progression = ProgressionAttentionEncoder()
        self.gate = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(64, 64),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, batch):
        base = self.base(batch)
        progression = self.progression(batch)
        gate = self.gate(torch.cat([base, progression], dim=-1))
        return base + torch.sigmoid(self.alpha) * gate * progression


class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_mlp(13, 128, 64, 2, 0.35)

    def forward(self, batch):
        return self.net(batch["static13"])


class FiLMFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_d = nn.Linear(64, 64)
        self.proj_s = nn.Linear(64, 64)
        self.film = make_mlp(64, 128, 128, 2, 0.35)

    def forward(self, static, dynamic):
        gamma, beta = self.film(static).chunk(2, dim=-1)
        dynamic_projected = self.proj_d(dynamic)
        static_projected = self.proj_s(static)
        return (
            static_projected
            + (1.0 + 0.1 * torch.tanh(gamma)) * dynamic_projected
            + 0.1 * beta
        )


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(64, 128)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.out = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))

    def forward(self, x):
        return self.out(self.blocks(F.gelu(self.input(x)))).view(-1)


class RuntimeScRNAFeatureExtractor(nn.Module):
    """Reproduce the released scRNA summaries from complete cell matrices."""

    @staticmethod
    def _as_numpy(values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    @staticmethod
    def _aggregate_stage(lognorm, raw, source_mask):
        lognorm = np.asarray(lognorm, dtype=np.float32)
        raw = np.asarray(raw, dtype=np.float32)
        source_mask = np.asarray(source_mask, dtype=bool)
        n_genes = raw.shape[0]
        n_sources = source_mask.shape[0]

        raw_sum = np.zeros((n_genes, n_sources), dtype=np.float32)
        detection_rate = np.zeros_like(raw_sum)
        lognorm_mean = np.zeros_like(raw_sum)
        expressing_mean = np.zeros_like(raw_sum)
        n_cells = np.zeros(n_sources, dtype=np.float32)

        for source_index, mask in enumerate(source_mask):
            source_raw = raw[:, mask]
            source_lognorm = lognorm[:, mask]
            detected = source_raw > 0
            detected_count = detected.sum(axis=1)
            count = source_raw.shape[1]
            n_cells[source_index] = count
            raw_sum[:, source_index] = source_raw.sum(
                axis=1, dtype=np.float64
            ).astype(np.float32)
            detection_rate[:, source_index] = (
                detected_count / max(count, 1)
            ).astype(np.float32)
            lognorm_mean[:, source_index] = source_lognorm.mean(
                axis=1, dtype=np.float64
            ).astype(np.float32)
            expressing_sum = (source_lognorm * detected).sum(
                axis=1, dtype=np.float64
            )
            expressing_mean[:, source_index] = np.divide(
                expressing_sum,
                detected_count,
                out=np.zeros_like(expressing_sum, dtype=np.float64),
                where=detected_count > 0,
            ).astype(np.float32)

        return raw_sum, detection_rate, lognorm_mean, expressing_mean, n_cells

    @staticmethod
    def _source_correct(values):
        source_stage_mean = values.mean(axis=1, keepdims=True)
        gene_global_mean = values.mean(axis=(1, 2), keepdims=True)
        return values - source_stage_mean + gene_global_mean

    @staticmethod
    def _delta_consistency(delta_by_source, eps=1e-6):
        mean_delta = delta_by_source.mean(axis=1)
        mean_sign = np.sign(mean_delta)
        source_sign = np.sign(delta_by_source)
        agree = (source_sign == mean_sign[:, None]).mean(axis=1)
        near_zero = (np.abs(delta_by_source) <= eps).mean(axis=1)
        return np.where(np.abs(mean_delta) > eps, agree, near_zero).astype(np.float32)

    @classmethod
    def _zscore_train_apply(cls, values, train_mask, eps=1e-6):
        values = np.asarray(cls._as_numpy(values), dtype=np.float32)
        train_mask = np.asarray(cls._as_numpy(train_mask), dtype=bool)
        train_values = values[train_mask]
        mean = train_values.mean(axis=0, keepdims=True)
        std = train_values.std(axis=0, keepdims=True)
        std[std < eps] = 1.0
        result = (values - mean) / std
        return np.nan_to_num(
            result, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    @torch.no_grad()
    def forward(self, raw_batch):
        stage_aggregates = [
            self._aggregate_stage(
                self._as_numpy(lognorm),
                self._as_numpy(raw),
                self._as_numpy(source_mask),
            )
            for lognorm, raw, source_mask in zip(
                raw_batch["lognorm_matrices"],
                raw_batch["raw_matrices"],
                raw_batch["source_masks"],
            )
        ]
        raw_sum = np.stack([item[0] for item in stage_aggregates], axis=1)
        detection_rate = np.stack([item[1] for item in stage_aggregates], axis=1)
        lognorm_mean = np.stack([item[2] for item in stage_aggregates], axis=1)
        expressing_mean = np.stack([item[3] for item in stage_aggregates], axis=1)
        n_cells = np.stack([item[4] for item in stage_aggregates], axis=0)
        raw_per_cell = np.log1p(
            raw_sum / (n_cells.reshape(1, 3, 3) + 1e-6)
        ).astype(np.float32)
        pb_token = np.stack(
            [lognorm_mean, detection_rate, expressing_mean, raw_per_cell],
            axis=-1,
        ).astype(np.float32)
        lognorm_by_source = pb_token[..., 0]
        expressing_by_source = pb_token[..., 2]
        stage_log = self._source_correct(lognorm_by_source).mean(axis=2)
        stage_detect = pb_token[..., 1].mean(axis=2)
        stage_expr = self._source_correct(expressing_by_source).mean(axis=2)

        delta_pn_by_source = lognorm_by_source[:, 1] - lognorm_by_source[:, 0]
        delta_cp_by_source = lognorm_by_source[:, 2] - lognorm_by_source[:, 1]
        delta_cn_by_source = lognorm_by_source[:, 2] - lognorm_by_source[:, 0]
        source_consistency = ((
            self._delta_consistency(delta_pn_by_source)
            + self._delta_consistency(delta_cp_by_source)
            + self._delta_consistency(delta_cn_by_source)
        ) / 3.0).astype(np.float32)

        static13 = np.column_stack(
            [
                stage_log[:, 0],
                stage_log[:, 1],
                stage_log[:, 2],
                stage_detect[:, 0],
                stage_detect[:, 1],
                stage_detect[:, 2],
                stage_expr[:, 0],
                stage_expr[:, 1],
                stage_expr[:, 2],
                stage_log[:, 1] - stage_log[:, 0],
                stage_log[:, 2] - stage_log[:, 1],
                stage_log[:, 2] - stage_log[:, 0],
                source_consistency,
            ]
        ).astype(np.float32)
        return {
            "static13_raw": torch.from_numpy(static13),
            "pb_token_raw": torch.from_numpy(pb_token),
        }

    @torch.no_grad()
    def normalize_for_fold(self, extracted, train_mask, device=None):
        static13 = self._zscore_train_apply(extracted["static13_raw"], train_mask)
        pb_token_raw = extracted["pb_token_raw"]
        pb_shape = tuple(pb_token_raw.shape)
        flat_pb = self._as_numpy(pb_token_raw).reshape(pb_shape[0], -1)
        pb_token = self._zscore_train_apply(flat_pb, train_mask).reshape(pb_shape)
        output_device = train_mask.device if device is None else torch.device(device)
        return {
            "static13": torch.from_numpy(static13).to(output_device),
            "pb_token": torch.from_numpy(pb_token).to(output_device),
        }


class ScrnaBranch(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.static = StaticEncoder()
        self.dynamic = (
            StageSourceTransformerEncoder()
            if encoder == "transformer"
            else HybridProgressionEncoder()
        )
        self.fusion = FiLMFusion()
        self.classifier = ResidualClassifier()

    def forward(self, batch):
        static = self.static(batch)
        dynamic = self.dynamic(batch)
        latent = self.fusion(static, dynamic)
        return self.classifier(latent), latent


class LinearLogitMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 0.10], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, bulk_logit, scrna_logit):
        return (
            self.weight[0] * bulk_logit
            + self.weight[1] * scrna_logit
            + self.bias
        )


class SCMG(nn.Module):
    """SCMG with separate bulk and scRNA branches and late-logit fusion."""

    def __init__(self):
        super().__init__()
        self.bulk = BulkBranch()
        transformer_branch = ScrnaBranch("transformer")
        _consume_linear_initialization(64, 3)
        self.sc_models = nn.ModuleList(
            [transformer_branch, ScrnaBranch("hybrid")]
        )
        self.mixer = LinearLogitMixer()

    def forward(self, x_bulk, edge_index, sc_batch):
        _, bulk_logit = self.bulk(x_bulk, edge_index)
        sc_outputs = [model(sc_batch) for model in self.sc_models]
        scrna_logit = torch.stack([item[0] for item in sc_outputs], dim=0).mean(dim=0)
        final_logit = self.mixer(bulk_logit, scrna_logit)
        return final_logit, bulk_logit, scrna_logit
