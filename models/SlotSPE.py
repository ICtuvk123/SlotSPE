import torch
import torch.nn as nn
from models.slot_attention import MultiHeadSlotAttention, gumbel_topk_st, parallel_topk_st
from models.event_gated_slot_attention import EventGatedSlotAttention
from models.event_grounding import FrozenEventBank
from models.cross_modal_adapter import DyKoAdapter
from models.gc_rsm import GeneConditionedRankSpaceModulation
from models.transformer import IterativeCrossAttTransformer, Transformer
from models.omics_encoder import SNN_Block, WSI_Mlp
from utils.loss_func import NLLSurvLoss
import numpy as np
import torch.nn.functional as F


class MoESlotDecoder(nn.Module):
    def __init__(self, 
        dim, 
        num_slots, 
        num_classes=4, 
        temperature=0.01, 
        topk_ratio=0.25,
        top_k_method='parallel_topk_st'
    ):

        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.dim = dim
        self.temperature = temperature
        self.top_k_method = top_k_method
        self.k = max(1, int(num_slots * topk_ratio))

        self.map = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.decoder = nn.Linear(dim, num_classes)

        self.pred_keep_slot = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),  # No bias to avoid learning a constant offset
        )

    def forward(self, slots):
        # --- 1. Map slots ---
        slots = self.map(slots)  # (B, S, D)

        # --- 2. Slot-level logits ---
        slot_logits = self.decoder(slots)  # (B, S, C)

        # --- 3. Predict keep scores ---
        keep_score = self.pred_keep_slot(slots).squeeze(-1)  # (B, S)
        # print("keep_score shape:", keep_score.shape)

        # --- 4. Gumbel-Softmax-K ---
        if self.top_k_method == 'gumbel_topk_st':
            hard_keep, _ = gumbel_topk_st(keep_score, temperature=self.temperature, k=self.k)
        elif self.top_k_method == 'parallel_topk_st':
            hard_keep, _ = parallel_topk_st(keep_score, temperature=self.temperature, k=self.k)
        else:
            raise ValueError(f"Invalid top_k_method: {self.top_k_method}")

        # --- 5. Slot gate (weighted by soft attention) ---
        slot_gate = torch.softmax(keep_score / self.temperature, dim=-1)  # (B, S)
        slot_gate = slot_gate * hard_keep  # (B, S)
        slot_gate = slot_gate / (slot_gate.sum(dim=1, keepdim=True) + 1e-8)

        # --- 6. Final weighted prediction ---
        x = torch.einsum('bs,bsc->bc', slot_gate, slot_logits)  # (B, C)

        return x, slot_gate, hard_keep

class PlaceholderQueryGenerator(nn.Module):
    def __init__(self, dim, mode="linear_from_detached", num_queries=None):
        super().__init__()
        self.mode = mode
        self.dim = dim

        if mode == "learned_token":
            self.token = nn.Parameter(torch.randn(1, 1, dim))
        elif mode == "range_init":
            assert num_queries is not None, "num_queries must be provided for range_init"
            positions = torch.arange(num_queries).unsqueeze(1).float()  # (G, 1)
            self.register_buffer('range_embed', positions)
            self.embedding = nn.Embedding(num_queries, dim)
        elif mode == "wsi_patches":
            self.frozen_mlp = nn.Linear(dim, dim)
            for param in self.frozen_mlp.parameters():
                param.requires_grad = False
    def forward(self, x):
        if self.mode == "zeros":
            return torch.zeros_like(x)
        elif self.mode == "random":
            return torch.randn_like(x)
        elif self.mode == "learned_token":
            B, N, _ = x.shape
            return self.token.expand(B, N, -1)
        elif self.mode == "range_init":
            B, _, _ = x.shape
            query_embed = self.embedding(self.range_embed.squeeze(1).long())  # (G, D)
            query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, G, D)
            return query_embed
        elif self.mode == "wsi_patches":
            with torch.no_grad():
                query_embed = self.frozen_mlp(x.detach())
            return query_embed
        else:
            raise ValueError(f"Unknown query init mode: {self.mode}")


class ReconstructionHead(nn.Module):
    def __init__(self, dim, num_heads=8, mode="range_init", num_queries=None):
        super(ReconstructionHead, self).__init__()
        self.query_gen = PlaceholderQueryGenerator(dim, mode=mode, num_queries=num_queries)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, slots, mask=None):
        query = self.query_gen(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # PyTorch expects: True = mask out

        recon, _ = self.cross_attn(query, slots, slots, key_padding_mask=key_padding_mask)

        recon = recon + self.mlp(self.norm(recon))
        return recon



class SlotSPE(nn.Module):
    def __init__(
            self,
            args,
            omic_names = [],
            omic_input_dim = None)->None:
        super(SlotSPE, self).__init__()

        # ---> general props
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = args.n_classes

        # ---> wsi encoder
        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim
        self.wsi_projection_dropout_p = float(getattr(args, "wsi_projection_dropout", 0.0))
        self.fusion_dropout_p = float(getattr(args, "fusion_dropout", 0.0))
        self.lambda_decoder_loss = float(getattr(args, "lambda_decoder_loss", 1.0))
        self.vl_adapter_type = str(getattr(args, "vl_adapter_type", "none")).casefold()
        self.vl_adapter_reduction = int(getattr(args, "vl_adapter_reduction", 4))
        self.lambda_vl_alignment = float(getattr(args, "lambda_vl_alignment", 0.0))
        self.gc_rsm_mode = str(getattr(args, "gc_rsm_mode", "none")).casefold()
        self.gc_rsm_rank = int(getattr(args, "gc_rsm_rank", 16))
        self.gc_rsm_hidden_dim = int(getattr(args, "gc_rsm_hidden_dim", 128))
        self.gc_rsm_dropout = float(getattr(args, "gc_rsm_dropout", 0.0))
        self.gc_rsm_residual_scale = float(getattr(args, "gc_rsm_residual_scale", 1.0))
        for name, value in (
            ("wsi_projection_dropout", self.wsi_projection_dropout_p),
            ("fusion_dropout", self.fusion_dropout_p),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {value}")
        if self.lambda_decoder_loss < 0.0:
            raise ValueError("lambda_decoder_loss must be non-negative")
        if self.vl_adapter_type not in {"none", "dyko"}:
            raise ValueError("vl_adapter_type must be 'none' or 'dyko'")
        if self.lambda_vl_alignment < 0.0:
            raise ValueError("lambda_vl_alignment must be non-negative")
        if self.gc_rsm_mode not in {"none", "feature"}:
            raise ValueError("gc_rsm_mode must be 'none' or 'feature'")

        # ---> omics props
        self.omics_input_dim = omic_input_dim
        self.init_per_path_model(self.omic_sizes, args.rna_format)  # --->dim 256

        # ---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        # ---> wsi mlp
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=self.wsi_projection_dim)
        self.gc_rsm_feature = None
        if self.gc_rsm_mode == "feature":
            self.gc_rsm_feature = GeneConditionedRankSpaceModulation(
                in_dim=self.wsi_embedding_dim,
                out_dim=self.wsi_projection_dim,
                gene_dim=self.wsi_projection_dim,
                rank=self.gc_rsm_rank,
                hidden_dim=self.gc_rsm_hidden_dim,
                dropout=self.gc_rsm_dropout,
                residual_scale=self.gc_rsm_residual_scale,
            )
        self.wsi_projection_dropout = nn.Dropout(self.wsi_projection_dropout_p)
        self.fusion_dropout = nn.Dropout(self.fusion_dropout_p)
        self.last_loss_components = {}

        # ---> slot attention
        self.event_bank = None
        self.path_adapter = None
        self.text_adapter = None
        if args.slot_attention_type == "original":
            self.slot_attention_wsi = MultiHeadSlotAttention(dim=self.wsi_projection_dim,
                                                        num_slots=args.slot_num_wsi,
                                                        iters=args.slot_iters,
                                                        heads=8)
        elif args.slot_attention_type == "event_gated":
            if not args.event_bank_path:
                raise ValueError(
                    "event_bank_path is required for slot_attention_type='event_gated'"
                )
            self.event_bank = FrozenEventBank(
                args.event_bank_path, trainable=args.event_bank_trainable
            )
            encoder_feature_spaces = {
                "conch": "conch_contrastive",
                "conch_v1_5": "conch_v1_5_patch_768",
            }
            raw_patch_feature_space = encoder_feature_spaces.get(
                str(args.slot_feature_encoder).casefold(), "unknown"
            )
            if self.vl_adapter_type == "dyko":
                if raw_patch_feature_space != "conch_v1_5_patch_768":
                    raise ValueError("DyKo adapters require slot_feature_encoder='conch_v1_5'")
                if self.event_bank.feature_space != "titan_text_768":
                    raise ValueError(
                        "DyKo adapters require a TITAN event bank marked 'titan_text_768'"
                    )
                if self.wsi_embedding_dim != 768 or self.event_bank.embedding_dim != 768:
                    raise ValueError("DyKo CONCH v1.5/TITAN adapters require 768-D inputs")
                if self.event_bank.trainable:
                    raise ValueError("Keep the TITAN event bank frozen when training adapters")
                self.path_adapter = DyKoAdapter(
                    self.wsi_embedding_dim, reduction=self.vl_adapter_reduction
                )
                self.text_adapter = DyKoAdapter(
                    self.event_bank.embedding_dim, reduction=self.vl_adapter_reduction
                )
                self.patch_event_feature_space = "conch_v1_5_titan_dyko_adapter_768"
                self.event_feature_space = self.patch_event_feature_space
            else:
                self.patch_event_feature_space = raw_patch_feature_space
                self.event_feature_space = self.event_bank.feature_space

            if self.patch_event_feature_space != self.event_feature_space:
                raise ValueError(
                    "Slot feature encoder and event bank are incompatible: "
                    f"encoder={args.slot_feature_encoder!r} maps to "
                    f"{self.patch_event_feature_space!r}, bank={self.event_feature_space!r}. "
                    "CONCH v1.5 patch and TITAN text tokens require --vl_adapter_type dyko."
                )
            self.slot_attention_wsi = EventGatedSlotAttention(
                dim=self.wsi_projection_dim,
                num_slots=args.slot_num_wsi,
                iters=args.slot_iters,
                heads=8,
                event_embedding_dim=self.event_bank.embedding_dim,
                event_projection_dim=args.event_projection_dim,
                tau_event=args.tau_event,
                event_gate_start_iter=args.event_gate_start_iter,
                patch_event_support_mode=args.patch_event_support_mode,
                delta_patch_event=args.delta_patch_event,
                beta_patch_event=args.beta_patch_event,
                tau_patch_event=args.tau_patch_event,
                delta_sem=args.delta_sem,
                beta_sem=args.beta_sem,
                delta_vis=args.delta_vis,
                beta_vis=args.beta_vis,
                use_agreement_gate=args.use_agreement_gate,
                lambda_js=args.lambda_js,
                lambda_event=args.lambda_event,
                event_residual_dropout=args.event_residual_dropout,
                store_all_iterations=args.store_all_iterations,
            )
        else:
            raise ValueError(f"Unknown slot_attention_type: {args.slot_attention_type}")
        self.slot_attention_omic = MultiHeadSlotAttention(dim=self.wsi_projection_dim,
                                                     num_slots=args.slot_num_omics,
                                                     iters=args.slot_iters,
                                                     heads=8)

        # ---> slot decoder
        self.slot_decoder_wsi = MoESlotDecoder(dim=self.wsi_projection_dim, 
                                               num_slots=args.slot_num_wsi, 
                                               num_classes=self.num_classes, 
                                               temperature=args.temperature, 
                                               topk_ratio=args.topk_ratio,
                                               top_k_method=args.top_k_method)
        self.slot_decoder_omic = MoESlotDecoder(dim=self.wsi_projection_dim, 
                                               num_slots=args.slot_num_omics, 
                                               num_classes=self.num_classes, 
                                               temperature=args.temperature, 
                                               topk_ratio=args.topk_ratio,
                                               top_k_method=args.top_k_method)

        # ---> transformer
        self.self_attention_wsi = Transformer(dim=self.wsi_projection_dim)
        self.self_attention_omic = Transformer(dim=self.wsi_projection_dim)
        self.cross_attention = IterativeCrossAttTransformer(dim=self.wsi_projection_dim)

        # ---> classification
        if args.bag_loss == "cox_surv":
            self.to_logits = nn.Linear(self.wsi_projection_dim*3, 1)
        else:
            self.to_logits = nn.Linear(self.wsi_projection_dim*3, self.num_classes)

        # decoder loss for moe-style decoder
        self.loss_fn = NLLSurvLoss(alpha=args.alpha_surv)


        # reconstruction head
        num_q = len(self.omic_sizes) if isinstance(self.omic_sizes, list) else self.omic_sizes
        self.reconstruction_head_omic = ReconstructionHead(self.wsi_projection_dim, mode="range_init", num_queries=num_q)
        self.reconstruction_head_wsi = ReconstructionHead(self.wsi_projection_dim, mode="wsi_patches")
        self.reconstruction_head_joint = ReconstructionHead(self.wsi_projection_dim, mode="range_init",num_queries=num_q)



    def init_per_path_model(self, omic_sizes, omics_format):
        if omics_format == "Pathways":  # TODO: gene_embeddings is not used
            self.num_pathways = len(omic_sizes)
            hidden = [self.wsi_projection_dim, self.wsi_projection_dim]
            # strategy 1, same with SurvPath
            sig_networks = []
            for input_dim in omic_sizes:
                fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
                for i, _ in enumerate(hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.sig_networks = nn.ModuleList(sig_networks)

            # # strategy 2, less parameters
            # self.shared_input_dim = max(omic_sizes)
            # fc_omic = [SNN_Block(dim1=self.shared_input_dim, dim2=hidden[0])]
            # for i, _ in enumerate(hidden[1:]):
            #     fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            # self.sig_networks = nn.Sequential(*fc_omic)

        elif omics_format == "GeneEmbedding":
            # features from gene embeddings
            self.sig_networks = SNN_Block(dim1=768, dim2=self.wsi_projection_dim)


        elif omics_format == "RNASeq":
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim,
                                          dim2=self.wsi_projection_dim)

        else:
            raise ValueError('omics_format should be pathways, gene or groups')

    def _encode_omics(self, kwargs):
        if self.args.rna_format == "Pathways":
            x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
                        enumerate(x_omic)]
            x_omics = torch.stack(h_omic)
            return x_omics.permute(1, 0, 2)
        x_omic = kwargs["x_omics"]
        return self.sig_networks(x_omic)

    @staticmethod
    def _pool_omics_context(x_omics):
        if x_omics.ndim == 3:
            return x_omics.mean(dim=1)
        if x_omics.ndim == 2:
            return x_omics
        raise ValueError(f"Unexpected omics tensor shape: {tuple(x_omics.shape)}")



    def forward(self, **kwargs):
        x_wsi = kwargs['x_wsi']  # (batch_size, num_patches, dim)
        z_event = kwargs.get("z_conch")
        event_embeddings = self.event_bank.embeddings if self.event_bank is not None else None
        if self.vl_adapter_type == "dyko":
            if z_event is None or event_embeddings is None:
                raise ValueError("DyKo adapter mode requires patch and event tokens")
            z_event = self.path_adapter(z_event)
            event_embeddings = self.text_adapter(event_embeddings)
            if getattr(self.args, "reuse_slot_features_as_conch", False):
                if x_wsi.shape != z_event.shape:
                    raise ValueError("Reused CONCH v1.5 patch tokens changed shape unexpectedly")
                x_wsi = z_event

        self.last_loss_components = {}
        # Encoder
        omic_missing = kwargs["omic_missing"]
        x_omics = self._encode_omics(kwargs)
        omics_context = self._pool_omics_context(x_omics)

        x_wsi_base = self.wsi_mlp(x_wsi)
        if self.gc_rsm_feature is not None:
            x_wsi_clean = self.gc_rsm_feature(x_wsi, x_wsi_base, omics_context)
        else:
            x_wsi_clean = x_wsi_base
        x_wsi_proj = self.wsi_projection_dropout(x_wsi_clean)

        if not self.training:
            if not omic_missing:
                pass
            else:
                print("Omics missing, reconstructing omics from WSI")
                slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
                recon_omic_joint = self.reconstruction_head_joint(x_omics, slots_omic_from_wsi)
                x_omics = recon_omic_joint

        event_details = None
        vl_alignment_loss = x_wsi_proj.new_zeros((), dtype=torch.float32)
        if self.args.slot_attention_type == "event_gated":
            need_alignment = self.training and self.lambda_vl_alignment > 0.0
            slot_result = self.slot_attention_wsi(
                x_wsi_proj,
                z_event,
                event_embeddings,
                mask=kwargs.get("patch_mask"),
                z_feature_space=self.patch_event_feature_space,
                event_feature_space=self.event_feature_space,
                return_event_details=self.args.return_event_details,
                return_alignment_loss=need_alignment,
            )
            if self.args.return_event_details or need_alignment:
                slots_wsi, internal_event_details = slot_result
                vl_alignment_loss = internal_event_details["vl_alignment_loss"]
            else:
                internal_event_details = None
                slots_wsi = slot_result
            if self.args.return_event_details:
                event_details = internal_event_details
                event_details["event_ids"] = list(self.event_bank.event_ids)
                event_details["event_names"] = list(self.event_bank.event_names)
        else:
            slots_wsi = self.slot_attention_wsi(x_wsi_proj) # (batch_size, num_slots, dim)
        # print(x_omics.shape)
        slots_omic = self.slot_attention_omic(x_omics)

        # ---> decoder
        # wsi decoder
        logits_wsi, _, wsi_keep_slots = self.slot_decoder_wsi(slots_wsi)
        # omic decoder
        logits_omic, _, omic_keep_slots = self.slot_decoder_omic(slots_omic)

        # ---> cross attention
        x_inter = self.cross_attention(slots_wsi, slots_omic)
        # ---> self attention
        wsi_intra = self.self_attention_wsi(slots_wsi, mask=wsi_keep_slots.bool())
        omic_intra = self.self_attention_omic(slots_omic, mask=omic_keep_slots.bool())

        # survival prediction
        x = torch.cat([x_inter.mean(dim=1), wsi_intra.mean(dim=1), omic_intra.mean(dim=1)], dim=1)
        x = self.fusion_dropout(x)
        logits = self.to_logits(x)

        if self.training:
            h_omic_bag_origin = x_omics.detach()
            # ---> reconstruction
            recon_omic = self.reconstruction_head_omic(x_omics, slots_omic, mask=omic_keep_slots.bool())
            recon_mse = F.mse_loss(recon_omic, h_omic_bag_origin)

            slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
            recon_omic_joint = self.reconstruction_head_joint(x_omics, slots_omic_from_wsi)
            recon_mse += F.mse_loss(recon_omic_joint, h_omic_bag_origin)

            recon_wsi = self.reconstruction_head_wsi(x_wsi_clean, slots_wsi, mask=wsi_keep_slots.bool())
            recon_wsi = F.normalize(recon_wsi, dim=-1)
            target_wsi = F.normalize(x_wsi_clean, dim=-1)
            recon_loss_wsi = 1-F.cosine_similarity(recon_wsi, target_wsi, dim=-1).mean()

            # ---> reconstruction loss
            recon_loss = recon_mse + recon_loss_wsi

            # ---> loss for slot decoder
            y = kwargs["y"]
            c = kwargs["c"]
            loss_decoder = self.loss_fn(logits_wsi, y=y, c=c,t=None) + self.loss_fn(logits_omic, y=y, c=c,t=None)
            loss_decoder = loss_decoder / y.shape[0]

            # ---> total auxiliary loss
            weighted_decoder_loss = self.lambda_decoder_loss * loss_decoder
            weighted_recon_loss = self.args.lambda_recon_loss * recon_loss
            weighted_vl_alignment_loss = self.lambda_vl_alignment * vl_alignment_loss
            aux_loss = weighted_decoder_loss + weighted_recon_loss + weighted_vl_alignment_loss
            self.last_loss_components = {
                "decoder_loss": loss_decoder.detach(),
                "reconstruction_loss": recon_loss.detach(),
                "weighted_decoder_loss": weighted_decoder_loss.detach(),
                "weighted_reconstruction_loss": weighted_recon_loss.detach(),
                "vl_alignment_loss": vl_alignment_loss.detach(),
                "weighted_vl_alignment_loss": weighted_vl_alignment_loss.detach(),
                "aux_loss": aux_loss.detach(),
            }
        else:
            # ---> total auxiliary loss
            aux_loss = 0.0
        if self.args.slot_attention_type == "event_gated" and self.args.return_event_details:
            return logits, aux_loss, event_details
        return logits, aux_loss
