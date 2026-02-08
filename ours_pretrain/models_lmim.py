# References:
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F


class MAEEncoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # ---------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio is not None:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


class LMIMViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
        (Baseline Version: Standard Decoder, Single Input)
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.model_maeenc = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio,
                                       norm_layer)
        num_patches = self.model_maeenc.patch_embed.num_patches
        # ---------------------------------------------------------------------

        # Momentum Teacher (Kept for Target Generation)
        self.model_maeenc_momentum = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio,
                                                norm_layer)

        for param_b, param_m in zip(self.model_maeenc.parameters(), self.model_maeenc_momentum.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        # ---------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        # [BASELINE CHANGE] Use Standard Transformer Blocks, NOT Cross-Attention
        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, embed_dim,
            bias=True)  # decoder to patch
        # ---------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.model_maeenc.patch_embed.num_patches ** .5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                   x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Standard Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, target, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        mask = mask.view(mask.shape[0], -1)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # [Step 1] Masked Encoder
        latent, mask, ids_restore = self.model_maeenc(imgs, mask_ratio)

        # [Step 2] Standard Decoder (No Cross Attention)
        pred = self.forward_decoder(latent, ids_restore)

        # [Step 3] Generate Target (Using Frozen Teacher)
        with torch.no_grad():
            target, _, _ = self.model_maeenc_momentum(imgs)
            target = target[:, 1:, :]
            target = F.layer_norm(target, (target.size(-1),))  # normalize over feature-dim

        # [Step 4] Calculate Loss (Reconstruction Only)
        loss = self.forward_loss(imgs, pred, target, mask)

        return loss, pred, mask, imgs


def mae_vit_small_patch4_dec512d8b(**kwargs):
    model = LMIMViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = LMIMViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


# set recommended archs
mae_vit_small_patch4 = mae_vit_small_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa