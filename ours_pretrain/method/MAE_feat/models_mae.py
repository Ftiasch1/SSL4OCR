# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone

    Supports two modes:
    1. Standard MAE: reconstruct pixels (use_momentum_encoder=False)
    2. Feature reconstruction: reconstruct features from momentum encoder (use_momentum_encoder=True)
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
                 norm_pix_loss=False,
                 use_momentum_encoder=False,
                 momentum=0.996):
        super().__init__()

        self.use_momentum_encoder = use_momentum_encoder
        self.momentum = momentum

        # --------------------------------------------------------------------------
        # MAE encoder specifics (Student Encoder)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim

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
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Momentum Encoder (Teacher) - only created if use_momentum_encoder=True
        if self.use_momentum_encoder:
            self.patch_embed_momentum = PatchEmbed(img_size, patch_size, in_chans,
                                                   embed_dim)
            self.cls_token_momentum = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_momentum = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim),
                requires_grad=False)

            self.blocks_momentum = nn.ModuleList([
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer) for i in range(depth)
            ])
            self.norm_momentum = norm_layer(embed_dim)

            # Will be initialized in initialize_weights()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

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

        # Decoder prediction head
        if self.use_momentum_encoder:
            # Feature reconstruction: output embed_dim
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, embed_dim, bias=True)
        else:
            # Pixel reconstruction: output patch_size^2 * 3
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.use_momentum_encoder:
            self.pos_embed_momentum.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        if self.use_momentum_encoder:
            torch.nn.init.normal_(self.cls_token_momentum, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # Initialize momentum encoder after weights are initialized
        if self.use_momentum_encoder:
            self._init_momentum_encoder()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_momentum_encoder(self):
        """Initialize momentum encoder as a copy of student encoder"""
        if not self.use_momentum_encoder:
            return

        # Copy patch_embed weights
        self.patch_embed_momentum.load_state_dict(self.patch_embed.state_dict())

        # Copy cls_token and pos_embed
        self.cls_token_momentum.data.copy_(self.cls_token.data)
        self.pos_embed_momentum.data.copy_(self.pos_embed.data)

        # Copy transformer blocks
        for blk, blk_m in zip(self.blocks, self.blocks_momentum):
            blk_m.load_state_dict(blk.state_dict())

        # Copy norm layer
        self.norm_momentum.load_state_dict(self.norm.state_dict())

        # Freeze momentum encoder (no gradients)
        for param in self.patch_embed_momentum.parameters():
            param.requires_grad = False
        self.cls_token_momentum.requires_grad = False
        self.pos_embed_momentum.requires_grad = False
        for blk_m in self.blocks_momentum:
            for param in blk_m.parameters():
                param.requires_grad = False
        for param in self.norm_momentum.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update - 确保完全无梯度"""
        if not self.use_momentum_encoder:
            return

        m = self.momentum

        # ✅ 修复：显式 detach
        for param_s, param_m in zip(self.patch_embed.parameters(),
                                    self.patch_embed_momentum.parameters()):
            param_m.data.copy_(
                m * param_m.data + (1 - m) * param_s.data.detach()
            )

        # cls_token
        self.cls_token_momentum.data.copy_(
            m * self.cls_token_momentum.data + (1 - m) * self.cls_token.data.detach()
        )

        # Transformer blocks
        for blk_s, blk_m in zip(self.blocks, self.blocks_momentum):
            for param_s, param_m in zip(blk_s.parameters(), blk_m.parameters()):
                param_m.data.copy_(
                    m * param_m.data + (1 - m) * param_s.data.detach()
                )

        # Norm layer
        for param_s, param_m in zip(self.norm.parameters(),
                                    self.norm_momentum.parameters()):
            param_m.data.copy_(
                m * param_m.data + (1 - m) * param_s.data.detach()
            )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

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

    def forward_encoder(self, x, mask_ratio):
        """Student encoder with masking"""
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    @torch.no_grad()
    def forward_encoder_momentum(self, x):
        """Momentum encoder without masking (no gradients)"""
        # embed patches
        x = self.patch_embed_momentum(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed_momentum[:, 1:, :]

        # append cls token (no masking)
        cls_token = self.cls_token_momentum + self.pos_embed_momentum[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks_momentum:
            x = blk(x)
        x = self.norm_momentum(x)

        return x

    def forward_decoder(self, x, ids_restore):
        """Decoder with self-attention (standard MAE decoder)"""
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, target=None):
        """
        Compute loss for either pixel or feature reconstruction

        imgs: [N, 3, H, W]
        pred: [N, L, D] where D = patch_size^2*3 (pixel) or embed_dim (feature)
        mask: [N, L], 0 is keep, 1 is remove
        target: [N, L, embed_dim] features from momentum encoder (optional)
        """
        if self.use_momentum_encoder and target is not None:
            # Feature reconstruction loss
            # Normalize target features (layer norm over feature dimension)
            target = F.layer_norm(target, (target.size(-1),))

            # MSE loss in feature space
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        else:
            # Pixel reconstruction loss (standard MAE)
            target = self.patchify(imgs)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6) ** .5

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass for training

        Args:
            imgs: [N, 3, H, W] input images
            mask_ratio: ratio of patches to mask

        Returns:
            loss: scalar loss
            pred: [N, L, D] predictions (pixels or features)
            mask: [N, L] binary mask
            imgs: [N, 3, H, W] original images
        """
        # Student encoder (with masking)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Decoder
        pred = self.forward_decoder(latent, ids_restore)

        # Compute loss
        if self.use_momentum_encoder:
            with torch.no_grad():
                target_full = self.forward_encoder_momentum(imgs)  # [N, L+1, D]
                # ✅ 修复：detach + clone，完全切断计算图
                target = target_full[:, 1:, :].detach().clone()

            loss = self.forward_loss(imgs, pred, mask, target=target)

            # Update momentum encoder (EMA)
            self._update_momentum_encoder()
        else:
            # Standard MAE: pixel reconstruction
            loss = self.forward_loss(imgs, pred, mask, target=None)

        return loss, pred, mask, imgs


def mae_vit_small_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
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
    model = MaskedAutoencoderViT(
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
mae_vit_small_patch4 = mae_vit_small_patch4_dec512d8b  # decoder: 512 dim, 2 blocks
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks