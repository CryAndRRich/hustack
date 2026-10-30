import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from sapiens2 import Sapiens2

SAPIENS2 = ("Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm",
            "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
            "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
            "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth",
            "Upper_Teeth", "Tongue")
CIHP_FROM_SAPIENS2 = (
    ("Background",), ("Apparel",), ("Hair",), ("Apparel",), ("Eyeglass",), ("Upper_Clothing",),
    ("Upper_Clothing", "Lower_Clothing"), ("Upper_Clothing",), ("Left_Sock", "Right_Sock"), ("Lower_Clothing",),
    ("Torso",), ("Apparel",), ("Lower_Clothing",), ("Face_Neck",), ("Left_Upper_Arm", "Left_Lower_Arm", "Left_Hand"),
    ("Right_Upper_Arm", "Right_Lower_Arm", "Right_Hand"), ("Left_Upper_Leg", "Left_Lower_Leg", "Left_Foot"),
    ("Right_Upper_Leg", "Right_Lower_Leg", "Right_Foot"), ("Left_Shoe",), ("Right_Shoe",))


class RMSNorm32(nn.RMSNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


def _up(i, o):
    return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.InstanceNorm2d(o), nn.SiLU(inplace=True)]


def _pw(i, o):
    return [nn.Conv2d(i, o, 1), nn.InstanceNorm2d(o), nn.SiLU(inplace=True)]


class Head(nn.Module):
    def __init__(self, dim, n_cls, deconv=(512, 256, 128, 64), conv=(64, 64)):
        super().__init__()
        chans = (dim,) + deconv
        self.deconv_layers = nn.Sequential(*[m for i, o in zip(chans, deconv) for m in _up(i, o)])
        chans = deconv[-1:] + conv
        self.conv_layers = nn.Sequential(*[m for i, o in zip(chans, conv) for m in _pw(i, o)])
        self.conv_seg = nn.Conv2d(conv[-1], n_cls, 1)

    def forward(self, x):
        return self.conv_seg(self.conv_layers(self.deconv_layers(x)))


class Sapiens2Seg(nn.Module):
    def __init__(self, img, n_cls, freeze, ckpt_blocks, arch="sapiens2_0.4b"):
        super().__init__()
        self.backbone = Sapiens2(arch=arch, img_size=(img, img), patch_size=16, out_type="featmap", final_norm=True,
                                 with_cls_token=True, use_tokenizer=False, pos_embed_rope_dtype="fp32")
        for m in self.backbone.modules():
            if type(m) is nn.RMSNorm:
                m.__class__ = RMSNorm32
        self.decode_head = Head(self.backbone.embed_dims, n_cls)
        self.freeze, self.ckpt_blocks = freeze, ckpt_blocks
        if freeze:
            bb = self.backbone
            for p in [bb.cls_token, bb.storage_tokens, *bb.patch_embed.parameters(),
                      *(q for b in bb.blocks[:freeze] for q in b.parameters())]:
                p.requires_grad_(False)

    def forward(self, x):
        bb, B, grad = self.backbone, len(x), torch.is_grad_enabled()
        with torch.set_grad_enabled(grad and self.freeze == 0):
            x, hw = bb.patch_embed(x)
            x = torch.cat([bb.cls_token.expand(B, -1, -1), bb.storage_tokens.expand(B, -1, -1), x], 1)
            rope = bb.rope_embed(H=hw[0], W=hw[1])
            for blk in bb.blocks[:self.freeze]:
                x = blk(x, rope=rope)
        for i, blk in enumerate(bb.blocks[self.freeze:]):
            x = checkpoint(blk, x, rope, use_reentrant=False) if grad and i < self.ckpt_blocks else blk(x, rope=rope)
        x = bb.ln1(x)[:, bb.num_extra_tokens:].transpose(1, 2).reshape(B, -1, *hw)
        return F.avg_pool2d(self.decode_head(x), 2)


def load_sapiens2(net, path):
    sd = load_file(path)
    gw, gb = sd.pop("decode_head.conv_seg.weight"), sd.pop("decode_head.conv_seg.bias")
    rows = [[SAPIENS2.index(n) for n in src] for src in CIHP_FROM_SAPIENS2]
    sd["decode_head.conv_seg.weight"] = torch.stack([gw[r].mean(0) for r in rows])
    sd["decode_head.conv_seg.bias"] = torch.stack([gb[r].mean(0) for r in rows])
    net.load_state_dict(sd, strict=True)
