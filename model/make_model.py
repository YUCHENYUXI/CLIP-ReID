from timm.layers import  trunc_normal_
import torch
import torch.nn as nn
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip.clip import tokenize as tokenizer
from .clip import clip

_tokenizer = _Tokenizer()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        # classifier
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.clip = clip_model
        self.ve = clip_model.visual
        self.te = clip_model.transformer
# #
        # clip_model.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask()
        # )

        # clip_model.vocab_size = vocab_size
        # clip_model.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # clip_model.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # clip_model.ln_final = LayerNorm(transformer_width)

        # clip_model.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # clip_model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
 
# #
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self,dpac ):
        # x, label=None, cam_label= None, view_label=None
        # dpac {'aer': aer,'rgb': rgb,'pid': pid,'cid': camid}

        if self.model_name == 'RN50':
            exit(0)
        #     image_features_last, image_features, image_features_proj = self.ve(x) #B,512  B,128,512
        #     img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
        #     img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
        #     img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if dpac['cid'] is not  None:
                cv_embed = self.sie_coe * self.cv_embed[dpac['cid']]
            else:
                cv_embed = None
                
            ve = self.ve
            te = self.te
            # rgb
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.view(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] NLD
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            if cv_emb is not None: 
                x[:,0] = x[:,0] + cv_emb
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
            
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            x11 = self.transformer.resblocks[:11](x) 
            x12 = self.transformer.resblocks[11](x11) 
            x11 = x11.permute(1, 0, 2)  # LND -> NLD  
            x12 = x12.permute(1, 0, 2)  # LND -> NLD  

            x12 = self.ln_post(x12)  

            if self.proj is not None:
                xproj = x12 @ self.proj   

            return x11, x12, xproj
            # aer

            # # #
            image_features_last, image_features, image_features_proj = self.ve(x, cv_embed) #B,512  B,128,512
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        print("\033[91mLoading JIT archive failed, loading state_dict instead\033[0m")
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model