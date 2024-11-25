import torch
import clip

from .prompts import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES_SINGLE

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adapter(torch.nn.Module):
    def __init__(self, clip_model, classnames, templates, template_index=0):
        
        super().__init__()

        # Set vision model
        self.backbone = clip_model.visual

        # Obtain class prototypes
        text_embeddings_avg, text_embeddings = self.get_text_prototypes(clip_model, classnames)
        self.text_embeddings_avg, self.text_embeddings = text_embeddings_avg.cpu(), text_embeddings.cpu()
        self.templates = templates
        # Set the template index to choose an individual template for logits calculation
        self.template_index = template_index

        # Pass the selected template embedding to LinearProbeHead
        self.adapter = LinearProbeHead(self.text_embeddings[:, self.template_index, :], clip_model.logit_scale)

        # move to device
        self.to(device).float()

    def forward(self, x):

        # Forward trough vision encoder
        feats = self.backbone(x)

        # Forward classifier
        out = self.adapter(feats)

        return out

    @staticmethod
    def get_text_prototypes(clip_model, classnames):
        clip_model.eval()

        print("Extracting text prototypes from class names...")
        with torch.no_grad():
            text_embeddings = []
            for text in classnames:
                if isinstance(classnames, dict):
                    tokens = clip.tokenize(
                        [template.format(classnames[text]) for template in IMAGENET_TEMPLATES_SELECT]).to(device)
                else:
                    tokens = clip.tokenize(
                        [template.format(text) for template in IMAGENET_TEMPLATES_SELECT]).to(device)

                prototype = clip_model.encode_text(tokens.to(device))
                text_embeddings.append(prototype)

        text_embeddings = torch.stack(text_embeddings)
        text_embeddings_avg = text_embeddings.mean(1)
        return text_embeddings_avg, text_embeddings

    @property
    def visual(self):
        return self.clip_model.visual  # Exposes the visual encoder
        

class LinearProbeHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale, init="zero_shot"):
        super().__init__()
        self.logit_scale = logit_scale
        self.logit_scale.requires_grad = False
        self.init = init
        self.zero_shot_prot = zero_shot_prot.clone()

        if init == "zero_shot":
            print("Using Zero-Shot initialization in Linear Probing", end="\n")
            self.prototypes = torch.nn.Parameter(zero_shot_prot.clone())
        else:
            print("Using RANDOM initialization in Linear Probing", end="\n")
            self.prototypes = torch.nn.Parameter(
                torch.nn.init.kaiming_normal_(torch.empty(zero_shot_prot.shape)))

    def forward(self, features):
        # Get trained prototype
        prototypes = self.prototypes

        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits, image_features_norm