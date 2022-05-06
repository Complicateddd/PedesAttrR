from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES

from models.backbone.coatNet import CoAtNet_4


def build_backbone(key, multi_scale=False):

    model_dict = {
        'resnet34': 512,
        'resnet18': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'tresnet': 2432,
        'swin_s': 768,
        'swin_b': 1024,
        'vit_s': 768,
        'vit_b': 768,
        'bninception': 1024,
        'tresnetM': 2048,
        'tresnetL': 2048,
        'swin_b_fpn':1024,
        'swin_l':1536,
        'swin_l_fpn':1536,
        'swin_l_384':1536,
        'CoAtNet_4':1536,
        'convnext_l':1536,
        'convnext_xl':2048



    }
    # BACKBONE.register("CoAtNet_4",CoAtNet_4())

    print(BACKBONE.keys())

    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

