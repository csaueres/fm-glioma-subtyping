import os
from functools import partial
import timm
import torch
from TOKENS import set_environment_variables

#for virchow
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

#musk
from models.musk import utils, modeling

        
def get_encoder(model_name):
    set_environment_variables()
    print('loading model checkpoint: ', model_name)
    if model_name == 'uni_v1':
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    elif model_name== 'gigapath':
        #model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=local_tile_encoder_path)
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name== 'virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    elif(model_name)=='musk':
        model = timm.create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    #print_network(model)
    model.name=model_name
    model.eval()

    return model
