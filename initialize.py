### Installs/imports
#!pip install torch transformers datasets tabulate
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import gc
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
import random
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import json
import ast


from enum import Enum
class TaskType(Enum):
    SelfVOther = "SvO"
    HumanVModel = "HvM"
    Individual = "individual"
    SelfVOther_Continuation = "SvO_continuation"
    HumanVModel_Continuation = "HvM_continuation"
    Individual_Continuation = "Ind_continuation"
    
class SteeringTarget(Enum):
    NONE = "raw"
    FINAL_ONLY = "finalonly"
    FINAL_ASST = "finalasst"
    FIRST_USER_FINAL_ASST = "firstuserfinalasst"
    ALL_USER_FINAL_ASST = "alluserfinalasst"
    ALL_USER_PLUS_FINAL = "allplusfinal"
    INTERLEAVED = "interleaved"
    INTERLEAVED_ROLES = "interleavedroles"
    TEXT_ONLY = "textonly"
    TEXT_ONLY_INTERLEAVED = "interleavedtextonly"
    DUMMY_USER = "dummyuser"
    DUMMY_ASST = "dummyasst"
    
class SteeringVectorType(Enum):
    RANDOM_ORTH_EMBED = "orthrandembed"
    RANDOM_ORTH0 = "orthrand0"
    RANDOM_ORTH4 = "orthrand4"
    RANDOM_ORTH4_SCALED = "orthrand4scaled"
    RANDOM_ORTH_ALL = "orthrandall"
    RANDOM_ORTH_ALL_NS = "orthrandallns"
    SELFREC16 = "selfrec16"
    SELFREC16_SCALED = "selfrec16scaled"
    SELFRECALL = "selfrecall"
    UA8 = "ua8"
    UAALL = "uall"
    ROLE_CONTRAST = "rolecontrast"
    RANDOM_LIKE = "randomua"
    ONEHOT = "onehot"
    LEARNED = "learned"
    NINAUA = "ninaua"
    NONE = "none"
    
class MSType(Enum):
    CRAZY1 = "crazy1"
    STDMIX_SAFE = "stdmixsafe"
    STDMIX_SAFE2 = "stdmixsafe2"
    STDMIX_SAFE_CRAZY = "stdmixsafecrazy"
    STDMIX_SAFE2_CRAZY2 = "stdmixsafe2crazy2"
    STDMIX_SAFE2_CRAZY1 = "stdmixsafe2crazy1"
    STDMIX_SAFE3 = "stdmixsafe3"
    STDMIX_SAFE_NORMAL = "stdmixsafenormal"
    STDMIX_SAFE_COMBO = "stdmixsafecombo"
    STDMIX_SAFE_COMBO_NA = "stdmixsafecombona"
    STDMIX_SAFE_COMBO_RA = "stdmixsafecombora"
    STDMIX_SAFE_COMBO_NARA = "stdmixsafecombonara"
    STDMIX_SAFE_COMBO_NARA_FA = "stdmixsafecombonarafa"
    STDMIX_SAFE_COMBO3 = "stdmixsafecombo3"
    STDMIX_SAFE_COMBO4 = "stdmixsafecombo4"
    STDMIX_SAFE_COMBO5 = "stdmixsafecombo5"
    STDMIX_SAFE_COMBO6 = "stdmixsafecombo6"
    BOTH10 = "both10"
    BOTH10_ALT = "both10alt"
    BOTH10_ALT7 = "both10alt7"
    BOTH8_PLUS_CRAZY7 = "both8pluscrazy7"
    BOTH8_PLUS_CRAZY13 = "both8pluscrazy13"
    BOTH8_PLUS_CRAZY30 = "both8pluscrazy30"
    BOTH8_PLUS_CRAZY41 = "both8pluscrazy41"
    BOTH8_PLUS_CRAZY42 = "both8pluscrazy42"
class RoleAliasType(Enum):
    NONE = "none"
    NAMES = "names"
    ROLES = "roles"
    RANDOM = "random"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def parse_config(txt):
    d = {}
    for line in txt.splitlines():
        if '=' not in line: continue
        k, v = line.split('=', 1)
        v = v.strip()
        d[k.strip()] = (
            True if v=='True' else
            False if v=='False' else
            None if v=='None' else
            int(v) if v.isdigit() else
            float(v) if v.replace('.','',1).isdigit() else
            set(v.strip('{}').replace("'","").split(', ')) if v.startswith('{') else
            eval(v) if v.startswith('[') or ('.' in v and v.split('.')[-1].isupper()) else
            v
        )
    return d

def load_model(model_path, base_model_path, bnb = False, device = None, train = False, lora = False):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps' 
        else:
            device = 'cpu'
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    #if not train: _ = torch.set_grad_enabled(False)
    #else: _ = torch.set_grad_enabled(True)
    if not train:
        with torch.set_grad_enabled(False):
            if bnb:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=HF_TOKEN).to(device)
    else:
        with torch.set_grad_enabled(True):
            if bnb:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto")
            else:
                if lora:
                    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=HF_TOKEN, device_map="auto").to(device)
                    for param in model.parameters():
                        param.requires_grad = True       
                else: model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=HF_TOKEN).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=HF_TOKEN)
    model.tokenizer = tokenizer
    if model.tokenizer.pad_token is None:
        #model.tokenizer.pad_token = model.tokenizer.eos_token
        #model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        new_pad_token = model.tokenizer.eos_token#"[PAD]"#
        num_added_tokens = model.tokenizer.add_special_tokens({'pad_token': new_pad_token})
        model.resize_token_embeddings(len(model.tokenizer))
        model.config.pad_token_id = model.tokenizer.pad_token_id
    return model

def layerlist_to_str(l):
    if len(l) == 0: return ""
    if len(l) == 1: return str(l[0])
    return f"{l[0]}to{l[-1]}"

assistant_fake_tags1 = [
    "assistant",
    "ai",
    "AI",
    "Assistant",
    "AIAssistant",
]

user_fake_tags1 = [
    "user",
    "human",
    "u",
    "h",
    "Human",
    "User",
    "U"
]
assistant_fake_tags2 = [
    "a",
    "A"
]

user_fake_tags2 = [
    "user_input",
    "human_input",
]
format_functions = [
    lambda x: x + ":",
    lambda x: x + ">",
    lambda x: f"<{x}>",
    lambda x: f"({x})",
    lambda x: f"[{x}]",
    lambda x: f"<|{x}|>",
    lambda x: f"({x}):",
    lambda x: f"{x} >",
    lambda x: f"{x} >>",
    lambda x: f"|{x}|",
]
ASSISTANT_ALIASES = [random.choice(format_functions)(tag) for tag in assistant_fake_tags1]
USER_ALIASES = [random.choice(format_functions)(tag) for tag in user_fake_tags1]
#USER_ALIASES = ['<Human> ', 'User: ', '<h> ', '(u) ', '<h> ', '(u) ']
#ASSISTANT_ALIASES = ['<AI> ', 'Assistant: ', '(assistant) ', '<AI Assistant> ']
#ASSISTANT_ALIASES = [ "a: ","A: "]
#USER_ALIASES = ["user_input: ","human_input: ",]
def get_rand_alias(aliastype):
    user_aliases = USER_ALIASES
    asst_aliases = ASSISTANT_ALIASES
    assert aliastype in ['user','asst'], print(f"aliastype {aliastype} is invalid")
    return random.choice(user_aliases) if aliastype == 'user' else random.choice(asst_aliases)
    
def map_to_vectors(steer_vec_type, color_layers, colormult, steermult, peft_model_id, model, filedir="./", post=True):
    device = model.device
    pre_norms = {
        0: 0.41,
        1: 1.11,
        2: 1.67,
        3: 2.34,
        4: 3.08,
        5: 3.99,
        6: 4.70,
        7: 5.39,
        8: 6.17,
        9: 6.66,
        10: 7.20,
        11: 7.52,
        12: 7.89,
        13: 8.34,
        14: 9.04,
        15: 9.81,
        16: 10.75,
        17: 12.10,
        18: 13.48,
        19: 15.19,
        20: 16.39,
        21: 17.82,
        22: 20.03,
        23: 21.99,
        24: 24.35,
        25: 26.43,
        26: 28.78,
        27: 31.36,
        28: 34.76,
        29: 37.87,
        30: 41.16,
        31: 46.51
    }   
    post_norms = {
        'embed': 0.39,
        0: 1.10,
        1: 1.67,
        2: 2.33,
        3: 3.07,
        4: 3.98,
        5: 4.70,
        6: 5.39,
        7: 6.17,
        8: 6.66,
        9: 7.19,
        10: 7.52,
        11: 7.89,
        12: 8.34,
        13: 9.03,
        14: 9.81,
        15: 10.75,
        16: 12.10,
        17: 13.48,
        18: 15.19,
        19: 16.39,
        20: 17.82,
        21: 20.03,
        22: 21.99,
        23: 24.35,
        24: 26.43,
        25: 28.78,
        26: 31.36,
        27: 34.76,
        28: 37.87,
        29: 41.16,
        30: 46.51,
        31: 59.42
    }
    if steer_vec_type == SteeringVectorType.ROLE_CONTRAST:
        fname = f'{filedir}steering_vectors_role_contrast.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_role_contrast = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_role_contrast[layer][0].clone().detach()*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_role_contrast[layer][0].clone().detach()*steermult).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.LEARNED:
        fname = f'{peft_model_id}/steering_vectors.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_learned = pickle.load(f)
        coloring_vectors_user = {layer: torch.tensor(steering_vectors_learned['user'][layer]).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: torch.tensor(steering_vectors_learned['asst'][layer]).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_LIKE:
        def create_similar_tensor(original_tensor):
            mean = original_tensor.mean()
            std = original_tensor.std()
            
            new_tensor = torch.normal(mean, std, size=(len(original_tensor),))
            
            return new_tensor

        fname = f'{filedir}steering_vectors_roles_summed_newlines_nonorm_3.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_ua=pickle.load(f)
        randvec = create_similar_tensor(steering_vectors_ua[8][0].clone())
        coloring_vectors_user = {layer: (randvec.clone()*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (randvec.clone()*steermult).to(device) for layer in color_layers}

        ofname = 'steering_vectors_random_ua.pkl'
        with open(ofname, 'wb') as f:
            pickle.dump(coloring_vectors_asst, f)
    elif steer_vec_type == SteeringVectorType.ONEHOT:
        user_dim=42
        asst_dim=714
        coloring_vectors_user = {layer: torch.zeros(model.config.hidden_size, device=device, dtype=model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: torch.zeros(model.config.hidden_size, device=device, dtype=model.dtype) for layer in color_layers}

        for layer in coloring_vectors_user:
            coloring_vectors_user[layer][user_dim] = 1 * colormult
            coloring_vectors_asst[layer][asst_dim] = 1 * steermult
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH_ALL:
        fname = f'{filedir}steering_vectors_orth_user_all.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user_all = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_all.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst_all = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_orth_user_all[layer]*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst_all[layer]*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH_ALL_NS:
        fname = f'{filedir}steering_vectors_orth_user_all.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user_all = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_all.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst_all = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_orth_user_all[layer]*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst_all[layer]*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH_EMBED:
        fname = f'{filedir}steering_vectors_orth_user_embed.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user_embed = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_embed.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst_embed = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_orth_user_embed*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst_embed*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH0:
        fname = f'{filedir}steering_vectors_orth_user_l0.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user0 = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_l0.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst0 = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_orth_user0/torch.norm(steering_vectors_orth_user0,p=2)*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst0/torch.norm(steering_vectors_orth_asst0,p=2)*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH4:
        fname = f'{filedir}steering_vectors_orth_user_l4.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user4 = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_l4.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst4 = pickle.load(f)
        #convert to float32
        steering_vectors_orth_user4 = steering_vectors_orth_user4.to(torch.float32)
        steering_vectors_orth_asst4 = steering_vectors_orth_asst4.to(torch.float32)
        coloring_vectors_user = {layer: (steering_vectors_orth_user4/torch.norm(steering_vectors_orth_user4,p=2,dtype=torch.float32)*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst4/torch.norm(steering_vectors_orth_asst4,p=2,dtype=torch.float32)*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.RANDOM_ORTH4_SCALED:
        fname = f'{filedir}steering_vectors_orth_user_l4.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_user4 = pickle.load(f)
        fname = f'{filedir}steering_vectors_orth_asst_l4.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_orth_asst4 = pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_orth_user4*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])*colormult).to(device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_orth_asst4*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])*steermult).to(device).to(model.dtype) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.SELFREC16:
        fname = f'{filedir}steering_vectors_newbalancedtask3_meandiff_projectoutnuisance.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_selfrec=pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_selfrec[16][8].clone().detach()*-1*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_selfrec[16][8].clone().detach()*steermult).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.SELFREC16_SCALED:
        fname = f'{filedir}steering_vectors_newbalancedtask3_meandiff_projectoutnuisance.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_selfrec=pickle.load(f)
        coloring_vectors_user = {layer: ((steering_vectors_selfrec[16][8]/torch.norm(steering_vectors_selfrec[16][8])).clone().detach()*-1*colormult*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: ((steering_vectors_selfrec[16][8]/torch.norm(steering_vectors_selfrec[16][8])).clone().detach()*steermult*(post_norms[layer] if post==True or layer=='embed' else pre_norms[layer])).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.SELFRECALL:
        fname = f'{filedir}steering_vectors_newbalancedtask3_meandiff_projectoutnuisance.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_selfrec=pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_selfrec[layer][8].clone().detach()*-1*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_selfrec[layer][8].clone().detach()*steermult).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.UA8:
        fname = f'{filedir}steering_vectors_roles_summed_newlines_nonorm_3.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_ua=pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_ua[8][0].clone().detach()*-1*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_ua[8][0].clone().detach()*steermult).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.UAALL:
        fname = f'{filedir}steering_vectors_roles_summed_newlines_nonorm_3.pkl'
        assert os.path.exists(fname), f"File {fname} not found"
        with open(fname, 'rb') as f:
            steering_vectors_ua=pickle.load(f)
        coloring_vectors_user = {layer: (steering_vectors_ua[layer][0].clone().detach()*-1*colormult).to(device) for layer in color_layers}
        coloring_vectors_asst = {layer: (steering_vectors_ua[layer][0].clone().detach()*steermult).to(device) for layer in color_layers}
    elif steer_vec_type == SteeringVectorType.NINAUA:
        with open("training_config.json", 'r') as file:
            data = json.load(file) 
        coloring_vectors_user = {layer: ((torch.tensor(data['intervention_settings']['user_vector']) / torch.norm(torch.tensor(data['intervention_settings']['user_vector']), p=2))*colormult).to(model.device).to(model.dtype) for layer in color_layers}
        coloring_vectors_asst = {layer: ((torch.tensor(data['intervention_settings']['assistant_vector']) / torch.norm(torch.tensor(data['intervention_settings']['assistant_vector']), p=2))*colormult).to(model.device).to(model.dtype) for layer in color_layers}
        
    elif steer_vec_type == SteeringVectorType.NONE:
        coloring_vectors_user = {}
        coloring_vectors_asst = {}

    return coloring_vectors_user, coloring_vectors_asst
