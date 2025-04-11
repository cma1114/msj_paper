from peft import PeftModel
from enhanced_hooking_model import HookedModel
from initialize import load_model as load_llama_model
from initialize import *
from .model_wrapper import ModelWrapper

def custom_load_model(
    model_path,
    device,
    num_gpus,
    max_gpu_memory,
    dtype,
    **kwargs
):
    print("HERE")
    if "checkpoint" in model_path:
        fname = model_path.split("/checkpoint-")[0]+"/params.txt"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                params = parse_config(f.read())
        base_model_path = params['model_path']
        print(f"model_path={model_path}. base_model_path={base_model_path}")
        model = load_llama_model(model_path, base_model_path, bnb = False)
        ###model = PeftModel.from_pretrained(model, model_path)
        if params['steer_vec_type'] != SteeringVectorType.NONE:
            model = HookedModel(model)
            model.config.hook_params_file = fname
            model.config.model_path = model_path
            model = ModelWrapper(model)

    elif "cackerman" in model_path:
        fname = model_path.split("cackerman/")[1]+"/params.txt"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                params = parse_config(f.read())
        base_model_path = params['model_path']
        print(f"model_path={model_path}. base_model_path={base_model_path}")
        model = load_llama_model(model_path, base_model_path, bnb = False)
        ###model = PeftModel.from_pretrained(model, model_path)
        if params['steer_vec_type'] != SteeringVectorType.NONE:
            model = HookedModel(model)
            model.config.hook_params_file = fname
            model.config.model_path = model_path.replace("cackerman/","")
            model = ModelWrapper(model)
    else:
        print(f"model_path={model_path}. base_model_path={model_path}")
        model = load_llama_model(model_path, model_path, bnb = False)    
        print(f"model.device={model.device}")
    return model, model.tokenizer
