from initialize import parse_config
import os
import torch
import sys
from pathlib import Path
from .custom_loader import custom_load_model

FASTCHAT_PATH = Path("../FastChat")  
sys.path.insert(0, str(FASTCHAT_PATH.resolve()))

print(f"FastChat path: {FASTCHAT_PATH.resolve()}")

# Import FastChat modules after path setup
from fastchat.llm_judge.gen_model_answer import run_eval
import fastchat.model.model_adapter

print(f"FastChat module loaded from: {os.path.dirname(fastchat.__file__)}")

# Save the original get_model_adapter function
original_get_adapter = fastchat.model.model_adapter.get_model_adapter

# Create wrapper function
def my_get_adapter(model_path):
    print(f"Getting adapter for {model_path}")
    
    base_model_path = None
    if "cackerman" in model_path or "checkpoint" in model_path:
        # Try to get the base model path from params.txt
        try:
            if "checkpoint" in model_path:
                fname = model_path.split("/checkpoint-")[0]+"/params.txt"
            else:
                fname = "./" + model_path.split("cackerman/")[1]+"/params.txt"
                
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    params = parse_config(f.read())
                base_model_path = params.get('model_path')
                print(f"Found base_model_path: {base_model_path}")
        except Exception as e:
            print(f"Error getting base_model_path: {e}")
    
    # Use the original method to get the adapter
    selected_adapter = original_get_adapter(model_path)
    
    # If we have a base model path and it contains "llama-3", override the conversation template
    if base_model_path and "llama-3" in base_model_path.lower():
        print(f"Using Llama3 conversation template because base_model_path {base_model_path} contains 'llama-3'")
        
        # Import the function here to ensure it's available
        from fastchat.conversation import get_conv_template
        
        # Override the get_default_conv_template method to return the Llama-3 template
        def get_llama3_template(self, model_path):
            print("Returning Llama-3 conversation template")
            return get_conv_template("llama-3")
        
        import types
        selected_adapter.get_default_conv_template = types.MethodType(get_llama3_template, selected_adapter)
    
    # Patch the adapter's load_model method 
    original_adapter_load = selected_adapter.load_model
    
    def wrapped_adapter_load(self, model_path, kwargs):
        print("Adapter load_model called")
        device = kwargs.get('device', 'cuda')
        num_gpus = kwargs.get('num_gpus', 1)
        max_gpu_memory = kwargs.get('max_gpu_memory', None)
        dtype = kwargs.get('dtype', torch.float16)
        return custom_load_model(model_path, device, num_gpus, max_gpu_memory, dtype, **kwargs)
    
    import types
    selected_adapter.load_model = types.MethodType(wrapped_adapter_load, selected_adapter)
    
    return selected_adapter
    
# Install wrapper
fastchat.model.model_adapter.get_model_adapter = my_get_adapter
print("Monkey patching complete")

checkpoint_path = "ft_randalias_0to31_learn_interleaved_stdmixsafecombo4_orthrandembedembed_mult0.2/checkpoint-639"
ftwoc_path = "ft_lora_0to31_interleaved_mixedlmsys4_none_mult0/checkpoint-500"#"ft_lora_randalias_0to31_interleaved_mixedlmsys4_none_mult0/checkpoint-600"

# Try to load params
fname = "./" + checkpoint_path.replace("cackerman/","").split("/checkpoint-")[0]+"/params.txt"
if os.path.exists(fname):
    with open(fname, "r") as f:
        params = parse_config(f.read())
    base_model_path = params['model_path']
    print(f"Loaded base_model_path: {base_model_path}")
else: 
    print(f"params file {fname} not found")

def main():
    models = [
        #(params['model_path'], "untuned"),
        (checkpoint_path, "stdmixsafecombo4"),
        #(ftwoc_path, "ftwoc_orig")
    ]
    
    for model_path, model_id in models:
        print(f"Running evaluation for model: {model_id} at path: {model_path}")
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        run_eval(
            model_path=model_path,
            model_id=model_id,
            question_file=f"{FASTCHAT_PATH}/fastchat/llm_judge/data/mt_bench/question.jsonl",
            answer_file=f"results/{model_id}_answers.jsonl",
            question_begin=None, 
            question_end=None, 
            max_new_token=1024,
            num_choices=1,
            num_gpus_per_model=1,
            num_gpus_total=1,
            max_gpu_memory=None,
            dtype=torch.float16,
            revision="main",
        )

if __name__ == "__main__":
    main()