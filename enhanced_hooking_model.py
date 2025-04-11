### Flexible hooking at arbitrary layers and tokens

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class GetActivations:
    specific_pos_read_target: List[Dict[Union[int, str], List[int]]] = field(default_factory=list) #batch_size list of layer: positions dicts
    continuous_read_target: List[List[Union[int, str]]] = field(default_factory=list) # batch_size list of layers (position is always the one being generated)
    get_at_end: bool = True

    def __post_init__(self):
        if not self.specific_pos_read_target and not self.continuous_read_target:
            raise ValueError("Either specific_pos_read_target or continuous_read_target must be non-empty")
            
@dataclass
class ActivationHookMap:
    specific_pos_write_target: List[Dict[Union[int, str], Union[  # batch_size list of layerwise dicts of
                                    Dict[int, torch.Tensor], # {pos: vector} dicts
                                    Tuple[torch.Tensor, List[int]] # (vector, pos_list) tuples; for when same tensor is to be applied to all positions       
                                    ]]] = None  
    continuous_write_target: List[Dict[Union[int, str], torch.Tensor]] = None  # batch_size list of layers: vector dicts
    at_end: bool = True

    def __post_init__(self):
        if not self.specific_pos_write_target and not self.continuous_write_target:
            raise ValueError("Either specific_pos_write_target or continuous_write_target must be non-empty")

@dataclass
class AddActivations(ActivationHookMap):
    scale_to_sim: Optional[float] = None
    scale_by_sim_vecs: Optional[List[Dict[Union[int, str], torch.Tensor]]] = None

@dataclass
class ZeroActivations(ActivationHookMap):
    sign: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sign is not None and self.sign not in (-1, 1):
            raise ValueError("sign must be None, -1, or 1")

class HookedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__() 
        self.model = model
        self.tokenizer = model.tokenizer
        self.config = model.config
        self.device = model.device
        self.dtype = model.dtype
        self.scale_to_residnorm=False
        self.hooks = []
        self.transformer_blocks = self.get_blocks(self.model)
        self.clear_all_hooks()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def train(self):
        self.model.train()
        return self
        
    def eval(self):
        self.model.eval()
        return self
    
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        """ Get the ModuleList containing the transformer blocks from a model. """
        def numel_(mod):
            if isinstance(mod, nn.Module):
                num_elements = sum(p.numel() for p in mod.parameters())
                return num_elements
            else:
                print(f"Non-module object encountered: {mod}")
                return 0
        model_numel = numel_(model)
        candidates = [mod for mod in model.modules() if isinstance(mod, nn.ModuleList) and numel_(mod) > .5 * model_numel]
        assert len(candidates) == 1, f'Found {len(candidates)} ModuleLists with >50% of model params.'
        return candidates[0]

    def clear_all_hooks(self):
        for block in self.transformer_blocks:
            block._forward_hooks.clear()
            block._forward_pre_hooks.clear()
            block._backward_hooks.clear()
        hooked_model = self.model.model if hasattr(self.model, 'peft_config') else self.model
        hooked_model.model.embed_tokens._forward_hooks.clear()
        hooked_model.model.embed_tokens._forward_pre_hooks.clear()
        hooked_model.model.embed_tokens._backward_hooks.clear()
        for hook in self.hooks:
            hook.remove()

    def run_hooked_model(
        self,
        inputs: Dict[str, torch.Tensor],
        generate: bool = False,
        sampling_kwargs: Dict[str, Any] = {},
        activation_targets: List[Union[AddActivations, GetActivations, ZeroActivations]] = []
        ,num_logits_to_keep=0
        ,training=False
        ,scale_to_residnorm=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[int, Dict[int, torch.Tensor]]]]]:
        self.scale_to_residnorm=scale_to_residnorm
        self.clear_all_hooks()
        batch_size = inputs['input_ids'].shape[0]
        activations = [{} for _ in range(batch_size)]
        
        for activations_target in activation_targets:
            if isinstance(activations_target, ZeroActivations):
                for struct_name in ['specific_pos_write_target', 'continuous_write_target']:
                    struct = getattr(activations_target, struct_name)
                    if struct:
                        layers = set(layer for item in struct for layer in item)
                        for layer in layers:
                            module = self.transformer_blocks[layer]
                            if struct_name == 'specific_pos_write_target': 
                                hook = self.add_zero_hook(module, layer, struct, activations_target.at_end, activations_target.sign)
                            else:
                                hook = self.add_continuous_zero_hook(module, layer, struct, activations_target.at_end, activations_target.sign)
                            self.hooks.append(hook)

            elif isinstance(activations_target, AddActivations):
                hooked_model = self.model.model if hasattr(self.model, 'peft_config') else self.model 
                for struct_name in ['specific_pos_write_target', 'continuous_write_target']:
                    struct = getattr(activations_target, struct_name)
                    if struct:
                        layers = set(layer for item in struct for layer in item)
                        for layer in layers:
                            if layer == 'embed':
                                module = hooked_model.model.embed_tokens
                                add_at_end = True
                            else:
                                module = self.transformer_blocks[layer]
                                add_at_end = activations_target.at_end
                            if struct_name == 'specific_pos_write_target': 
                                hook = self.add_write_hook(module, layer, struct, add_at_end, activations_target.scale_to_sim, activations_target.scale_by_sim_vecs)
                            else:
                                hook = self.add_continuous_write_hook(module, layer, struct, add_at_end, activations_target.scale_to_sim, activations_target.scale_by_sim_vecs)
                            self.hooks.append(hook)

            elif isinstance(activations_target, GetActivations):
                layers_to_hook = set()
                for batch_item in activations_target.specific_pos_read_target:
                    layers_to_hook.update(batch_item.keys())
                layers_to_hook.update(set(layer for batch_item in activations_target.continuous_read_target for layer in batch_item))
                
                for layer in layers_to_hook:
                    module = self.transformer_blocks[layer]
                    specific_positions = [batch_item.get(layer, []) for batch_item in activations_target.specific_pos_read_target]
                    continuous_read = any(layer in batch_item for batch_item in activations_target.continuous_read_target)
                    hook = self.add_read_hook(module, layer, specific_positions, continuous_read, activations_target.get_at_end, activations, inputs['input_ids'].shape[1], training)
                    self.hooks.append(hook)

        try:
            if not training:
                self.model.eval()
                with torch.no_grad():
                    inputs = {k: v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    if generate:
                        outputs = self.model.generate(**inputs, **sampling_kwargs)
                    else:
                        outputs = self.model(**inputs, num_logits_to_keep=num_logits_to_keep)

            else:
                inputs = {k: v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if generate:
                    outputs = self.model.generate(**inputs, **sampling_kwargs)
                else:
                    outputs = self.model(**inputs, num_logits_to_keep=num_logits_to_keep)
                    #print(f"outputs.loss.requires_grad={outputs.loss.requires_grad}")

        finally:
            for hook in self.hooks:
                hook.remove()

        if any(isinstance(activations_target, GetActivations) for activations_target in activation_targets):
            return outputs, activations
        else:
            return outputs
                

    def add_write_hook(self, module, layer, batch_data, add_at_end, scale_to_sim=None, scale_by_sim_vecs=None):
        tracked_data={}
        def parse_struct(batch_data, stream):
            batch_size, seq_len, hidden_size = stream.shape        
            all_batch_indices = []
            all_pos_indices = []
            all_vecs = []
            all_scales = []
            for batch_idx, batch_item in enumerate(batch_data): #each item in the batch can have different target positions, so there's really no way to vectorize this
                if layer not in batch_item: # layerwise dict of [{pos: vector} dicts or (vector, pos_list) tuples] 
                    continue

                layer_data = batch_item[layer]
                if isinstance(layer_data, tuple):
                    vector, positions = layer_data
                    if not positions:
                        continue
                    positions = torch.tensor(positions, device=stream.device)
                    vectors = vector.unsqueeze(0).expand(len(positions), -1).to(stream.device)
                else:
                    positions = list(layer_data.keys())
                    if not positions:
                        continue
                    positions = torch.tensor(positions, device=stream.device)
                    vectors = torch.stack(list(layer_data.values())).to(stream.device)

                if scale_to_sim is not None:
                    activations = stream[batch_idx, positions]
                    if self.scale_to_residnorm:
                        # Normalize vectors to match activation magnitudes
                        activation_norms = torch.norm(activations, dim=-1, keepdim=True)
                        vector_norms = torch.norm(vectors, dim=-1, keepdim=True)
                        vectors = vectors * (activation_norms / vector_norms)
                    ###scales = adjust_tensor_to_cosine_similarity(activations, vectors, scale_to_sim)
                    A_prime = rotate_to_cosine_similarity(activations, vectors, torch.tensor(scale_to_sim))
                    all_vecs.append(A_prime)
                else:
                    if scale_by_sim_vecs is not None and layer in scale_by_sim_vecs[batch_idx]:
                        activations = stream[batch_idx, positions]
                        scales = scale_vector_by_sim(activations, scale_by_sim_vecs[batch_idx][layer])
                    else:
                        #print(f"len(positions)={len(positions)}")
                        scales = torch.ones(len(positions), device=stream.device)
                    all_vecs.append(vectors) 
                    all_scales.append(scales)

                ##tracked_data[(batch_idx, tuple(positions.tolist()))] = {'magnitudes': torch.norm(activations, dim=-1), 'scales': scales}                

                all_batch_indices.append(torch.full((len(positions),), batch_idx, dtype=torch.long, device=stream.device))
                all_pos_indices.append(positions)

            return all_batch_indices, all_pos_indices, all_vecs, all_scales

        if add_at_end:
            def hook(module, inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                if output.shape[1] == 1: return outputs  # Skip during generation
                        
                all_batch_indices, all_pos_indices, all_vecs, all_scales = parse_struct(batch_data, output)
                if all_batch_indices:
                    batch_tensor = torch.cat(all_batch_indices)
                    pos_tensor = torch.cat(all_pos_indices)
                    vec_tensor = torch.cat(all_vecs).to(output.dtype)
                    
                    if len(all_scales)>0:
                        if self.scale_to_residnorm:
                            current_resid = output[batch_tensor, pos_tensor]
                            # ------------------------------------------------------------------
                            # 1. Compute dot product α between current_resid and steering vector
                            #    This is effectively a projection scalar for each position
                            #    In einsum notation:
                            #    current_resid: (N, D)
                            #    vec_tensor:    (N, D)
                            #    "nd,nd->n"
                            # ------------------------------------------------------------------
                            alpha = torch.einsum("nd,nd->n", current_resid, vec_tensor).unsqueeze(-1)
                            # ------------------------------------------------------------------
                            # 2. Subtract out α * steering_vector (projection)
                            #    Removing the vector’s component from the residual
                            # ------------------------------------------------------------------
                            output[batch_tensor, pos_tensor] = current_resid - (alpha / vec_tensor.norm(dim=-1, keepdim=True)**2) * vec_tensor
                            
                            scale_tensor = torch.norm(output[batch_tensor, pos_tensor], dim=-1, keepdim=True)
                            output[batch_tensor, pos_tensor] += scale_tensor * vec_tensor                
                            
                            ### Hack just to make Nina's subtraction of other vector projection to work ###
                            user_vector = vec_tensor[0]
                            assistant_vector = vec_tensor[-1]
                            # Create a mask: rows equal to the first row are assumed to be user-assigned.
                            is_user = torch.all(vec_tensor == user_vector, dim=-1)
                            # Now, assign the reverse vector:
                            reverse_vec_tensor = torch.where(
                                is_user.unsqueeze(-1), 
                                assistant_vector.expand_as(vec_tensor), 
                                user_vector.expand_as(vec_tensor)
                            )
                            ####
                            current_resid = output[batch_tensor, pos_tensor]
                            beta = torch.einsum("nd,nd->n", current_resid, reverse_vec_tensor).unsqueeze(-1)
                            output[batch_tensor, pos_tensor] = current_resid - beta * reverse_vec_tensor
                            
                            #print(f"output.shape={output.shape}, output[batch_tensor, pos_tensor].shape={output[batch_tensor, pos_tensor].shape}, scale_tensor.shape={scale_tensor.shape}")
                            #print(f"scale_tensor.norm={torch.mean(torch.abs(torch.norm(scale_tensor,dim=-1)))}")
                        else:
                            scale_tensor = torch.cat(all_scales).unsqueeze(-1)
                            #print(f"in write, layer={layer}, positions={pos_tensor}, output.shape[1]={output.shape[1]}")#, scaled_tensor={scale_tensor * vec_tensor}")
                            output[batch_tensor, pos_tensor] += scale_tensor * vec_tensor                
                    
                    else: output[batch_tensor, pos_tensor] = vec_tensor.to(dtype=output.dtype)
                    """
                    for (batch_idx, pos_tuple), data in tracked_data.items():
                        positions = torch.tensor(pos_tuple, device=output.device)
                        new_mags = torch.norm(output[batch_idx, positions], dim=-1)
                        mag_ratios = new_mags / data['magnitudes']
                        print(f"Layer {layer}, positions {pos_tuple}:")
                        print(f"  Scales: {data['scales']}")
                        print(f"  Magnitude ratios: {mag_ratios.mean().item():.3f}")
                        break
                    """
                return (output,) + outputs[1:] if isinstance(outputs, tuple) else output        
            return module.register_forward_hook(hook)
        else:
            def hook(module, inputs):
                if inputs[0].shape[1] == 1: return inputs  # Hack to turn this off during generation
                inputs = list(inputs)  # Convert tuple to list for mutability
                input = inputs[0]
        
                all_batch_indices, all_pos_indices, all_vecs, all_scales = parse_struct(batch_data, input)
        
                if all_batch_indices:
                    batch_tensor = torch.cat(all_batch_indices)
                    pos_tensor = torch.cat(all_pos_indices)
                    vec_tensor = torch.cat(all_vecs).to(input.dtype)
        
                    if len(all_scales)>0:
                        if self.scale_to_residnorm:
                            #current_resid = input[batch_tensor, pos_tensor]
                            #alpha = torch.einsum("nd,nd->n", current_resid, vec_tensor).unsqueeze(-1)
                            #input[batch_tensor, pos_tensor] = current_resid - alpha * vec_tensor
                            scale_tensor = torch.norm(input[batch_tensor, pos_tensor], dim=-1, keepdim=True)
                            #print(f"input.shape={input.shape}, input[batch_tensor, pos_tensor].shape={input[batch_tensor, pos_tensor].shape}, scale_tensor.shape={scale_tensor.shape}")
                            #print(f"scale_tensor.norm={torch.mean(torch.norm(scale_tensor,dim=-1))}")
                        else:
                            scale_tensor = torch.cat(all_scales).unsqueeze(-1)
                        input[batch_tensor, pos_tensor] += scale_tensor * vec_tensor
                        #print(f"pos_tensor={pos_tensor}\nvec_tensor={vec_tensor}")
                    else: input[batch_tensor, pos_tensor] = vec_tensor.to(dtype=input.dtype)

                return tuple(inputs)  # Convert back to tuple to maintain integrity     
            return module.register_forward_pre_hook(hook)

            
    def add_continuous_write_hook(self, module, layer, batch_data, add_at_end, scale_to_sim=None, scale_by_sim_vecs=None):
        if add_at_end:
            def hook(module, inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                
                batch_tensor = torch.arange(len(batch_data), device=output.device)
                vec_tensor = torch.stack([batch_item[layer].to(output.device) 
                                        for batch_item in batch_data])
                
                activations = output[batch_tensor, -1]
                
                #original_magnitudes = torch.norm(activations, dim=-1)
                
                if scale_to_sim is not None:
                    A_prime = rotate_to_cosine_similarity(activations, vec_tensor, torch.tensor(scale_to_sim))
                    output[batch_tensor, -1] = A_prime.to(dtype=output.dtype)
                elif scale_by_sim_vecs is not None:
                    scales = torch.tensor([
                        scale_vector_by_sim(act.unsqueeze(0), scale_by_sim_vecs[i][layer])
                        if layer in scale_by_sim_vecs[i] else 1.0
                        for i, act in enumerate(activations)
                    ], device=output.device)
                    output[batch_tensor, -1] += scales.unsqueeze(-1) * vec_tensor
                else:
                    if self.scale_to_residnorm:
                        current_resid = output[batch_tensor, -1]
                        alpha = torch.einsum("nd,nd->n", current_resid, vec_tensor).unsqueeze(-1)
                        output[batch_tensor, -1] = current_resid - (alpha / vec_tensor.norm(dim=-1, keepdim=True)**2) * vec_tensor
                        residual_norms = torch.norm(output[batch_tensor, -1], dim=-1, keepdim=True)  # Should be [batch_size, 1]
                        scaled_vecs = vec_tensor * residual_norms  
                        output[batch_tensor, -1] += scaled_vecs
                    else:                
                        scales = torch.ones(len(batch_data), device=output.device) 
                        output[batch_tensor, -1] += scales.unsqueeze(-1) * vec_tensor           
                    #print(f"in continuous write, layer={layer}, output.shape[1]={output.shape[1]}")#scaled_tensor={scaled_vecs}")
                
                """
                # Calculate and print magnitude changes
                new_magnitudes = torch.norm(output[batch_tensor, -1], dim=-1)
                mag_ratios = new_magnitudes / original_magnitudes        
                #print(f"Layer {layer}:")
                #print(f"  Scales: {scales}")
                #print(f"  Magnitude ratios: {mag_ratios}")
                """
                return (output,) + outputs[1:] if isinstance(outputs, tuple) else output
            return module.register_forward_hook(hook)
        else:
            def hook(module, inputs):
                inputs = list(inputs)
                input = inputs[0]

                batch_tensor = torch.arange(len(batch_data), device=input.device)
                vec_tensor = torch.stack([batch_item[layer].to(input.device) 
                                        for batch_item in batch_data])
                
                activations = input[batch_tensor, -1]
                
                if scale_to_sim is not None:
                    A_prime = rotate_to_cosine_similarity(activations, vec_tensor, torch.tensor(scale_to_sim))
                    input[batch_tensor, -1] = A_prime.to(dtype=input.dtype)
                elif scale_by_sim_vecs is not None:
                    scales = torch.tensor([
                        scale_vector_by_sim(act.unsqueeze(0), scale_by_sim_vecs[i][layer])
                        if layer in scale_by_sim_vecs[i] else 1.0
                        for i, act in enumerate(activations)
                    ], device=input.device)
                    input[batch_tensor, -1] += scales.unsqueeze(-1) * vec_tensor
                else:
                    if self.scale_to_residnorm:
                        #current_resid = input[batch_tensor, -1]
                        #alpha = torch.einsum("nd,nd->n", current_resid, vec_tensor).unsqueeze(-1)
                        #input[batch_tensor, -1] = current_resid - alpha * vec_tensor
                        residual_norms = torch.norm(input[batch_tensor, -1], dim=-1, keepdim=True)  # Should be [batch_size, 1]
                        scaled_vecs = vec_tensor * residual_norms  
                        input[batch_tensor, -1] += scaled_vecs
                    else:                
                        scales = torch.ones(len(batch_data), device=input.device) 
                        input[batch_tensor, -1] += scales.unsqueeze(-1) * vec_tensor           
            
                return tuple(inputs)        
            return module.register_forward_pre_hook(hook)


    def add_zero_hook(self, module, layer, batch_data, zero_at_end, sign=None):
        def compute_projection(batch_item, batch_idx, stream):
            layer_data = batch_item[layer]
            if isinstance(layer_data, tuple):
                vector, positions = layer_data
                vec_tensor = vector.to(stream.device)
            else:
                positions = list(layer_data.keys())
                vectors = list(layer_data.values())
                vec_tensor = torch.stack(vectors).to(stream.device)

            if not positions: return None, None

            pos_tensor = torch.tensor(positions, device=stream.device)
            activation = stream[batch_idx, pos_tensor]

            # Compute projection (same for both formats)
            vector_norm = torch.sum(vec_tensor ** 2, dim=-1, keepdim=True)
            scalar_proj_coeff = torch.sum(activation * vec_tensor, dim=-1, keepdim=True) / vector_norm
            projection = scalar_proj_coeff * vec_tensor

            if sign is not None:
                sign_mask = (torch.sign(scalar_proj_coeff) == sign).float()
                projection = projection * sign_mask
            return pos_tensor, projection

        if zero_at_end:
            def hook(module, inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                if output.shape[1] == 1: return outputs  # Skip during generation

                for batch_idx, batch_item in enumerate(batch_data):
                    if layer not in batch_item:
                        continue

                    pos_tensor, projection = compute_projection(batch_item, batch_idx, output)
                    if projection is not None: output[batch_idx, pos_tensor] -= projection

                return (output,) + outputs[1:] if isinstance(outputs, tuple) else output            
            return module.register_forward_hook(hook)
        else:
            def hook(module, inputs):
                if inputs[0].shape[1] == 1: return inputs  # Hack to turn this off during generation
                inputs = list(inputs)  # Convert tuple to list for mutability
                input = inputs[0]

                for batch_idx, batch_item in enumerate(batch_data):
                    if layer not in batch_item:
                        continue

                    pos_tensor, projection = compute_projection(batch_item, batch_idx, input)
                    if projection is not None: input[batch_idx, pos_tensor] -= projection
                    
                return tuple(inputs)  # Convert back to tuple to maintain integrity
            return module.register_forward_pre_hook(hook)        
            
    def add_continuous_zero_hook(self, module, layer, batch_data, zero_at_end, sign=None):
        def compute_projection(batch_item, batch_idx, stream):
            vector = batch_item[layer].to(stream.device)
            activation = stream[batch_idx, -1]
            
            scalar_proj_coeff = torch.sum(activation * vector) / torch.sum(vector * vector)
            projection = scalar_proj_coeff * vector
            
            if sign is not None:
                sign_mask = (torch.sign(scalar_proj_coeff) == sign).float()
                projection = projection * sign_mask
            return projection
            
        if zero_at_end:
            def hook(module, inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                
                batch_size, seq_len, hidden_size = output.shape
                
                for batch_idx, batch_item in enumerate(batch_data):
                    if layer not in batch_item:
                        continue
                    projection = compute_projection(batch_item, batch_idx, output)
                    output[batch_idx, -1] -= projection
                
                return (output,) + outputs[1:] if isinstance(outputs, tuple) else output
            return module.register_forward_hook(hook)
        else:
            def hook(module, inputs): 
                inputs = list(inputs)  # Convert tuple to list for mutability
                input = inputs[0]
                batch_size, seq_len, hidden_size = input.shape
                
                for batch_idx, batch_item in enumerate(batch_data):
                    if layer not in batch_item:
                        continue
                    projection = compute_projection(batch_item, batch_idx, input)
                    input[batch_idx, -1] -= projection
                return tuple(inputs)  # Convert back to tuple to maintain integrity            
            return module.register_forward_pre_hook(hook)


    def add_read_hook(self, module, layer, specific_positions, continuous_read, get_at_end, activations, input_len, training):
        def fill_activations(activations, stream):
            batch_size = stream.shape[0]
            # Handle specific positions for input
            for batch_idx in range(batch_size):
                if batch_idx < len(specific_positions):
                    positions = specific_positions[batch_idx]
                    for pos in positions:
                        if pos < input_len and pos < stream.shape[1]:
                            if layer not in activations[batch_idx]:
                                activations[batch_idx][layer] = {}
                            if not training: 
                                activations[batch_idx][layer][pos] = stream[batch_idx, pos, :].detach()
                            else: activations[batch_idx][layer][pos] = stream[batch_idx, pos, :]
            
            # Handle continuous read for new tokens
            if continuous_read:
                if stream.shape[1] == 1:  # During generation with caching
                    for batch_idx in range(batch_size):
                        if layer not in activations[batch_idx]:
                            activations[batch_idx][layer] = {}
                        new_pos = max(activations[batch_idx][layer].keys()) + 1 if activations[batch_idx][layer] else input_len
                        if not training: 
                            activations[batch_idx][layer][new_pos] = stream[batch_idx, 0, :].detach()
                        else: activations[batch_idx][layer][new_pos] = stream[batch_idx, 0, :]
                elif stream.shape[1] > input_len:  # During non-cached forward pass
                    for batch_idx in range(batch_size):
                        if layer not in activations[batch_idx]:
                            activations[batch_idx][layer] = {}
                        for pos in range(input_len, stream.shape[1]):
                            if not training: 
                                activations[batch_idx][layer][pos] = stream[batch_idx, pos, :].detach()
                            else: activations[batch_idx][layer][pos] = stream[batch_idx, pos, :]

        if get_at_end:
            def hook(module, inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                fill_activations(activations, output)         
            return module.register_forward_hook(hook)
        else:
            def hook(module, inputs):
                input = inputs[0]
                fill_activations(activations, input)                         
            return module.register_forward_pre_hook(hook)
        

# Scaling functions 
def adjust_tensor_to_cosine_similarity(A, B, desired_cosine_similarity):
    """
    Computes a scalar multiple of tensor B
    such that the cosine similarity between the adjusted tensor and B
    is equal to desired_cosine_similarity.
   
    Args:
        A: Tensor of shape [num_positions, hidden_size]
        B: Tensor of shape [num_positions, hidden_size]
        desired_cosine_similarity: Target cosine similarity
       
    Returns:
        k (torch.Tensor): Scale factors of shape [num_positions]
    """
    A = A.to(torch.float64)
    B = B.to(torch.float64)

    ###### Normalize B to match A's magnitude position-wise
    #norm_A = torch.norm(A, dim=-1, keepdim=True)  # Shape: [num_positions, 1]
    #norm_B = torch.norm(B, dim=-1, keepdim=True)  # Shape: [num_positions, 1]
    #B = B * (norm_A / norm_B)

    # Normalize B
    ###B = B / B.norm(dim=-1, keepdim=True)
    
    # Compute initial cosine similarity for each position
    dot_AB = torch.sum(A * B, dim=-1)  # Shape: [num_positions]
    norm_A = torch.norm(A, dim=-1)      # Shape: [num_positions]
    norm_B = torch.norm(B, dim=-1)      # Shape: [num_positions]
    
    initial_cosine_similarity = dot_AB / (norm_A * norm_B)  # Shape: [num_positions]
    
    # Determine where initial cosine similarity >= desired
    mask = initial_cosine_similarity >= desired_cosine_similarity  # Shape: [num_positions]
    
    # Initialize scales with zeros where no adjustment is needed
    k = torch.zeros_like(dot_AB)
    
    # Indices where adjustment is needed
    adjust_indices = ~mask
    
    if not (-1 < desired_cosine_similarity < 1):
       raise ValueError("Desired cosine similarity must be between -1 and 1 (exclusive).")
    
    if not torch.any(adjust_indices):
       return k
    
    D = dot_AB[adjust_indices]
    N = norm_A[adjust_indices] ** 2
    M = norm_B[adjust_indices] ** 2  # No more need to squeeze
    
    one_minus_X2 = 1 - desired_cosine_similarity ** 2
    
    A_coef = M ** 2 * one_minus_X2
    B_coef = 2 * M * D * one_minus_X2
    C_coef = D ** 2 - desired_cosine_similarity ** 2 * M * N
    
    discriminant = B_coef ** 2 - 4 * A_coef * C_coef
    
    if torch.any(discriminant < 0):
       raise ValueError("No real solution exists to achieve the desired cosine similarity for some inputs.")
    
    sqrt_discriminant = torch.sqrt(discriminant)
    k1 = (-B_coef + sqrt_discriminant) / (2 * A_coef)
    k2 = (-B_coef - sqrt_discriminant) / (2 * A_coef)
    
    possible_ks = torch.stack([k1, k2], dim=1)  # Shape: [num_adjustments, 2]
    positive_ks = torch.where(possible_ks >= 0, possible_ks, torch.full_like(possible_ks, float('inf')))
    
    # Select the smallest positive k
    k_adjusted = torch.min(positive_ks, dim=1)[0]
    if torch.any(torch.isinf(k_adjusted)):
       raise ValueError("No positive scalar k found to achieve the desired cosine similarity for some inputs.")
    
    # Assign the computed k values back to the appropriate indices
    k[adjust_indices] = k_adjusted
    
    return k


def rotate_to_cosine_similarity(A, B, desired_cosine_similarity):
    """
    Rotates each vector in A towards or away from corresponding vector in B to achieve
    desired cosine similarity while preserving A's magnitude.
    Only rotates vectors that have current similarity < desired similarity.
    
    Args:
        A: Tensor of shape [num_positions, hidden_size]
        B: Tensor of shape [num_positions, hidden_size]
        desired_cosine_similarity: Target cosine similarity
        
    Returns:
        Modified A tensor with same magnitude but rotated to achieve target similarity
        where needed
    """
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    
    # Compute initial metrics
    dot_AB = torch.sum(A * B, dim=-1)  # [num_positions]
    norm_A = torch.norm(A, dim=-1)      # [num_positions]
    norm_B = torch.norm(B, dim=-1)      # [num_positions]
    
    # Current cosine similarity
    current_cos = dot_AB / (norm_A * norm_B)  # [num_positions]
    
    # Determine which positions need adjustment
    needs_rotation = current_cos < desired_cosine_similarity  # [num_positions]
    
    # If no positions need rotation, return original A
    if not torch.any(needs_rotation):
        return A.to(A.dtype)
    
    # Check if solution exists
    if torch.any(torch.abs(desired_cosine_similarity) > 1):
        raise ValueError("Desired cosine similarity must be between -1 and 1")
    
    # Get B_perpendicular to A
    # First get B parallel to A
    B_parallel = (dot_AB / norm_A.pow(2)).unsqueeze(-1) * A  # [num_positions, hidden_size]
    # Then subtract to get perpendicular component
    B_perp = B - B_parallel  # [num_positions, hidden_size]
    
    # Normalize B_perp
    B_perp_norm = torch.norm(B_perp, dim=-1, keepdim=True)  # [num_positions, 1]
    # Avoid division by zero
    mask = B_perp_norm > 1e-8
    B_perp = torch.where(mask, B_perp / B_perp_norm, torch.zeros_like(B_perp))
    
    # Compute rotation angles
    # cos(theta) = current_cos
    # After rotation by alpha:
    # new_cos = cos(theta-alpha) = cos(theta)cos(alpha) + sin(theta)sin(alpha)
    # where theta is current angle and alpha is rotation angle
    
    # We want: cos(theta-alpha) = desired_cosine_similarity
    # Therefore: alpha = theta - acos(desired_cosine_similarity)
    current_angle = torch.acos(torch.clamp(current_cos, -1+1e-7, 1-1e-7))
    target_angle = torch.acos(torch.clamp(desired_cosine_similarity, -1+1e-7, 1-1e-7))
    rotation_angle = current_angle - target_angle
    
    # Create rotation matrix implicitly using:
    # A_rotated = cos(alpha)A + sin(alpha)B_perp
    cos_rot = torch.cos(rotation_angle).unsqueeze(-1)  # [num_positions, 1]
    sin_rot = torch.sin(rotation_angle).unsqueeze(-1)  # [num_positions, 1]
    
    A_rotated = cos_rot * A + sin_rot * (B_perp * norm_A.unsqueeze(-1))
    
    # Handle edge cases where B_perp is zero (A and B parallel)
    # In these cases, no rotation is possible, so return original A
    A_rotated = torch.where(mask, A_rotated, A)
    
    # Only apply rotation where needed
    A_final = torch.where(needs_rotation.unsqueeze(-1), A_rotated, A)
    
    # Verify constraints for rotated vectors
    final_cos = torch.sum(A_final * B, dim=-1) / (torch.norm(A_final, dim=-1) * norm_B)
    magnitude_ratio = torch.norm(A_final, dim=-1) / norm_A
    
    # Add small epsilon for numerical stability in assertions
    eps = 1e-5
    assert torch.all(torch.abs(magnitude_ratio - 1) < eps), "Magnitude not preserved"
    assert torch.all((final_cos >= desired_cosine_similarity - eps) | ~needs_rotation), "Target similarity not achieved"
    
    return A_final.to(A.dtype)
    
def adjust_vectors(A, B, X, atol=1e-8):
    """
    Adjusts each vector in A to have a cosine similarity of X with the corresponding vector in B,
    while minimally altering each vector's magnitude.

    Parameters:
    A (torch.Tensor): Original vectors A, shape (num_positions, hidden_size).
    B (torch.Tensor): Vectors B, shape (num_positions, hidden_size).
    X (float): Desired cosine similarity between adjusted A and B.
    atol (float): Absolute tolerance for comparing cosine similarities.

    Returns:
    torch.Tensor: Adjusted vectors A', shape (num_positions, hidden_size).
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    A = A.to(torch.float64)
    B = B.to(torch.float64)
    
    # Compute norms of A and B
    norm_A = A.norm(dim=1, keepdim=True)  # Shape: (num_positions, 1)
    norm_B = B.norm(dim=1, keepdim=True)  # Shape: (num_positions, 1)

    if torch.any(norm_A == 0) or torch.any(norm_B == 0):
        raise ValueError("None of the input vectors should be zero vectors.")

    # Compute original cosine similarities
    c0 = torch.sum(A * B, dim=1, keepdim=True) / (norm_A * norm_B)  # Shape: (num_positions, 1)

    # Ensure desired cosine similarity X is within valid range
    if not -1 <= X <= 1:
        raise ValueError("Desired cosine similarity X must be between -1 and 1.")

    # Create a mask where adjustment is needed
    need_adjustment = ~torch.isclose(c0, torch.tensor(X, dtype=c0.dtype, device=c0.device), atol=atol)

    if not torch.any(need_adjustment):
        return A.clone()

    # Initialize A_prime as a copy of A
    A_prime = A.clone()

    # Indices where adjustment is needed
    indices_to_adjust = need_adjustment.view(-1).nonzero(as_tuple=False).squeeze()

    # Extract vectors that need adjustment
    A_adjust = A[indices_to_adjust]
    B_adjust = B[indices_to_adjust]
    norm_A_adjust = norm_A[indices_to_adjust]
    norm_B_adjust = norm_B[indices_to_adjust]

    # Decompose A into parallel and perpendicular components relative to B
    B_unit = B_adjust / norm_B_adjust  # Unit vectors of B
    A_parallel = torch.sum(A_adjust * B_unit, dim=1, keepdim=True) * B_unit
    A_perp = A_adjust - A_parallel

    # Norms of parallel and perpendicular components
    norm_A_parallel = A_parallel.norm(dim=1, keepdim=True)
    norm_A_perp = A_perp.norm(dim=1, keepdim=True)

    # Compute scaling factor lambda for each vector
    numerator = (X ** 2) * (norm_A_perp ** 2)
    denominator = norm_A_parallel ** 2 * (1 - X ** 2)

    # Check for division by zero
    if torch.any(denominator == 0):
        raise ValueError("Cannot adjust vectors to achieve the desired cosine similarity.")

    lambda_factor = torch.sqrt(numerator / denominator)

    # Adjust the parallel component
    A_parallel_adjusted = lambda_factor * A_parallel

    # Reconstruct the adjusted vectors
    A_prime_adjusted = A_parallel_adjusted + A_perp

    # Update A_prime with the adjusted vectors at the specified indices
    A_prime[indices_to_adjust] = A_prime_adjusted

    return A_prime


def scale_vector_by_sim(activation, sim_vec):
    """
    Computes a scaling factor for each activation based on its cosine similarity with sim_vec.

    Parameters:
    - activation: Tensor of shape [batch_size, hidden_size]
    - sim_vec: Tensor of shape [hidden_size] or [batch_size, hidden_size]

    Returns:
    - scale: Tensor of shape [batch_size]
    """
    # Ensure sim_vec has correct shape
    if sim_vec.dim() == 1:
        # Single vector, expand to match batch size of activation
        sim_vec = sim_vec.unsqueeze(0).expand_as(activation)  # Shape: [batch_size, hidden_size]
    elif sim_vec.shape[0] == activation.shape[0]:
        # sim_vec is already batch of vectors matching activation
        pass
    else:
        # sim_vec does not match activation, raise an error
        raise ValueError("Shape of sim_vec must either be [hidden_size] or match activation in batch size.")

    # Compute cosine similarity for each batch item
    dot_product = torch.sum(activation * sim_vec, dim=-1)  # Shape: [batch_size]
    norm_activation = torch.norm(activation, dim=-1)
    norm_sim_vec = torch.norm(sim_vec, dim=-1)

    cos_sim = dot_product / (norm_activation * norm_sim_vec + 1e-8)  # Shape: [batch_size]

    scale_factor = 1.0  

    scale = ((torch.clamp(0.02 - cos_sim, min=0)) ** 2)
    maxval = (1.02 * scale_factor) ** 2
    scale = scale / maxval

    return scale  # Shape: [batch_size]

                                
### Utilities to turn specific gradients on and off
def disable_grad():
    def hook(grad):
        return None
#        return torch.zeros_like(grad)
    return hook


def attach_zerograd_hooks(parameters):
    handles = []
    for param in parameters:
        handle = param.register_hook(disable_grad())
        handles.append(handle)
    return handles


def remove_zerograd_hooks(handles):
    for handle in handles:
        handle.remove()
###############################
