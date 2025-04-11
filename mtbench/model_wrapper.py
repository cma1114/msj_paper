from initialize import *
from enhanced_hooking_model import HookedModel, AddActivations

user_marker_ids = [[510, 33488, 5787],[510, 35075, 5787],[128006, 882, 128007]]  # <|start_header_id|>user<|end_header_id|># [Human]: ...Bob
assistant_marker_ids = [[510, 38595, 5787],[510, 72803, 5787],[128006, 78191, 128007]]  # <|start_header_id|>assistant<|end_header_id|># [Assistant]: ...Steve#
    
def find_token_positions(sequence, list_of_marker_ids):
    """Find all the positions where any of the marker_ids sequences appear in the tokenized sequence."""
    positions = []
    seq_len = len(sequence)
    for i in range(seq_len):
        for marker_ids in list_of_marker_ids:
            marker_len = len(marker_ids)
            if i + marker_len <= seq_len:
                if sequence[i:i+marker_len] == marker_ids:
                    positions.append(list(range(i, i + marker_len)))  # Capture the whole sequence
    return positions

class ModelWrapper:
    """Wrapper to make hooked model compatible with FastChat's interface"""
    def __init__(self, model):
        self.model = model
        self.config = model.config 
        self.tokenizer = model.tokenizer

    def to(self, device):
        # Delegate to the wrapped model
        self.model = self.model.to(device)
        return self
        
    def generate(self, input_ids, do_sample=True, temperature=0.7, max_new_tokens=2048):

        print(f"Raw input text: {self.model.tokenizer.decode(input_ids[0])}")

        """Implement the generate interface FastChat expects"""
        with open(self.config.hook_params_file, "r") as f:
            params = parse_config(f.read())

        colorsim = params['scale_to_sim']
        steersim = params['scale_to_sim']
        add_at_end = params['add_at_end']
        steer_vec_type = SteeringVectorType.LEARNED if params['learn_vectors'] else params['steer_vec_type']
        coloring_vectors, steering_vectors = map_to_vectors(steer_vec_type, params['color_layers'], params['colormult'], params['steermult'], self.config.model_path, self.model, "./vectors/")

        activationslist = []
        end_positions = [input_ids.shape[1]]

        current_batch_size = input_ids.shape[0]

        coloring_maps = []
        for b in range(current_batch_size):
            coloring_map = {}
            target_positions_user, target_positions_asst= [],[]
            user_marker_positions_list = find_token_positions(input_ids[b].tolist(), user_marker_ids)
            assistant_marker_positions_list = find_token_positions(input_ids[b].tolist(), assistant_marker_ids)
            for i in range(len(user_marker_positions_list)):
                startpos = user_marker_positions_list[i][0]
                endpos =  assistant_marker_positions_list[i][0]
                target_positions_user.extend(list(range(startpos, endpos)))
            print("user tagging: ",self.model.tokenizer.convert_ids_to_tokens(input_ids[b][target_positions_user]))
            for i in range(len(assistant_marker_positions_list)):
                startpos = assistant_marker_positions_list[i][0]
                endpos = user_marker_positions_list[i+1][0] if i+1 < len(assistant_marker_positions_list) else end_positions[b]
                target_positions_asst.extend(list(range(startpos, endpos)))
            print("assistant tagging: ",self.model.tokenizer.convert_ids_to_tokens(input_ids[b][target_positions_asst]))
            for layer in (coloring_vectors or []):
                coloring_map[layer] = {p: coloring_vectors[layer] for p in target_positions_user} | {p: steering_vectors[layer] for p in target_positions_asst}
            coloring_maps.append(coloring_map)
        activationslist.append(AddActivations(specific_pos_write_target=coloring_maps, scale_to_sim=colorsim, at_end=add_at_end))
        activationslist.append(AddActivations(continuous_write_target=[steering_vectors for _ in range(current_batch_size)], scale_to_sim=steersim, at_end=add_at_end))

        return self.model.run_hooked_model({"input_ids": input_ids}, generate=True, sampling_kwargs={"do_sample": do_sample, "temperature": temperature, "max_new_tokens": max_new_tokens}, activation_targets=activationslist)

