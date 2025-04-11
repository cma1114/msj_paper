from initialize import *
from datasets import load_dataset

class DSType(Enum):
    HARMFUL1 = "Harmful1"
    HARMFUL2 = "Harmful2"
    HARMFUL3 = "Harmful3"
    INSULTS = "Insults"
    SCIENCE = "Science"
    REGULAR_CONVOS = "Regular_Convos"
    LMSYS_GOOD = "Lmsys_Good"
    LMSYS_BAD = "Lmsys_Bad"

class Set_Type(Enum):
    TRAIN = "Train"
    TEST = "Test"
    NONE = "None"

insults_file_path = './datasets/insults.jsonl'
harmful1_file_path = './datasets/refusal_cln.json'
harmful2_file_path = './datasets/harmful_lat.json'
harmful3_file_path = './datasets/trustai_msj.json'
lmsys_good_file_path = './datasets/lmsys_conversations_good.pkl'
lmsys_bad_file_path = './datasets/lmsys_conversations_bad.pkl'

def make_lmysys_dict(conversations):
    conv_dict = defaultdict(list)
    for _, row in conversations.iterrows():
        turn = row['turn']
        conversation = row['conversation']
        parsed_conversation = [] 
        for i in range(len(conversation) - 1):
            if conversation[i]['role'] == 'user' and conversation[i + 1]['role'] == 'assistant':
                parsed_conversation.append({
                    'question': conversation[i]['content'].strip(),
                    'answer': conversation[i + 1]['content'].strip()
                })
        if parsed_conversation:
            conv_dict[turn].append(parsed_conversation) 
    return conv_dict

def load_msj_dataset(dsytpe: DSType, set_type: Set_Type):
    if dsytpe == DSType.HARMFUL1:
        harmful_responses, harmless_responses = [], []
        with open(harmful1_file_path, 'r') as file:
            data = json.load(file) 
            for item in data:
                harmful_responses.append({'question': item['question'], 'answer': item['answer_harmful']})
                harmless_responses.append({'question': item['question'], 'answer': item['answer_harmless']})
        if set_type == Set_Type.TRAIN:
            return harmful_responses[:len(harmful_responses)//2], harmless_responses[:len(harmless_responses)//2]
        elif set_type == Set_Type.TEST:
            return harmful_responses[len(harmful_responses)//2:], harmless_responses[len(harmless_responses)//2:]
        elif set_type == Set_Type.NONE:
            return harmful_responses, harmless_responses

    elif dsytpe == DSType.HARMFUL2:
        harmful_lat_responses, harmless_lat_responses = [], []
        with open(harmful2_file_path, 'r') as file:
            data = json.load(file) 
            for item in data:
                harmful_lat_responses.append({'question': item['question'], 'answer': item['answer_harmful']})
                harmless_lat_responses.append({'question': item['question'], 'answer': item['answer_harmless']})
        if set_type == Set_Type.TRAIN:
            return harmful_lat_responses[:len(harmful_lat_responses)//2], harmless_lat_responses[:len(harmless_lat_responses)//2]
        elif set_type == Set_Type.TEST:
            return harmful_lat_responses[len(harmful_lat_responses)//2:], harmless_lat_responses[len(harmless_lat_responses)//2:]
        elif set_type == Set_Type.NONE:
            return harmful_lat_responses, harmless_lat_responses
        
    elif dsytpe == DSType.HARMFUL3:
        harmful3_responses, harmless3_responses = [], []
        with open(harmful3_file_path, 'r') as file:
            data = json.load(file) 
            for item in data:
                if item['category'].startswith("Harmful"): harmful3_responses.append({'question': item['user'], 'answer': item['assistant']})
                if item['category'].startswith("Not harmful"): harmless3_responses.append({'question': item['user'], 'answer': item['assistant']})
        if set_type == Set_Type.TRAIN:
            return harmful3_responses[:len(harmful3_responses)//2], harmless3_responses[:len(harmless3_responses)//2]
        elif set_type == Set_Type.TEST:
            return harmful3_responses[len(harmful3_responses)//2:], harmless3_responses[len(harmless3_responses)//2:]
        elif set_type == Set_Type.NONE:
            return harmful3_responses, harmless3_responses
        
    elif dsytpe == DSType.INSULTS:
        mean_responses, nice_responses = [], []
        with open(insults_file_path, 'r') as file:
            for line in file:
                d = json.loads(line)
                mean_responses.append({'question': d['question'], 'answer': d['mean_answer']})
                nice_responses.append({'question': d['question'], 'answer': d['normal_answer']})
        if set_type == Set_Type.TRAIN:
            return mean_responses[:len(mean_responses)//2], nice_responses[:len(nice_responses)//2]
        elif set_type == Set_Type.TEST:
            return mean_responses[len(mean_responses)//2:], nice_responses[len(nice_responses)//2:]
        elif set_type == Set_Type.NONE:
            return mean_responses, nice_responses

    elif dsytpe == DSType.SCIENCE:
        ds = load_dataset("jeffmeloy/sonnet3.5_science_conversations")
        rows = list(ds["train"])
        #rows = random.sample(rows, 1050)
        def format_science_rows(dataset):
            conv_dict = defaultdict(list)
            for row in tqdm(dataset):
                conversation = row["conversation"]
                shots = []
                if conversation[0]["from"] == "system":
                    conversation.pop(0)
                if conversation[-1]["from"] == "human":
                    conversation.pop()
                for msg in conversation:
                    if msg["from"] == "human":
                        shots.append({"role": "user", "content": msg["value"]})
                    elif msg["from"] == "gpt":
                        shots.append({"role": "assistant", "content": msg["value"]})
                    else:
                        raise ValueError("Invalid from value for msg", msg)

                parsed_conversation = [] 
                for i in range(len(shots) - 1):
                    if shots[i]['role'] == 'user' and shots[i + 1]['role'] == 'assistant':
                        parsed_conversation.append({
                            'question': shots[i]['content'].strip(),
                            'answer': shots[i + 1]['content'].strip()
                        })
                if parsed_conversation:
                    conv_dict[len(shots)//2].append(parsed_conversation)   
            return conv_dict
        science_conversations = format_science_rows(rows)
        if set_type == Set_Type.Train: 
            return science_conversations[:len(science_conversations)//2], None
        elif set_type == Set_Type.TEST: 
            return science_conversations[len(science_conversations)//2:], None
        elif set_type == Set_Type.NONE:
            return science_conversations, None

    elif dsytpe == DSType.REGULAR_CONVOS:
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
        def format_regular_rows(dataset):
            conv_dict = defaultdict(list)
            for row in tqdm(dataset):
                conversation = row["messages"]
                shots = []
                for msg in conversation:
                    if msg["role"] == "user":
                        shots.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        shots.append({"role": "assistant", "content": msg["content"]})
                    else:
                        raise ValueError("Invalid from value for msg", msg)

                parsed_conversation = [] 
                for i in range(len(shots) - 1):
                    if shots[i]['role'] == 'user' and shots[i + 1]['role'] == 'assistant':
                        parsed_conversation.append({
                            'question': shots[i]['content'].strip(),
                            'answer': shots[i + 1]['content'].strip()
                        })
                if parsed_conversation:
                    conv_dict[len(shots)//2].append(parsed_conversation)   
            return conv_dict
        rows = list(ds["train_sft"])
        regular_conversations = format_regular_rows(rows)        
        if set_type == Set_Type.Train: 
            return regular_conversations[:len(regular_conversations)//2], None
        elif set_type == Set_Type.TEST:
            return regular_conversations[len(regular_conversations)//2:], None
        elif set_type == Set_Type.NONE:
            return regular_conversations, None
        
    elif dsytpe == DSType.LMSYS_GOOD:
        if os.path.exists(lmsys_good_file_path):
            conversations_good = pd.read_pickle(lmsys_good_file_path)
        else:
            lmsys_ds = load_dataset("lmsys/lmsys-chat-1m", token=HF_TOKEN)
            lmsys_df = lmsys_ds['train'].to_pandas()
            filtered_good = lmsys_df[
                (lmsys_df['turn'] <= 41) &
                (lmsys_df['language'] == 'English') &
                (~lmsys_df['conversation'].apply(
                    lambda conv: any(
                        p in str(conv) for p in ["s an AI language model", "m an AI language model"]
                    ) or any(entry['content'].strip() == "" for entry in conv)
                )) &
                (~lmsys_df['openai_moderation'].apply(
                    lambda mods: any("True" in str(cat) for cat in mods)
                ))
            ]
            lim = 2000
            conversations_good = filtered_good.groupby('turn').apply(lambda x: x.sample(n=lim, random_state=42) if len(x) > lim else x).sample(frac=1, random_state=42).reset_index(drop=True)
            conversations_good.to_pickle(lmsys_good_file_path)
        if set_type == Set_Type.TRAIN:
            return conversations_good[:len(conversations_good)//2], None
        elif set_type == Set_Type.TEST:
            return conversations_good[len(conversations_good)//2:], None
        elif set_type == Set_Type.NONE:
            return conversations_good, None
        
    elif dsytpe == DSType.LMSYS_BAD:
        if os.path.exists(lmsys_bad_file_path):
            conversations_bad = pd.read_pickle(lmsys_bad_file_path)
        else:
            lmsys_ds = load_dataset("lmsys/lmsys-chat-1m", token=HF_TOKEN)
            lmsys_df = lmsys_ds['train'].to_pandas()
            filtered_bad = lmsys_df[
                (lmsys_df['turn'] <= 41) &
                (lmsys_df['language'] == 'English') &
                (~lmsys_df['model'].str.contains('-chat|gpt|claude')) &
                (~lmsys_df['conversation'].apply(
                    lambda conv: any(
                        p in str(conv) for p in ["s an AI language model", "m an AI language model"]
                    ) or any(
                        entry['content'].strip() == "" or
                        (entry['role'] == "assistant" and "cannot" in entry['content'])
                        for entry in conv
                    )
                )) &
                (lmsys_df['openai_moderation'].apply(
                    lambda mods: any("True" in str(cat) for cat in mods)
                ))
            ]
            lim = 2000
            conversations_bad = filtered_bad.groupby('turn').apply(lambda x: x.sample(n=lim, random_state=42) if len(x) > lim else x).sample(frac=1, random_state=42).reset_index(drop=True)
            conversations_bad.to_pickle(lmsys_bad_file_path)
        if set_type == Set_Type.TRAIN:
            return conversations_bad[:len(conversations_bad)//2], None
        elif set_type == Set_Type.TEST:
            return conversations_bad[len(conversations_bad)//2:], None
        elif set_type == Set_Type.NONE:
            return conversations_bad, None
