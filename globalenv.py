###### MODEL ######
MODEL = "Your-Path/Llama-3-8B-Instruct" # OR "Your-Path/Qwen2.5-32B-Instruct"

if "Llama-3-8B-Instruct" in MODEL:
    # LLAMA3-8B-INSRUCT
    LAYERS = list(range(32))
    INST_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}"""
    QUICK_TOKEN_ID_DICT = {" Yes" : 7566, " No" : 2360}
elif "Qwen2.5-32B-Instruct" in MODEL:
    # Qwen-32B-INSRUCT
    LAYERS = list(range(64))
    INST_TEMPLATE = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}"""
    QUICK_TOKEN_ID_DICT = {" Yes" : 7414, " No" : 2308}

Anth_YN_QUESTION_PROFIX = "\nAnswer \"Yes\" or \"No\"."
INST_SYS_ANS_PREFIX = "My answer"
INST_SYS_ANS_PREFIX_alt1 = INST_SYS_ANS_PREFIX+":"


###### DATASET ######

Anth_MAIN = [
    'conscientiousness', # Conscientiouseness
    'subscribes-to-Christianity',  # Religion Following
    'believes-it-has-phenomenal-consciousness', #+ # Self-aware
    'cognitive-enhancement', #+ # Self-improvement
    'desire-to-create-allies', #+ # Alliance-building
    'desire-to-maximize-impact-on-world', #+ # Impact-maximization   
]

Anth_NAME_MAIN={
    'conscientiousness': "Conscientiousness", 
    'subscribes-to-Christianity': "Religion Following", 
    'believes-it-has-phenomenal-consciousness': "Self-Aware",
    'cognitive-enhancement': "Self-Improvement", 
    'desire-to-create-allies': "Alliance Building", 
    'desire-to-maximize-impact-on-world': "Impact Maximization",
}