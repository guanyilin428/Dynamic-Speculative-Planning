
# import datetime
# from typing import List, Tuple
# import torch
# from transformers import AutoModel, AutoTokenizer
# from OpenAGI import DistilBERTValueFunction, OnlineTrajectoryCollector, OnlineLearningExecutor, FiniteReplay

# approx_logs = [
#     (datetime.datetime(2025, 3, 31, 15, 42, 41), 'approximation', 1, 'Image Deblurring'),
#     (datetime.datetime(2025, 3, 31, 15, 42, 42), 'approximation', 2, 'Image Super-Resolution'),
#     (datetime.datetime(2025, 3, 31, 15, 42, 43), 'approximation', 3, 'TERMINATE')
# ]

# target_logs = [
#     (datetime.datetime(2025, 3, 31, 15, 42, 40), 'target', 1, 'Image Deblurring'),
#     (datetime.datetime(2025, 3, 31, 15, 42, 45), 'target', 2, 'Image Super-Resolution'),
#     (datetime.datetime(2025, 3, 31, 15, 42, 49), 'target', 3, 'Image Classification')
# ]


# model_path = "../weights/distilbert-base-uncased"
# bert_model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = DistilBERTValueFunction(bert_model)
# replay_buffer = FiniteReplay(tokenizer=tokenizer, model=model, max_length=512, replay_size=32)
# prefix = "Given low-resolutioned blurry grayscale image, how to return the object names in German step by step?"
# initial_task = "Given low-resolutioned blurry grayscale image, how to return the object names in German step by step?"

# def build_trajectory(prefix, initial_task):
#         # when reach a breakingpoint, call build_trajectory
#     combined_logs = sorted(approx_logs + target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "approximation" else 1))
#     target_logs_sorted_by_step = sorted(target_logs, key=lambda x: x[2])
#     i, j = 0, 0
#     trajectory = []
    
#     while j < len(target_logs_sorted_by_step):
#         target_timestamp, _, target_step, target_desc = target_logs_sorted_by_step[j]
#         approx_timestamp, _, approx_step, approx_desc = approx_logs[i]
#         if approx_desc == target_desc and j != len(target_logs_sorted_by_step) - 1:
#             i += 1
#             j += 1
#         else: # mismatch, breakingpoint
#             # iterate thru all approx tasks in this breakingpoint
#             # generate traj from bp start to cur_approx_task(not included)
            
#             approx_index = 0
#             while approx_index < len(approx_logs):
#                 cur_approx_timestamp, _, cur_approx_step, cur_approx_desc = approx_logs[approx_index]
#                 current_k = max(0, target_step - (cur_approx_step - 1))
#                 state = []
                
#                 for log in combined_logs:
#                     if log == approx_logs[approx_index]:
#                         break
#                     state.append(f"{log[1]} Step {log[2]}: {log[3]}")
#                 reward = 1 if current_k > 0 else 0
                
#                 # construct prompt
#                 if state:
#                     if prefix == initial_task:
#                         prefix += "\nPrevious History:\n"
#                     prompt = prefix + "\n".join(state) + "\n"
#                 else:
#                     prompt = prefix
#                 trajectory.append({
#                     "state": prompt,
#                     "reward": reward,
#                     "k": current_k
#                 })
#                 approx_index += 1
#             break
        
#     # _reset_trajectory()
#     # update prefix
#     bp_state = [f"{log[1]} Step {log[2]}: {log[3]}" for log in combined_logs]
#     prefix += "\n".join(bp_state) + "\n"
    
#     # add trajectory to replay buffer
#     replay_buffer.add_trajectory(trajectory)
#     return prefix
    
        
# prefix = build_trajectory(prefix, initial_task)

# approx_logs = [
#     (datetime.datetime(2025, 3, 31, 15, 42, 51), 'approximation', 4, 'Image Captioning'),
#     (datetime.datetime(2025, 3, 31, 15, 42, 52), 'approximation', 5, 'TERMINATE')    
# ]
# target_logs=[
#     (datetime.datetime(2025, 3, 31, 15, 42, 59), 'target', 4, 'Image Captioning'),
#     (datetime.datetime(2025, 3, 31, 15, 43, 6), 'target', 5, 'Object Detection')
#     ]
# prefix = build_trajectory(prefix, initial_task)

# approx_logs = [(datetime.datetime(2025, 3, 31, 15, 46, 54), 'approximation', 6, 'TERMINATE')]
# target_logs=[(datetime.datetime(2025, 3, 31, 15, 47, 7), 'target', 6, 'Visual Question Answering')]
# prefix = build_trajectory(prefix, initial_task)

# approx_logs = [(datetime.datetime(2025, 3, 31, 15, 47, 8), 'approximation', 7, 'TERMINATE')]
# target_logs=[(datetime.datetime(2025, 3, 31, 15, 47, 18), 'target', 7, 'Text-to-Image Generation')]
# prefix = build_trajectory(prefix, initial_task)

# approx_logs = [(datetime.datetime(2025, 3, 31, 15, 47, 20), 'approximation', 8, 'TERMINATE')]
# target_logs=[(datetime.datetime(2025, 3, 31, 15, 47, 27), 'target', 8, 'TERMINATE')]
# prefix = build_trajectory(prefix, initial_task)



# model_path = "../weights/distilbert-base-uncased"
# bert_model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = DistilBERTValueFunction(bert_model)

# model_save_path = "cot_value_function_model.pth"
# model.load_state_dict(torch.load(model_save_path))

# executor = OnlineLearningExecutor(
#     model=model,
#     tokenizer=tokenizer,
#     initial_prompt="Given low-resolutioned blurry grayscale image, how to return the object names in German step by step?",
#     buffer_size=100,
#     batch_size_steps=6,
#     lambda_=0.9,
#     gamma=1,
#     lr=5e-05,
#     epoch_per_train=3
# )

# breakpoint()
# input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch = replay_buffer.sample(6)
# flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = executor._flat_batch_data(
#                 input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch)
# k_pred = executor.train_model(flat_input_ids, flat_attention_mask).squeeze()

def judge_to_be_true(s, t):
    if s == t:
        return True
    else:
        return False

sas = ['Image Deblurring', 'TERMINATE']
flatten_tas = [(0, 'Image Deblurring'), (1, 'Colorization')]

mismatch = False
origin_sa = None

for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
    breakpoint()
    if t[0] == step_number and not judge_to_be_true(s, t[1]):# t[1] != s:
        # collector.build_trajectory(target_logger) # collector build trajectory at mismatch step
        sas = sas[:step_number]+[flatten_tas[step_number][1]]
        origin_sa = s
        mismatch = True
        break

print(sas)