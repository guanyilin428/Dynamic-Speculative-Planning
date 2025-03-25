
import datetime
from typing import List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from OpenAGI import DistilBERTValueFunction, OnlineTrajectoryCollector, OnlineLearningExecutor, FiniteReplay

approx_logs = [
    (datetime.datetime(2025, 3, 25, 20, 40, 43, 189967), 'approximation', 1, 'Image Deblurring'),
    (datetime.datetime(2025, 3, 25, 20, 40, 45, 282907), 'approximation', 2, 'Image Super-Resolution'),
    (datetime.datetime(2025, 3, 25, 20, 40, 46, 347241), 'approximation', 3, 'Object Detection'),
    (datetime.datetime(2025, 3, 25, 20, 40, 47, 464739), 'approximation', 4, 'Machine Translation')
]
target_logs = [(datetime.datetime(2025, 3, 25, 20, 40, 50, 433159), 'target', 1, 'Image Deblurring'), 
               (datetime.datetime(2025, 3, 25, 20, 40, 50, 434754), 'target', 2, 'Image Super-Resolution'), 
               (datetime.datetime(2025, 3, 25, 20, 40, 54, 140240), 'target', 3, 'Object Detection'), 
               (datetime.datetime(2025, 3, 25, 20, 40, 57, 346195), 'target', 4, 'Machine Translation')]

model_path = "../weights/distilbert-base-uncased"
bert_model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DistilBERTValueFunction(bert_model)
replay_buffer = FiniteReplay(tokenizer=tokenizer, model=model, max_length=512, replay_size=32)
prefix = "Given low-resolutioned blurry grayscale image, how to return the object names in German step by step?"
initial_task = "Given low-resolutioned blurry grayscale image, how to return the object names in German step by step?"

def build_trajectory(prefix, initial_task):
        # when reach a breakingpoint, call build_trajectory
    combined_logs = sorted(approx_logs + target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "approximation" else 1))
    target_logs_sorted_by_step = sorted(target_logs, key=lambda x: x[2])
    i, j = 0, 0
    trajectory = []
    breakpoint()
    
    while j < len(target_logs_sorted_by_step):
        target_timestamp, _, target_step, target_desc = target_logs_sorted_by_step[j]
        approx_timestamp, _, approx_step, approx_desc = approx_logs[i]
        if approx_desc == target_desc and j != len(target_logs_sorted_by_step) - 1:
            i += 1
            j += 1
        else: # mismatch, breakingpoint
            # iterate thru all approx tasks in this breakingpoint
            # generate traj from bp start to cur_approx_task(not included)
            
            approx_index = 0
            while approx_index < len(approx_logs):
                cur_approx_timestamp, _, cur_approx_step, cur_approx_desc = approx_logs[approx_index]
                current_k = max(0, target_step - (cur_approx_step - 1))
                state = []
                
                for log in combined_logs:
                    if log == approx_logs[approx_index]:
                        break
                    state.append(f"{log[1]} Step {log[2]}: {log[3]}")
                reward = 1 if current_k > 0 else 0
                
                # construct prompt
                if state:
                    if prefix == initial_task:
                        prefix += "\nPrevious History:\n"
                    prompt = prefix + "\n".join(state) + "\n"
                else:
                    prompt = prefix
                trajectory.append({
                    "state": prompt,
                    "reward": reward,
                    "k": current_k
                })
                approx_index += 1
            break
        
    # _reset_trajectory()
    # update prefix
    bp_state = [f"{log[1]} Step {log[2]}: {log[3]}" for log in combined_logs]
    prefix += "\n".join(bp_state) + "\n"
    
    # add trajectory to replay buffer
    replay_buffer.add_trajectory(trajectory)
    
        
build_trajectory(prefix, initial_task)
        

