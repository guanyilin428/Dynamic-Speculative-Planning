"""Runner for the travel planner speculative planning system."""

import os
import argparse
import torch
import asyncio
from datasets import load_dataset

from transformers import AutoModel, AutoTokenizer
import tiktoken

from config import Config
from planner import TravelPlannerSpeculativePlanner
from predictor import DistilBERTValueFunction
from async_online_utils import OnlineLearningExecutor
from util import Logger
from agents.tool_agents_sp import DirectAgent, ReactAgent, CoTAgent, MultiAgent

config = Config()

def parse_args():
    parser = argparse.ArgumentParser(description='Travel Planner Speculative Planning')
    parser.add_argument('--set_type', type=str, default='validation', help='validation or test')
    parser.add_argument('--k', type=int, default=4, help='number of approximation steps to generate everytime')
    parser.add_argument('--approx_type', type=str, default='direct', help='cot, direct')
    parser.add_argument('--target_type', type=str, default='react', help='react, multi_agent')
    parser.add_argument('--model_type', type=str, default="gpt-4.1-mini", help='gpt-4.1-mini, deepseek-chat')
    parser.add_argument('--pred', action='store_true', help='enable speculative planning with predictor')
    parser.add_argument('--no-pred', dest='pred', action='store_false', help='disable speculative planning with predictor')
    parser.set_defaults(pred=True)
    
    # Online learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='online learning lr')
    parser.add_argument('--ep', type=int, default=3, help='online learning epoch per train')
    parser.add_argument('--bf', type=int, default=2500, help='online learning buffer size')
    parser.add_argument('--bs', type=int, default=16, help='online learning batch size')
    parser.add_argument('--gma', type=float, default=1, help='online learning gamma for lambda return calculation')
    parser.add_argument('--lmd', type=float, default=0.95, help='online learning lambda for lambda return calculation')
    parser.add_argument('--load', dest='load', action='store_true', help='load previous trajectory and model')
    parser.add_argument('--no-load', dest='load', action='store_false', help='do not load previous trajectory and model')
    parser.add_argument('--tau', type=float, default=0.5, help='expectile loss tau')
    parser.add_argument('--s_task', type=int, default=1, help='start task id')
    parser.add_argument('--freq', type=int, default=1, help='online learning training frequency')
    parser.add_argument('--offset', type=int, default=0, help='biased inference offset for k')
    
    parser.set_defaults(load=False)
    return parser.parse_args()

def setup_executor(args, device):
    """Set up the online learning executor."""
    if args.pred: # dyn k
        traj_dir = f"../trajectory/travel_planner/online_traj/{args.approx_type}_{args.target_type}/{args.model_type}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/tau_{args.tau}_offset_{args.offset}.ndjson"
    else: # fix k
        traj_dir = f"../trajectory/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/fix_k_{args.k}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/task_{{}}.json"
    
    ckpt_dir = f"../ckpt/travel_planner/online/{args.approx_type}_{args.target_type}/{args.model_type}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/tau_{args.tau}_offset_{args.offset}.pth"
    
    # Set up model
    model_path = "distilbert-base-uncased"
    bert_model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBERTValueFunction(bert_model).to(device)
    
    # Create executor
    return OnlineLearningExecutor(
        device=device,
        model_save_path=ckpt_path,
        model=model,
        tokenizer=tokenizer,
        buffer_size=args.bf,
        batch_size_steps=args.bs,
        lambda_=args.lmd,
        gamma=args.gma,
        lr=args.lr,
        epoch_per_train=args.ep,
        load=args.load,
        traj_file=traj_file,
        tau=args.tau
    ), traj_file

def setup_agents(args):
    """Set up approximation and target agents with proper types."""
    tools_list = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    app_model_type = args.model_type
    tar_model_type = args.model_type if args.model_type == "gpt-4.1-mini" else "deepseek-reasoner"
    
    # setup target agent
    if args.target_type == "react":
        target_agent = ReactAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=tar_model_type,
            planner_llm_name=tar_model_type,
        )
    elif args.target_type == "multi_agent":
        target_agent = MultiAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=tar_model_type,
            planner_llm_name=tar_model_type,
        )
    else: 
        target_agent = None
    config.TARGET_TYPE = args.target_type
    
    # setup approximation agent
    if args.approx_type == "direct":
        approximation_agent = DirectAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=app_model_type,
            planner_llm_name=app_model_type,
        )
    elif args.approx_type == "cot":
        approximation_agent = CoTAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=app_model_type,
            planner_llm_name=app_model_type,
        )
    else: 
        approximation_agent = None
    
    return approximation_agent, target_agent

def setup_task_loggers(args, task_id):
    """Set up loggers for a specific task."""
    pred_type = "dyn_k" if args.pred else "fix_k"
    log_dir = f"../data/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/{pred_type}"
    
    if args.pred:
        base_path = f'{log_dir}/tau_{args.tau}_offset_{args.offset}'
    else:
        base_path = f'{log_dir}/k_{args.k}'
    
    logger = Logger(f'{base_path}/simulation_datapoint{task_id}.log', on=True)
    target_logger = Logger(f'{base_path}/target_datapoint{task_id}.log', on=True)
    approximation_logger = Logger(f'{base_path}/approximation_datapoint{task_id}.log', on=True)
    
    return logger, target_logger, approximation_logger

def log_task_metrics(logger, step_num, config):
    """Log metrics for the completed task."""
    # Calculate basic metrics
    normal_plan_time = sum(config.TARGET_NORMAL_TIME[i] for i in range(1, step_num+1))
    normal_app_time = sum(config.APPROX_NROMAL_TIME[i] for i in range(1, step_num+1))
    normal_tar_generation = sum(config.TARGET_NORMAL_GENERATION[i] for i in range(1, step_num+1))
    normal_app_generation = sum(config.APPROX_NORMAL_GENERATION[i] for i in range(1, step_num+1))
    normal_plan_generation = normal_tar_generation + normal_app_generation
    normal_tar_prompt = sum(config.TARGET_NORMAL_PROMPT[i] for i in range(1, step_num+1))
    normal_app_prompt = sum(config.APPROX_NORMAL_PROMPT[i] for i in range(1, step_num+1))
    normal_plan_prompt = normal_app_prompt + normal_tar_prompt
    
    # Log token metrics
    logger.log('normal approx token prompt: ' + str(normal_app_prompt))
    logger.log('normal approx token generation: ' + str(normal_app_generation))
    logger.log('sp approx token prompt: ' + str(config.APPROX_SP_PROMPT))
    logger.log('sp approx token generation: ' + str(config.APPROX_SP_GENERATION))
    logger.log('normal target token prompt: ' + str(normal_tar_prompt))
    logger.log('normal target token generation: ' + str(normal_tar_generation))
    logger.log('sp target token prompt: ' + str(config.TOTAL_TOKEN_PROMPT - config.APPROX_SP_PROMPT))
    logger.log('sp target token generation: ' + str(config.TOTAL_TOKEN_GENERATION - config.APPROX_SP_GENERATION))
    logger.log('total sp token prompt: ' + str(config.TOTAL_TOKEN_PROMPT))
    logger.log('total sp token generation: ' + str(config.TOTAL_TOKEN_GENERATION))
    
    # Log timing metrics
    logger.log('normal target step time: ' + str(normal_plan_time))
    logger.log('accuracy of approximation agent: ' + str(config.TOTAL_CORRECT_APPROXIMATION_CALLS/config.TOTAL_APPROXIMATION_CALLS))
    
    # Calculate and log averages
    avg_sp_token = round(config.TOTAL_TOKEN_GENERATION/step_num, 2)
    avg_normal_token = round(normal_plan_generation/step_num, 2)
    logger.log('sp token generation/step: ' + str(avg_sp_token))
    logger.log('normal token generation/step: ' + str(avg_normal_token))
    logger.log('generation token cost increased: ' + str(round((avg_sp_token/avg_normal_token-1)*100, 2))+"%.")
    
    avg_sp_prompt = config.TOTAL_TOKEN_PROMPT / step_num
    avg_normal_prompt = round(normal_plan_prompt/step_num, 2)
    logger.log('sp prompt/step: ' + str(round(avg_sp_prompt, 2)))
    logger.log('normal prompt/step: ' + str(avg_normal_prompt))
    logger.log('prompt token cost increased: ' + str(round((avg_sp_prompt/avg_normal_prompt-1)*100, 2))+"%.")
    
    avg_sp_time = round(config.TOTAL_SP_TIME/step_num, 2)
    logger.log('sp time/step: ' + str(avg_sp_time))
    avg_normal_time = round(normal_plan_time/step_num, 2)
    avg_approx_time = round(normal_app_time/step_num, 2)
    logger.log('normal target time/step: ' + str(avg_normal_time))
    logger.log('normal approx time/step: ' + str(avg_approx_time))
    logger.log('time decreased by: ' + str(round((1-avg_sp_time/avg_normal_time)*100, 2))+"%.")

def run_one_task(args, task_id, executor, encoding, planner, traj_file, query_data_list):
    query = query_data_list[task_id-1]["query"]
    
    # Set query for agents (consistent with original)
    planner.app_agent.query = query
    planner.tar_agent.query = query
    
    logger, target_logger, approximation_logger = setup_task_loggers(args, task_id)
    
    # Set up train_logger if training is enabled (consistent with original)
    train_logger = None
    if config.ENABLE_TRAIN:
        pred_type = "dyn_k" if args.pred else "fix_k"
        log_dir = f"../data/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/{pred_type}"
        train_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/train_datapoint{task_id}.log', on=True)
    
    planner.set_loggers(logger, target_logger, approximation_logger, train_logger)
    planner.log_task_description(query)
    
    config.reset_task_metrics()
    
    # Initialize planner for new task
    planner.initialize_for_new_task(query)
    
    executor.set_initial_task_prompt(query)
    
    steps = asyncio.run(planner.run(args, query))
    
    step_num = len(steps)
    log_task_metrics(logger, step_num, config)
    
    if args.pred:
        logger.log(f'predictor acc: {round(config.PREDICT_CORRECT / config.PREDICT_TOTAL, 2)}')
    logger.log(f'step number: {step_num}')
    
    if not args.pred:
        logger.log(f'k = {args.k}')
        traj_file = traj_file.format(task_id)
    else: 
        logger.log('dynamic k')
    
    executor.save_trajectory(traj_file, config.ENABLE_TRAIN)
    
    return steps

def main():
    """Main function to run the travel planner speculative planning system."""
    os.environ['DEEPSEEK_API_KEY'] = ""
    os.environ['OPENAI_API_KEY'] = ""
    os.environ['GOOGLE_API_KEY'] = ""
    
    args = parse_args()
    
    config.initialize_from_args(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    executor, traj_file = setup_executor(args, device)
    
    app_agent, tar_agent = setup_agents(args)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    planner = TravelPlannerSpeculativePlanner(
        collector=executor.collector,
        encoding=encoding,
        executor=executor,
        app_agent=app_agent,
        tar_agent=tar_agent,
        config=config
    )
    
    if args.set_type == "validation":
        query_data_list = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    elif args.set_type == "test":
        query_data_list = load_dataset("osunlp/TravelPlanner", "test")["test"]
    
    
    task_ids = list(range(args.s_task, 180))
    
    warmup_task = 0
    for task_id in task_ids:
        if warmup_task >= config.WARMUP:
            config.ENABLE_PRED = args.pred
        run_one_task(args, task_id, executor, encoding, planner, traj_file, query_data_list)
        warmup_task += 1

if __name__ == "__main__":
    main() 