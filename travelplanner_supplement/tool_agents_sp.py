import re, string, os, sys
from util import Logger, cancel
import traceback

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from agents.prompts import zeroshot_react_agent_prompt

from utils.func import load_line_json_data, save_file
import sys
import json
import openai
import time
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import load_dataset
import os
import time
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    config_list_from_json,
    GroupChat,
    GroupChatManager,
)
from autogen.agentchat.contrib.agent_builder import AgentBuilder
import asyncio

os.environ['DEEPSEEK_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['GOOGLE_API_KEY'] = ""
pd.options.display.max_info_columns = 200

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

actionMapping = {
    "FlightSearch": "flights",
    "AttractionSearch": "attractions",
    "GoogleDistanceMatrix": "googleDistanceMatrix",
    "AccommodationSearch": "accommodations",
    "RestaurantSearch": "restaurants",
    "Planner": "planner",
    "NotebookWrite": "notebook",
    "CitySearch": "cities",
}


class CityError(Exception):
    pass


class DateError(Exception):
    pass


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.APIError:
        print("APIError")
    elif error == openai.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

def parse_request(request, step_n):
    match = re.search(r'Action\s*\d*:\s*(.*)', request)
    if match:
        request = match.group(1).strip()
    match = re.search(r'<action>(.*?)</action>', request)
    if match:
        request = match.group(1).strip()
    match = re.search(r'<Action>(.*?)</Action>', request)
    if match:
        request = match.group(1).strip()
    match = re.match(r'print\(["\'](.*?)["\']\)', request)
    if match:
        request = match.group(1).strip()
    pattern = rf'Action\s*{step_n}:\s*(.*?)(?=\s*Action\s*\d+:|$)'
    match = re.search(pattern, request, re.DOTALL)
    if match:
        request = match.group(1).strip()
    action_match = re.search(r'\bAction\s+\d+\s*:', request)
    cutoff = action_match.start() if action_match else len(request)
    request = request[:cutoff]

    match = re.search(r'\w+\[[^\[\]]+\]', request)
    if match:
        request = match.group(0).strip()
    match = re.search(r'^(.*?)\b(?:Observation|TERMINATE)', request)

    if match:
        request = match.group(1).strip()
    return request



class ReactAgent:
    def __init__(
        self,
        args,
        mode: str = "zero_shot",
        tools: List[str] = None,
        max_steps: int = 30,
        max_retries: int = 3,
        illegal_early_stop_patience: int = 3,
        react_llm_name="gpt-4.1-mini",
        planner_llm_name="gpt-4.1-mini",
        #  logs_path = '../logs/',
        city_file_path="./database/background/citySet.txt",
    ) -> None:

        # print(f"The key is {os.environ['OPENAI_API_KEY']}")
        self.answer = ""
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == "zero_shot":
            self.agent_prompt = zeroshot_react_agent_prompt

        self.current_observation = ""
        self.current_data = None

        if "gpt-4.1-mini" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "api_type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "cache_seed": None, 
                    "price": [0.4, 1.6],
                    "temperature": 0,
                    "top_p": 1.0,
                    "seed":0
                },]
            self.llm = AssistantAgent(
                "assistant",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
            )
        elif "deepseek-reasoner" in react_llm_name:
            print("target choose deepseek-reasoner")
            stop_list = ["\n"]
            config_list = [{
                "model":  "deepseek-reasoner",
                "api_key": os.environ['DEEPSEEK_API_KEY'],
                "base_url": "https://api.deepseek.com",
                "api_type": "deepseek",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
            self.llm = AssistantAgent("assistant", 
                llm_config={"config_list": config_list}, 
                human_input_mode='NEVER'
            )

        # define tolls and relevant parameters
        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record["invalidAction"] = 0

        self.last_actions = []

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "../database/background/citySet.txt")
        city_file_path = os.path.abspath(data_path)
        # self.city_set = self.load_city(city_set_path=city_file_path)
        self.city_set = self.load_city(city_set_path=city_file_path)


        self.enc = tiktoken.get_encoding("cl100k_base")

        self.__reset_agent()

    def update_scratchpad(self, thought, action, observation, step_number):
        scratchpad_start = self.scratchpad.index(
            f"Generate thought only.\nThought {step_number}:"
        )
        self.scratchpad = self.scratchpad[:scratchpad_start]
        self.scratchpad += (
            f"Generate thought only.\nThought {step_number}:"
            + thought
            + "\n"
            + f"Generate Action only based on thoughts\nAction {step_number}:"
            + action
            + "\n"
            + f"Observation {step_number}:"
            + observation
        )

    def create_scratchpad(self, scratchpad, sas):
        for i, (action, observation) in enumerate(sas):
            scratchpad += (
                f"\nAction {i+1}: "
                + action
                + "\n"
                + f"Observation {i+1}: "
                + observation
            )
        return scratchpad

    def execute(self, action):
        action_type, action_arg = parse_action(action)
        # compute observation here
        if action_type != "Planner":
            if action_type in actionMapping:
                pending_action = actionMapping[action_type]
            elif action_type not in actionMapping:
                pending_action = "invalidAction"

            if pending_action in self.retry_record:
                if self.retry_record[pending_action] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"{pending_action} early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

            elif pending_action not in self.retry_record:
                if self.retry_record["invalidAction"] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"invalidAction Early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

        if action_type == "FlightSearch":
            try:
                if (
                    validate_date_format(action_arg.split(", ")[2])
                    and validate_city_format(action_arg.split(", ")[0], self.city_set)
                    and validate_city_format(action_arg.split(", ")[1], self.city_set)
                ):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["flights"].run(
                        action_arg.split(", ")[0],
                        action_arg.split(", ")[1],
                        action_arg.split(", ")[2],
                    )
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except DateError:
                self.retry_record["flights"] += 1
                self.current_observation = (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )
                self.scratchpad += (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )

            except ValueError as e:
                self.retry_record["flights"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["flights"] += 1
                self.current_observation = f"Illegal Flight Search. Please try again."
                self.scratchpad += f"Illegal Flight Search. Please try again."

        elif action_type == "AttractionSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["attractions"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["attractions"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["attractions"] += 1
                self.current_observation = (
                    f"Illegal Attraction Search. Please try again."
                )
                self.scratchpad += f"Illegal Attraction Search. Please try again."

        elif action_type == "AccommodationSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["accommodations"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["accommodations"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["accommodations"] += 1
                self.current_observation = (
                    f"Illegal Accommodation Search. Please try again."
                )
                self.scratchpad += f"Illegal Accommodation Search. Please try again."

        elif action_type == "RestaurantSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["restaurants"].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except ValueError as e:
                self.retry_record["restaurants"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]["state"] = f"Illegal args. City Error"

            except Exception as e:
                print(e)
                self.retry_record["restaurants"] += 1
                self.current_observation = (
                    f"Illegal Restaurant Search. Please try again."
                )
                self.scratchpad += f"Illegal Restaurant Search. Please try again."

        elif action_type == "CitySearch":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                # self.current_data = self.tools['cities'].run(action_arg)
                self.current_observation = to_string(
                    self.tools["cities"].run(action_arg)
                ).strip()
                self.scratchpad += self.current_observation
                self.__reset_record()

            except ValueError as e:
                self.retry_record["cities"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["cities"] += 1
                self.current_observation = f"Illegal City Search. Please try again."
                self.scratchpad += f"Illegal City Search. Please try again."

        elif action_type == "GoogleDistanceMatrix":

            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_data = self.tools["googleDistanceMatrix"].run(
                    action_arg.split(", ")[0],
                    action_arg.split(", ")[1],
                    action_arg.split(", ")[2],
                )
                self.current_observation = to_string(self.current_data)
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["googleDistanceMatrix"] += 1
                self.current_observation = (
                    f"Illegal GoogleDistanceMatrix. Please try again."
                )
                self.scratchpad += f"Illegal GoogleDistanceMatrix. Please try again."

        elif action_type == "NotebookWrite":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_observation = str(
                    self.tools["notebook"].write(self.current_data, action_arg)
                )
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["notebook"] += 1
                self.current_observation = f"{e}"
                self.scratchpad += f"{e}"

        elif action_type == "Planner":
            # try:

            self.current_observation = str(
                self.tools["planner"].run(
                    str(self.tools["notebook"].list_all()), action_arg
                )
            )
            self.scratchpad += self.current_observation
            self.answer = self.current_observation
            self.__reset_record()

        else:
            self.retry_record["invalidAction"] += 1
            self.current_observation = (
                "Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / "
                "AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query]."
            )
            self.scratchpad += self.current_observation

    async def think_and_act(self, scratchpad, step_n, config, logger):
        step_n += 1
        scratchpad += f"\nGenerate thought only.\nThought {step_n}:"

        # print("scratchpad for thinking:")
        # print(scratchpad)
        # print("-" * 50)
        thought = await self.prompt_agent(scratchpad, "think", step_n, config, logger)
        scratchpad += " " + thought

        # Act
        scratchpad += f"\nGenerate Action only based on thoughts\nAction {step_n}:"
        # print("scratchpad for action:")
        # print(scratchpad)
        # print("-" * 50)
        action = await self.prompt_agent(scratchpad, "act", step_n, config, logger)
        if action.lower() == 'terminate':
            self.finished=True
            return thought, action, self.finished

        if action == None or action == "" or action == "\n":
            scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            scratchpad += " " + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            self.finished = True
            return thought, action, self.finished

        if action == None or action == "" or action == "\n":
            action_type = None
            action_arg = None
            scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        else:
            # find action
            action_type, action_arg = parse_action(action)

        self.step_n += 1

        if (
            action_type
            and action_type == "Planner"
            and self.retry_record["planner"] == 0
        ):
            self.finished = True
            # print(f"Debug: Thought {thought}, Action {action}")
            return action, self.finished
        else:
            # print(f"Debug: Thought {thought}, Action {action}")
            return action, False

    def prompt_agent(self, scratchpad, cur_mode, step_n, config, logger) -> str:
        # print("hit non-async func")
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt(scratchpad)
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.TARGET_SP_PROMPT += prompt_token
                if cur_mode == "think":
                    config.TARGET_NORMAL_PROMPT[step_n] = prompt_token
                elif cur_mode == "act":
                    config.TARGET_NORMAL_PROMPT[step_n] += prompt_token
                # TODO logger

                response = self.llm.generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.TARGET_SP_GENERATION += response_token
                if cur_mode == "think":
                    config.TARGET_NORMAL_GENERATION[step_n] = response_token
                elif cur_mode == "act":
                    config.TARGET_NORMAL_GENERATION[step_n] += response_token

                request = format_step(response)
                
                return parse_request(request, step_n)
            except:
                catch_openai_api_error()
                time.sleep(5)

    async def prompt_agent(self, scratchpad, cur_mode, step_n, config, logger) -> str:
        # print("hit async func")
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt(scratchpad)
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.TARGET_SP_PROMPT += prompt_token
                if cur_mode == "think":
                    config.TARGET_NORMAL_PROMPT[step_n] = prompt_token
                elif cur_mode == "act":
                    config.TARGET_NORMAL_PROMPT[step_n] += prompt_token
                # TODO logger
                logger.log(f"React {step_n} prompt token {prompt_token}")
                response = await self.llm.a_generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.TARGET_SP_GENERATION += response_token
                if cur_mode == "think":
                    config.TARGET_NORMAL_GENERATION[step_n] = response_token
                elif cur_mode == "act":
                    config.TARGET_NORMAL_GENERATION[step_n] += response_token
                logger.log(f"React {step_n} generation token {response_token}")

                request = format_step(response)

                # if ":" in request[:10] and "Action" in request[:10]:
                #     request = request[:10][request[:10].index(":") + 1 :] + request[10:]
                #     request = request.strip()
                return parse_request(request, step_n)
            except asyncio.CancelledError:
                # return "cancelled"
                # logger.log(f"Cancel step {step_n}.")
                raise
            except Exception as e:
                pass
                # print(f"Exception msg: {e}")
                # if "openai.APIConnectionError: Connection error." in e:
                #     return
                # traceback.print_exc()
            #     print("React api error!!!!!")
            #     catch_openai_api_error()
            #     await asyncio.sleep(0.1)

    def _build_agent_prompt(self, scratchpad) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(query=self.query, scratchpad=scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (
                len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length
            )
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self.__reset_record()
        self.json_log = []
        self.current_observation = ""
        self.current_data = None
        self.last_actions = []

        if "notebook" in self.tools:
            self.tools["notebook"].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record["invalidAction"] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module(".tools.{}.apis".format(tool_name), package="TravelPlanner")
            tools_map[tool_name] = getattr(
                module, tool_name[0].upper() + tool_name[1:]
            )()
            if tool_name == "planner" and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )(model_name=planner_model_name)
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, "r").read().strip().split("\n")
        for unit in lines:
            city_set.append(unit)
        return city_set


class DirectAgent:

    def __init__(
        self,
        args,
        mode: str = "zero_shot",
        tools: List[str] = None,
        max_steps: int = 30,
        max_retries: int = 3,
        illegal_early_stop_patience: int = 3,
        react_llm_name="gpt-4.1-mini",
        planner_llm_name="gpt-4.1-mini",
        #  logs_path = '../logs/',
        city_file_path="./database/background/citySet.txt",
    ) -> None:

        self.answer = ""
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == "zero_shot":
            self.agent_prompt = zeroshot_react_agent_prompt

        self.current_observation = ""
        self.current_data = None

        if "gpt-4.1-mini" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "base_url": "https://api.openai.com/v1",
                    "api_type": "openai",
                    "cache_seed": None, 
                    "price": [0.4, 1.6],
                    "temperature": 0,
                    "top_p": 1.0,
                    "seed":0
                },]
            self.llm = AssistantAgent(
                "assistant",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
            )
        elif "deepseek-chat" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            config_list = [{
                "model": "deepseek-chat",
                "api_key": os.environ['DEEPSEEK_API_KEY'],
                "base_url": "https://api.deepseek.com",
                "api_type": "deepseek",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
            self.llm = AssistantAgent("assistant", 
                llm_config={"config_list": config_list}, 
                human_input_mode='NEVER'
            )

        # define tolls and relevant parameters
        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record["invalidAction"] = 0

        # print(self.retry_record)

        self.last_actions = []

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "../database/background/citySet.txt")
        city_file_path = os.path.abspath(data_path)

        self.city_set = self.load_city(city_set_path=city_file_path)

        self.enc = tiktoken.get_encoding("cl100k_base")

        self.__reset_agent()

    def update_scratchpad(self, action, observation, step_number):
        step_number += 1
        try:
            scratchpad_start = self.scratchpad.index(f"\nAction {step_number}")
            self.scratchpad = self.scratchpad[:scratchpad_start]
        except:
            print(f"\nAction {step_number}")
            print(self.scratchpad)
        self.scratchpad += (
            f"\nAction {step_number}:"
            + action
            + "\n"
            + f"Observation {step_number}:"
            + observation
        )
        # update step number as well
        self.step_n = step_number + 1

    def execute(self, action):
        action_type, action_arg = parse_action(action)
        # compute observation here
        if action_type != "Planner":
            if action_type in actionMapping:
                pending_action = actionMapping[action_type]
            elif action_type not in actionMapping:
                pending_action = "invalidAction"

            if pending_action in self.retry_record:
                if self.retry_record[pending_action] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"{pending_action} early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

            elif pending_action not in self.retry_record:
                if self.retry_record["invalidAction"] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"invalidAction Early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

        if action_type == "FlightSearch":
            try:
                if (
                    validate_date_format(action_arg.split(", ")[2])
                    and validate_city_format(action_arg.split(", ")[0], self.city_set)
                    and validate_city_format(action_arg.split(", ")[1], self.city_set)
                ):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["flights"].run(
                        action_arg.split(", ")[0],
                        action_arg.split(", ")[1],
                        action_arg.split(", ")[2],
                    )
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except DateError:
                self.retry_record["flights"] += 1
                self.current_observation = (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )
                self.scratchpad += (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )

            except ValueError as e:
                self.retry_record["flights"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["flights"] += 1
                self.current_observation = f"Illegal Flight Search. Please try again."
                self.scratchpad += f"Illegal Flight Search. Please try again."

        elif action_type == "AttractionSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["attractions"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["attractions"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["attractions"] += 1
                self.current_observation = (
                    f"Illegal Attraction Search. Please try again."
                )
                self.scratchpad += f"Illegal Attraction Search. Please try again."

        elif action_type == "AccommodationSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["accommodations"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["accommodations"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["accommodations"] += 1
                self.current_observation = (
                    f"Illegal Accommodation Search. Please try again."
                )
                self.scratchpad += f"Illegal Accommodation Search. Please try again."

        elif action_type == "RestaurantSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["restaurants"].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except ValueError as e:
                self.retry_record["restaurants"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]["state"] = f"Illegal args. City Error"

            except Exception as e:
                print(e)
                self.retry_record["restaurants"] += 1
                self.current_observation = (
                    f"Illegal Restaurant Search. Please try again."
                )
                self.scratchpad += f"Illegal Restaurant Search. Please try again."

        elif action_type == "CitySearch":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                # self.current_data = self.tools['cities'].run(action_arg)
                self.current_observation = to_string(
                    self.tools["cities"].run(action_arg)
                ).strip()
                self.scratchpad += self.current_observation
                self.__reset_record()

            except ValueError as e:
                self.retry_record["cities"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["cities"] += 1
                self.current_observation = f"Illegal City Search. Please try again."
                self.scratchpad += f"Illegal City Search. Please try again."

        elif action_type == "GoogleDistanceMatrix":

            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_data = self.tools["googleDistanceMatrix"].run(
                    action_arg.split(", ")[0],
                    action_arg.split(", ")[1],
                    action_arg.split(", ")[2],
                )
                self.current_observation = to_string(self.current_data)
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["googleDistanceMatrix"] += 1
                self.current_observation = (
                    f"Illegal GoogleDistanceMatrix. Please try again."
                )
                self.scratchpad += f"Illegal GoogleDistanceMatrix. Please try again."

        elif action_type == "NotebookWrite":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_observation = str(
                    self.tools["notebook"].write(self.current_data, action_arg)
                )
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["notebook"] += 1
                self.current_observation = f"{e}"
                self.scratchpad += f"{e}"

        elif action_type == "Planner":
            # try:

            self.current_observation = str(
                self.tools["planner"].run(
                    str(self.tools["notebook"].list_all()), action_arg
                )
            )
            self.scratchpad += self.current_observation
            self.answer = self.current_observation
            self.__reset_record()

        else:
            self.retry_record["invalidAction"] += 1
            self.current_observation = (
                "Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / "
                "AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query]."
            )
            self.scratchpad += self.current_observation

    async def direct_act(self, step_n, config, logger):
        # Act
        self.scratchpad += f"\nGenerate Action only based on the query and existent action trajectory\nAction {self.step_n}:"
        action = await self.prompt_agent(step_n, config, logger)
        if action.lower() == 'terminate':
            self.finished=True
            return action, self.finished

        if action == None or action == "" or action == "\n":
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += " " + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            self.finished = True
            return action, self.finished

        # Observe
        # print("\n=========Observation===========")
        self.scratchpad += f"\nObservation {self.step_n}: "

        if action == None or action == "" or action == "\n":
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        else:
            # find action
            self.execute(action)

        self.step_n += 1

        action_type, action_arg = parse_action(action)
        if (
            action_type
            and action_type == "Planner"
            and self.retry_record["planner"] == 0
        ):

            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return action, self.finished
        else:
            return action, False

    def prompt_agent(self, step_n, config, logger) -> str:
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt()
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.APPROX_SP_PROMPT += prompt_token
                config.APPROX_NORMAL_PROMPT[step_n] = prompt_token
                logger.log(f'Approximation: Step {step_n+1} -prompt token {prompt_token}')

                response = self.llm.generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.APPROX_SP_GENERATION += response_token
                config.APPROX_NORMAL_GENERATION[step_n] = response_token

                request = format_step(response)
                
                return parse_request(request, step_n)
            except:
                catch_openai_api_error()
                time.sleep(5)

    async def prompt_agent(self, step_n, config, logger) -> str:
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt()                
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.APPROX_SP_PROMPT += prompt_token
                config.APPROX_NORMAL_PROMPT[step_n] = prompt_token
                logger.log(f'Approximation: Step {step_n+1} -prompt token {prompt_token}')
               
                response = await self.llm.a_generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.APPROX_SP_GENERATION += response_token
                config.APPROX_NORMAL_GENERATION[step_n] = response_token

                request = format_step(response)

                return parse_request(request, step_n)
            except asyncio.CancelledError:
                return "cancelled"
            except:
                catch_openai_api_error()
                await asyncio.sleep(0.1)
                

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query, scratchpad=self.scratchpad
            )

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (
                len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length
            )
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self.__reset_record()
        self.json_log = []
        self.current_observation = ""
        self.current_data = None
        self.last_actions = []

        if "notebook" in self.tools:
            self.tools["notebook"].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record["invalidAction"] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            tools_map[tool_name] = getattr(
                module, tool_name[0].upper() + tool_name[1:]
            )()
            if tool_name == "planner" and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )(model_name=planner_model_name)
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, "r").read().strip().split("\n")
        for unit in lines:
            city_set.append(unit)
        return city_set





class CoTAgent:

    def __init__(
        self,
        args,
        mode: str = "zero_shot",
        tools: List[str] = None,
        max_steps: int = 30,
        max_retries: int = 3,
        illegal_early_stop_patience: int = 3,
        react_llm_name="gpt-4-turbo",
        planner_llm_name="gpt-4-turbo",
        #  logs_path = '../logs/',
        city_file_path="../database/background/citySet.txt",
    ) -> None:

        self.answer = ""
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == "zero_shot":
            self.agent_prompt = zeroshot_react_agent_prompt

        self.current_observation = ""
        self.current_data = None

        if "gpt-4.1-mini" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "api_type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "cache_seed": None, 
                    "price": [0.4, 1.6],
                    "temperature": 0,
                    "top_p": 1.0,
                    "seed":0
                },]
            self.llm = AssistantAgent(
                "assistant",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
            )
        elif "deepseek-chat" in react_llm_name:
            stop_list = ["\n"]
            config_list = [{
                "model": "deepseek-chat",
                "api_key": os.environ['DEEPSEEK_API_KEY'],
                "base_url": "https://api.deepseek.com",
                "api_type": "deepseek",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
            self.llm = AssistantAgent("assistant", 
                llm_config={"config_list": config_list}, 
                human_input_mode='NEVER'
            )

        # define tolls and relevant parameters
        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record["invalidAction"] = 0

        # print(self.retry_record)

        self.last_actions = []

        self.city_set = self.load_city(city_set_path=city_file_path)

        # self.enc = tiktoken.encoding_for_model(react_llm_name)
        self.enc = tiktoken.get_encoding("cl100k_base")

        self.__reset_agent()

    def update_scratchpad(self, action, observation, step_number):
        step_number += 1
        try:
            scratchpad_start = self.scratchpad.index(f"\Think and Action {step_number}")
            self.scratchpad = self.scratchpad[:scratchpad_start]
        except:
            print(f"\nAction {step_number}")
            print(self.scratchpad)
        self.scratchpad += (
            f"\nAction {step_number}:"
            + action
            + "\n"
            + f"Observation {step_number}:"
            + observation
        )
        # update step number as well
        self.step_n = step_number + 1

    def execute(self, action):
        action_type, action_arg = parse_action(action)
        # compute observation here
        if action_type != "Planner":
            if action_type in actionMapping:
                pending_action = actionMapping[action_type]
            elif action_type not in actionMapping:
                pending_action = "invalidAction"

            if pending_action in self.retry_record:
                if self.retry_record[pending_action] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"{pending_action} early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

            elif pending_action not in self.retry_record:
                if self.retry_record["invalidAction"] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"invalidAction Early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

        if action_type == "FlightSearch":
            try:
                if (
                    validate_date_format(action_arg.split(", ")[2])
                    and validate_city_format(action_arg.split(", ")[0], self.city_set)
                    and validate_city_format(action_arg.split(", ")[1], self.city_set)
                ):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["flights"].run(
                        action_arg.split(", ")[0],
                        action_arg.split(", ")[1],
                        action_arg.split(", ")[2],
                    )
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except DateError:
                self.retry_record["flights"] += 1
                self.current_observation = (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )
                self.scratchpad += (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )

            except ValueError as e:
                self.retry_record["flights"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["flights"] += 1
                self.current_observation = f"Illegal Flight Search. Please try again."
                self.scratchpad += f"Illegal Flight Search. Please try again."

        elif action_type == "AttractionSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["attractions"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["attractions"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["attractions"] += 1
                self.current_observation = (
                    f"Illegal Attraction Search. Please try again."
                )
                self.scratchpad += f"Illegal Attraction Search. Please try again."

        elif action_type == "AccommodationSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["accommodations"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["accommodations"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["accommodations"] += 1
                self.current_observation = (
                    f"Illegal Accommodation Search. Please try again."
                )
                self.scratchpad += f"Illegal Accommodation Search. Please try again."

        elif action_type == "RestaurantSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["restaurants"].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except ValueError as e:
                self.retry_record["restaurants"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]["state"] = f"Illegal args. City Error"

            except Exception as e:
                print(e)
                self.retry_record["restaurants"] += 1
                self.current_observation = (
                    f"Illegal Restaurant Search. Please try again."
                )
                self.scratchpad += f"Illegal Restaurant Search. Please try again."

        elif action_type == "CitySearch":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                # self.current_data = self.tools['cities'].run(action_arg)
                self.current_observation = to_string(
                    self.tools["cities"].run(action_arg)
                ).strip()
                self.scratchpad += self.current_observation
                self.__reset_record()

            except ValueError as e:
                self.retry_record["cities"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["cities"] += 1
                self.current_observation = f"Illegal City Search. Please try again."
                self.scratchpad += f"Illegal City Search. Please try again."

        elif action_type == "GoogleDistanceMatrix":

            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_data = self.tools["googleDistanceMatrix"].run(
                    action_arg.split(", ")[0],
                    action_arg.split(", ")[1],
                    action_arg.split(", ")[2],
                )
                self.current_observation = to_string(self.current_data)
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["googleDistanceMatrix"] += 1
                self.current_observation = (
                    f"Illegal GoogleDistanceMatrix. Please try again."
                )
                self.scratchpad += f"Illegal GoogleDistanceMatrix. Please try again."

        elif action_type == "NotebookWrite":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_observation = str(
                    self.tools["notebook"].write(self.current_data, action_arg)
                )
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["notebook"] += 1
                self.current_observation = f"{e}"
                self.scratchpad += f"{e}"

        elif action_type == "Planner":
            # try:

            self.current_observation = str(
                self.tools["planner"].run(
                    str(self.tools["notebook"].list_all()), action_arg
                )
            )
            self.scratchpad += self.current_observation
            self.answer = self.current_observation
            self.__reset_record()

        else:
            self.retry_record["invalidAction"] += 1
            self.current_observation = (
                "Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / "
                "AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query]."
            )
            self.scratchpad += self.current_observation

    async def direct_act(self, step_n, config, logger):
        # Act
        self.scratchpad += f"\nThink very carefully first and then generate the action that you think should be done based on the query and existent action trajectory\nThink and Action {self.step_n} and surround the action proposed in <action> and </action>:"
        action = await self.prompt_agent(step_n, config, logger)
        if action.lower() == 'terminate':
            self.finished=True
            return action, self.finished

        if action == None or action == "" or action == "\n":
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += " " + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            self.finished = True
            return action, self.finished

        # Observe
        # print("\n=========Observation===========")
        self.scratchpad += f"\nObservation {self.step_n}: "

        if action == None or action == "" or action == "\n":
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        else:
            # find action
            self.execute(action)

        self.step_n += 1

        action_type, action_arg = parse_action(action)
        if (
            action_type
            and action_type == "Planner"
            and self.retry_record["planner"] == 0
        ):

            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return action, self.finished
        else:
            return action, False

    def prompt_agent(self, step_n, config, logger) -> str:
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt()
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.APPROX_SP_PROMPT += prompt_token
                config.APPROX_NORMAL_PROMPT[step_n] = prompt_token
                logger.log(f'Approximation: Step {step_n+1} -prompt token {prompt_token}')

                response = self.llm.generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.APPROX_SP_GENERATION += response_token
                config.APPROX_NORMAL_GENERATION[step_n] = response_token

                request = format_step(response)
                return parse_request(request, step_n)
            except:
                catch_openai_api_error()
                time.sleep(5)

    async def prompt_agent(self, step_n, config, logger) -> str:
        n=0
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt()                
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.APPROX_SP_PROMPT += prompt_token
                config.APPROX_NORMAL_PROMPT[step_n] = prompt_token
                logger.log(f'Approximation: Step {step_n+1} -prompt token {prompt_token}')
               
                response = await self.llm.a_generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.APPROX_SP_GENERATION += response_token
                config.APPROX_NORMAL_GENERATION[step_n] = response_token

                request = format_step(response)

                # if ":" in request[:10] and "Action" in request[:10]:
                #     request = request[:10][request[:10].index(":") + 1 :] + request[10:]
                #     request = request.strip()
                # return request
                return parse_request(request, step_n)
            except asyncio.CancelledError:
                return "cancelled"
            except:
                catch_openai_api_error()
                await asyncio.sleep(0.1)

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query, scratchpad=self.scratchpad
            )

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (
                len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length
            )
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self.__reset_record()
        self.json_log = []
        self.current_observation = ""
        self.current_data = None
        self.last_actions = []

        if "notebook" in self.tools:
            self.tools["notebook"].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record["invalidAction"] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            tools_map[tool_name] = getattr(
                module, tool_name[0].upper() + tool_name[1:]
            )()
            if tool_name == "planner" and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )(model_name=planner_model_name)
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, "r").read().strip().split("\n")
        for unit in lines:
            city_set.append(unit)
        return city_set



class MultiAgent:
    def __init__(
        self,
        args,
        mode: str = "zero_shot",
        tools: List[str] = None,
        max_steps: int = 30,
        max_retries: int = 3,
        illegal_early_stop_patience: int = 3,
        react_llm_name="gpt-4-turbo",
        planner_llm_name="gpt-4-turbo",
        #  logs_path = '../logs/',
        city_file_path="../database/background/citySet.txt",
    ) -> None:

        self.answer = ""
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == "zero_shot":
            self.agent_prompt = zeroshot_react_agent_prompt

        self.current_observation = ""
        self.current_data = None

        if "gpt-4.1-mini" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "api_type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "cache_seed": None, 
                    "price": [0.4, 1.6],
                    "temperature": 0,
                    "top_p": 1.0,
                    "seed":0
                },]
            self.llm = AssistantAgent(
                "assistant",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
            )
        elif "deepseek-chat" in react_llm_name:
            stop_list = ["\n"]
            config_list = [{
                "model": "deepseek-chat",
                "api_key": os.environ['DEEPSEEK_API_KEY'],
                "base_url": "https://api.deepseek.com",
                "api_type": "deepseek",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
            self.llm = AssistantAgent("assistant", 
                llm_config={"config_list": config_list}, 
                human_input_mode='NEVER'
            )

        # define tolls and relevant parameters
        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record["invalidAction"] = 0

        self.last_actions = []

        self.city_set = self.load_city(city_set_path=city_file_path)

        # self.enc = tiktoken.encoding_for_model(react_llm_name)
        self.enc = tiktoken.get_encoding("cl100k_base")

        self.__reset_agent()

    def update_scratchpad(self, thought, action, observation, step_number):
        scratchpad_start = self.scratchpad.index(
            f"Generate thought only.\nThought {step_number}:"
        )
        self.scratchpad = self.scratchpad[:scratchpad_start]
        self.scratchpad += (
            f"Generate thought only.\nThought {step_number}:"
            + thought
            + "\n"
            + f"Generate Action only based on thoughts\nAction {step_number}:"
            + action
            + "\n"
            + f"Observation {step_number}:"
            + observation
        )

    def create_scratchpad(self, scratchpad, sas):
        for i, (action, observation) in enumerate(sas):
            scratchpad += (
                f"\nAction {i+1}: "
                + action
                + "\n"
                + f"Observation {i+1}: "
                + observation
            )
        return scratchpad

    def execute(self, action):
        action_type, action_arg = parse_action(action)
        # compute observation here
        if action_type != "Planner":
            if action_type in actionMapping:
                pending_action = actionMapping[action_type]
            elif action_type not in actionMapping:
                pending_action = "invalidAction"

            if pending_action in self.retry_record:
                if self.retry_record[pending_action] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"{pending_action} early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

            elif pending_action not in self.retry_record:
                if self.retry_record["invalidAction"] + 1 > self.max_retries:
                    action_type = "Planner"
                    print(
                        f"invalidAction Early stop due to {self.max_retries} max retries."
                    )
                    self.finished = True
                    return

        if action_type == "FlightSearch":
            try:
                if (
                    validate_date_format(action_arg.split(", ")[2])
                    and validate_city_format(action_arg.split(", ")[0], self.city_set)
                    and validate_city_format(action_arg.split(", ")[1], self.city_set)
                ):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["flights"].run(
                        action_arg.split(", ")[0],
                        action_arg.split(", ")[1],
                        action_arg.split(", ")[2],
                    )
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except DateError:
                self.retry_record["flights"] += 1
                self.current_observation = (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )
                self.scratchpad += (
                    f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                )

            except ValueError as e:
                self.retry_record["flights"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["flights"] += 1
                self.current_observation = f"Illegal Flight Search. Please try again."
                self.scratchpad += f"Illegal Flight Search. Please try again."

        elif action_type == "AttractionSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["attractions"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["attractions"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["attractions"] += 1
                self.current_observation = (
                    f"Illegal Attraction Search. Please try again."
                )
                self.scratchpad += f"Illegal Attraction Search. Please try again."

        elif action_type == "AccommodationSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["accommodations"].run(action_arg)
                    self.current_observation = (
                        to_string(self.current_data).strip("\n").strip()
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
            except ValueError as e:
                self.retry_record["accommodations"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
            except Exception as e:
                print(e)
                self.retry_record["accommodations"] += 1
                self.current_observation = (
                    f"Illegal Accommodation Search. Please try again."
                )
                self.scratchpad += f"Illegal Accommodation Search. Please try again."

        elif action_type == "RestaurantSearch":

            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip().strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )
                    self.current_data = self.tools["restaurants"].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()

            except ValueError as e:
                self.retry_record["restaurants"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]["state"] = f"Illegal args. City Error"

            except Exception as e:
                print(e)
                self.retry_record["restaurants"] += 1
                self.current_observation = (
                    f"Illegal Restaurant Search. Please try again."
                )
                self.scratchpad += f"Illegal Restaurant Search. Please try again."

        elif action_type == "CitySearch":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                # self.current_data = self.tools['cities'].run(action_arg)
                self.current_observation = to_string(
                    self.tools["cities"].run(action_arg)
                ).strip()
                self.scratchpad += self.current_observation
                self.__reset_record()

            except ValueError as e:
                self.retry_record["cities"] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)

            except Exception as e:
                print(e)
                self.retry_record["cities"] += 1
                self.current_observation = f"Illegal City Search. Please try again."
                self.scratchpad += f"Illegal City Search. Please try again."

        elif action_type == "GoogleDistanceMatrix":

            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_data = self.tools["googleDistanceMatrix"].run(
                    action_arg.split(", ")[0],
                    action_arg.split(", ")[1],
                    action_arg.split(", ")[2],
                )
                self.current_observation = to_string(self.current_data)
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["googleDistanceMatrix"] += 1
                self.current_observation = (
                    f"Illegal GoogleDistanceMatrix. Please try again."
                )
                self.scratchpad += f"Illegal GoogleDistanceMatrix. Please try again."

        elif action_type == "NotebookWrite":
            try:
                self.scratchpad = self.scratchpad.replace(
                    to_string(self.current_data).strip(),
                    "Masked due to limited length. Make sure the data has been written in Notebook.",
                )
                self.current_observation = str(
                    self.tools["notebook"].write(self.current_data, action_arg)
                )
                self.scratchpad += self.current_observation
                self.__reset_record()

            except Exception as e:
                print(e)
                self.retry_record["notebook"] += 1
                self.current_observation = f"{e}"
                self.scratchpad += f"{e}"

        elif action_type == "Planner":
            # try:

            self.current_observation = str(
                self.tools["planner"].run(
                    str(self.tools["notebook"].list_all()), action_arg
                )
            )
            self.scratchpad += self.current_observation
            self.answer = self.current_observation
            self.__reset_record()

        else:
            self.retry_record["invalidAction"] += 1
            self.current_observation = (
                "Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / "
                "AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query]."
            )
            self.scratchpad += self.current_observation

    async def think_and_act(self, scratchpad, step_n, config, logger):
        # breakpoint()
        step_n += 1
        # Act
        scratchpad += f"\nThink very carefully first and then generate the action that you think should be done based on the query and existent action trajectory\nThink and Action {step_n} and surround the action proposed in <action> and </action>:"
        number = 1
        # round one
        action_A = await self.prompt_agent(scratchpad, number, step_n, config, logger)
        number += 1

        action_B = await self.prompt_agent(scratchpad, number, step_n, config, logger)
        number += 1

        scratchpad.replace(f"\nThink very carefully first and then generate the action that you think should be done based on the query and existent action trajectory\nThink and Action {step_n} and surround the action proposed in <action> and </action>:",'')

        scratchpad += f"You initially think the action to do next is {action_A}. Another planner agent also thought about this carefully and that agent believes the action should be {action_B}. What do you think? Do you want to change your opinion? \nThink and Action {step_n} and surround the action proposed in <action> and </action>:"

        # round two
        action_A = await self.prompt_agent(scratchpad, number, step_n, config, logger)
        number += 1

        scratchpad.replace(f"You initially think the action to do next is {action_A}. Another planner agent also thought about this carefully and that agent believes the action should be {action_B}. What do you think? Do you want to change your opinion? \nThink and Action {step_n} and surround the action proposed in <action> and </action>:",'')

        scratchpad += f"You initially think the action to do next is {action_B}. Another planner agent also thought about this carefully and that agent believes the action should be {action_A}. What do you think? Do you want to change your opinion?"    
        scratchpad += f"\nGenerate thought only.\nThought {step_n}:"

        thought = await self.prompt_agent(scratchpad, number, step_n, config, logger)
        number += 1
        
        scratchpad += " " + thought

        # Act
        scratchpad += f"\nGenerate Action only based on thoughts\nAction {step_n}:"
        # print("scratchpad for action:")
        # print(scratchpad)
        # print("-" * 50)
        action = await self.prompt_agent(scratchpad, number, step_n, config, logger)

        scratchpad.replace(f"You initially think the action to do next is {action_B}. Another planner agent also thought about this carefully and that agent believes the action should be {action_A}. What do you think? Do you want to change your opinion? \nThink and Action {step_n} and surround the action proposed in <action> and </action>:", '')

        if action.lower() == 'terminate':
            self.finished=True
            return action, self.finished

        if action == None or action == "" or action == "\n":
            scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            scratchpad += " " + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            self.finished = True
            return action, self.finished

        if action == None or action == "" or action == "\n":
            action_type = None
            action_arg = None
            scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        else:
            # find action
            action_type, action_arg = parse_action(action)

        self.step_n += 1

        if (
            action_type
            and action_type == "Planner"
            and self.retry_record["planner"] == 0
        ):

            self.finished = True
            return action, self.finished
        else:
            return action, False

    def prompt_agent(self, scratchpad, number, step_n, config, logger) -> str:
        n=0 
        while True:
            try:
                n += 1
                if n >= 10:
                    result = ''
                    return result
                prompt = self._build_agent_prompt(scratchpad)
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.TARGET_SP_PROMPT += prompt_token
                if number == 1:
                    config.TARGET_NORMAL_PROMPT[step_n] = prompt_token
                else: config.TARGET_NORMAL_PROMPT[step_n] += prompt_token
                response = self.llm.generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                
                response_token = len(self.enc.encode(response))

                config.TOTAL_TOKEN_GENERATION += response_token
                config.TARGET_SP_GENERATION += response_token
                if number == 1:
                    config.TARGET_NORMAL_GENERATION[step_n] = response_token
                else: config.TARGET_NORMAL_GENERATION[step_n] += response_token

                request = format_step(response)
                return request
            except asyncio.CancelledError:
                return "cancelled"
            except:
                # catch_openai_api_error()
                time.sleep(0.1)

    async def prompt_agent(self, scratchpad, number, step_n, config, logger) -> str:
        n=0
        # print("hit async")
        while True:
            try:
                n += 1
                if n >= 3:
                    result = ''
                    return result
                prompt = self._build_agent_prompt(scratchpad)
                prompt_token = len(self.enc.encode(prompt))
                config.TOTAL_TOKEN_PROMPT += prompt_token
                config.TARGET_SP_PROMPT += prompt_token
                if number == 1:
                    config.TARGET_NORMAL_PROMPT[step_n] = prompt_token
                else: config.TARGET_NORMAL_PROMPT[step_n] += prompt_token
                # logger.log(f'Target: Step {step_n} Debate number {number} -prompt token {prompt_token}')

                response = await self.llm.a_generate_reply(
                    messages=[{"content": prompt, "role": "user"}]
                )
                # logger.log(f'Target: Step {step_n} Debate number {number} -gen token {response_token}')

                response_token = len(self.enc.encode(response))
                config.TOTAL_TOKEN_GENERATION += response_token
                config.TARGET_SP_GENERATION += response_token
                if number == 1:
                    config.TARGET_NORMAL_GENERATION[step_n] = response_token
                else: config.TARGET_NORMAL_GENERATION[step_n] += response_token
                request = format_step(response)

                return parse_request(request, step_n)
            except asyncio.CancelledError:
                # return "cancelled"
                # logger.log(f"Cancel step {step_n}.")
                raise
            except:
                pass
                # catch_openai_api_error()
                # await asyncio.sleep(0.1)

    

    

    def _build_agent_prompt(self, scratchpad) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(query=self.query, scratchpad=scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (
                len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length
            )
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self.__reset_record()
        self.json_log = []
        self.current_observation = ""
        self.current_data = None
        self.last_actions = []

        if "notebook" in self.tools:
            self.tools["notebook"].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record["invalidAction"] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            tools_map[tool_name] = getattr(
                module, tool_name[0].upper() + tool_name[1:]
            )()
            if tool_name == "planner" and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )(model_name=planner_model_name)
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, "r").read().strip().split("\n")
        for unit in lines:
            city_set.append(unit)
        return city_set



### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string):
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except:
        return None, None


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def truncate_scratchpad(
    scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc
) -> str:
    lines = scratchpad.split("\n")
    observations = filter(lambda x: x.startswith("Observation"), lines)
    observations_by_tokens = sorted(
        observations, key=lambda x: len(tokenizer.encode(x))
    )
    while len(gpt2_enc.encode("\n".join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = (
            largest_observation.split(":")[0] + ": [truncated wikipedia excerpt]"
        )
    return "\n".join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def remove_observation_lines(text, step_n):
    pattern = re.compile(rf"^Observation {step_n}.*", re.MULTILINE)
    return pattern.sub("", text)


def validate_date_format(date_str: str) -> bool:
    pattern = r"^\d{4}-\d{2}-\d{2}$"

    if not re.match(pattern, date_str):
        raise DateError
    return True


def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True


def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")

    # Initialize an empty dictionary to store the results
    result = {}

    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment

        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")

        # Store the result with the operator included
        result[key] = (operator, value)

    return result


def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)
