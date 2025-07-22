"""Agent implementations for the speculative planning system."""
from typing import List, Union
from autogen import AssistantAgent
from .openagi_utils import parse_response
import os

class BaseAgent:
    """Base agent class with direct generation approach."""
    def __init__(self, assistant: AssistantAgent, logger, encoding=None, config=None):
        self.assistant = assistant
        self.logger = logger
        self.encoding = encoding
        self.config = config

    async def generate(self, prompt: str, step_number: int) -> str:
        """Generate a response using direct approach."""
        prompt += f"\n\nDirectly tell me what **the ONE NEXT action step** based on the current action trajectory should be. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {step_number}?\n\nStep {step_number}:"
        
        # Record prompt tokens
        prompt_token = len(self.encoding.encode(prompt))
        self.config.TOTAL_TOKEN_PROMPT += prompt_token
        self.config.APPROX_SP_PROMPT += prompt_token
        self.config.APPROX_NORMAL_PROMPT[step_number] = prompt_token
        self.logger.log(f'Step {step_number} prompt token: {prompt_token}')

        # Generate response
        response = await self.assistant.a_generate_reply(
            messages=[{'content': prompt, 'role': 'user'}]
        )
        
        # Record generation tokens
        response_token = len(self.encoding.encode(response))
        self.config.TOTAL_TOKEN_GENERATION += response_token
        self.config.APPROX_SP_GENERATION += response_token
        self.config.APPROX_NORMAL_GENERATION[step_number] = response_token
        self.logger.log(f'Step {step_number} generation token: {response_token}')
        
        return parse_response(response)

class ReActAgent(BaseAgent):
    """Agent that uses ReAct approach for generation."""
    
    async def generate(self, prompt: str, step_number: int) -> str:
        """Generate response using ReAct approach."""
        # First generate thought
        thought_prompt = prompt + "\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory."
        thought_prompt += f"\nGenerate thought only.\nThought {step_number}:"
        
        self.logger.log('react launch api for thought.')
        
        # Record thought prompt tokens
        thought_prompt_token = len(self.encoding.encode(thought_prompt))
        self.config.TARGET_NORMAL_PROMPT[step_number] = thought_prompt_token
        self.config.TOTAL_TOKEN_PROMPT += thought_prompt_token
        self.logger.log(f"Target step {step_number} thought prompt token: {thought_prompt_token}")

        # Generate thought
        thought = await self.assistant.a_generate_reply(
            messages=[{'content': thought_prompt, 'role': 'user'}]
        )

        # Record thought generation tokens
        thought_token = len(self.encoding.encode(thought))
        self.config.TOTAL_TOKEN_GENERATION += thought_token
        self.config.TARGET_NORMAL_GENERATION[step_number] = thought_token
        self.logger.log(f"Target step {step_number} thought generation token: {thought_token}")

        # Generate action based on thought
        action_prompt = thought_prompt + " " + thought
        action_prompt += f"\nGenerate Action only based on thoughts. Remember to use xml tag <tool> and </tool> for formatting. \nAction {step_number}:"
        
        self.logger.log('react launch api for response.')
        
        # Record action prompt tokens
        action_prompt_token = len(self.encoding.encode(action_prompt))
        self.config.TARGET_NORMAL_PROMPT[step_number] += action_prompt_token
        self.config.TOTAL_TOKEN_PROMPT += action_prompt_token
        self.logger.log(f"Target step {step_number} response prompt token: {action_prompt_token}")

        # Generate action
        response = await self.assistant.a_generate_reply(
            messages=[{'content': action_prompt, 'role': 'user'}]
        )
        
        # Record action generation tokens
        response_token = len(self.encoding.encode(response))
        self.config.TOTAL_TOKEN_GENERATION += response_token
        self.config.TARGET_NORMAL_GENERATION[step_number] += response_token
        self.logger.log(f"Target step {step_number} response generation token: {response_token}")

        return parse_response(response)

class CoTAgent(BaseAgent):
    """Agent that uses Chain of Thought approach for generation."""
    
    async def generate(self, prompt: str, step_number: int) -> str:
        """Generate response using Chain of Thought approach."""
        prompt += "\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory, by first providing a clear reasoning chain. And then decide which tool to use for the current step."
        prompt += f"\nWhat should be the action in Step {step_number}? Remember to use xml tag <tool> and </tool> for formatting.\nStep {step_number}:"
        
        # Record prompt tokens
        prompt_token = len(self.encoding.encode(prompt))
        self.config.TOTAL_TOKEN_PROMPT += prompt_token
        self.config.APPROX_SP_PROMPT += prompt_token
        self.config.APPROX_NORMAL_PROMPT[step_number] = prompt_token
        self.logger.log(f'Step {step_number} prompt token: {prompt_token}')

        # Generate response
        response = await self.assistant.a_generate_reply(
            messages=[{'content': prompt, 'role': 'user'}]
        )
        
        # Record generation tokens
        response_token = len(self.encoding.encode(response))
        self.config.TOTAL_TOKEN_GENERATION += response_token
        self.config.APPROX_SP_GENERATION += response_token
        self.config.APPROX_NORMAL_GENERATION[step_number] = response_token
        self.logger.log(f'Step {step_number} generation token: {response_token}')
        
        return parse_response(response)

class MultiAgent(BaseAgent):
    """Agent that uses multi-agent discussion for generation."""
    
    def __init__(self, assistants: List[AssistantAgent], logger, encoding=None, config=None):
        if not isinstance(assistants, list) or len(assistants) != 2:
            raise ValueError("MultiAgent requires exactly 2 assistants")
        super().__init__(assistants[0], logger, encoding, config)
        self.assistant_b = assistants[1]

    async def generate(self, prompt: str, step_number: int) -> str:
        """Generate response using multi-agent discussion."""
        # First round of discussion
        prompt4A = prompt + f"\n\nYou will discuss with another agent about **the ONE NEXT action step** based on the current action trajectory. Please provide your thought and answer first.\nAction {step_number}:"
        
        # Record A1 prompt tokens
        prompt4A_token = len(self.encoding.encode(prompt4A))
        self.config.TARGET_NORMAL_PROMPT[step_number] = prompt4A_token
        self.config.TOTAL_TOKEN_PROMPT += prompt4A_token
        self.logger.log(f"Target step {step_number} thought A prompt token: {prompt4A_token}")

        # Generate A1 response
        thoughtA = await self.assistant.a_generate_reply(
            messages=[{'content': prompt4A, 'role': 'user'}]
        )
        if isinstance(thoughtA, dict):
            thoughtA = thoughtA['content']

        # Record A1 generation tokens
        thoughtA_token = len(self.encoding.encode(thoughtA))
        self.config.TOTAL_TOKEN_GENERATION += thoughtA_token
        self.config.TARGET_NORMAL_GENERATION[step_number] = thoughtA_token
        self.logger.log(f"Target step {step_number} thought A generation token: {thoughtA_token}")

        # B1's turn
        prompt4B = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nThe other agent's idea about this step is {thoughtA}.\nPlease think about whether the other agent's thought and idea is useful, and then provide your thought and answer now.\nAction {step_number}:"
        
        # Record B1 prompt tokens
        prompt4B_token = len(self.encoding.encode(prompt4B))
        self.config.TARGET_NORMAL_PROMPT[step_number] += prompt4B_token
        self.config.TOTAL_TOKEN_PROMPT += prompt4B_token
        self.logger.log(f"Target step {step_number} thought B prompt token: {prompt4B_token}")

        # Generate B1 response
        thoughtB = await self.assistant_b.a_generate_reply(
            messages=[{'content': prompt4B, 'role': 'user'}]
        )
        if isinstance(thoughtB, dict):
            thoughtB = thoughtB['content']

        # Record B1 generation tokens
        thoughtB_token = len(self.encoding.encode(thoughtB))
        self.config.TOTAL_TOKEN_GENERATION += thoughtB_token
        self.config.TARGET_NORMAL_GENERATION[step_number] += thoughtB_token
        self.logger.log(f"Target step {step_number} thought B generation token: {thoughtB_token}")

        # Second round - A2's turn
        prompt4A = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtA}.\nThe other agent's thought and idea about this step is {thoughtB}.\nPlease summarize and reflect, and then update your thought on what this step should be after updating.\nAction {step_number}:"
        
        # Record A2 prompt tokens
        prompt4A_token = len(self.encoding.encode(prompt4A))
        self.config.TARGET_NORMAL_PROMPT[step_number] += prompt4A_token
        self.config.TOTAL_TOKEN_PROMPT += prompt4A_token
        self.logger.log(f"Target step {step_number} round 2 thought A prompt token: {prompt4A_token}")

        # Generate A2 response
        thoughtA = await self.assistant.a_generate_reply(
            messages=[{'content': prompt4A, 'role': 'user'}]
        )
        if isinstance(thoughtA, dict):
            thoughtA = thoughtA['content']

        # Record A2 generation tokens
        thoughtA_token = len(self.encoding.encode(thoughtA))
        self.config.TOTAL_TOKEN_GENERATION += thoughtA_token
        self.config.TARGET_NORMAL_GENERATION[step_number] += thoughtA_token
        self.logger.log(f"Target step {step_number} round 2 thought A generation token: {thoughtA_token}")

        # B2's turn
        prompt4B = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtB}.\nThe other agent's thought and idea about this step is {thoughtA}.\nPlease summarize and reflect, and then update your thought on what this step should be after updating.\nAction {step_number}:"
        
        # Record B2 prompt tokens
        prompt4B_token = len(self.encoding.encode(prompt4B))
        self.config.TARGET_NORMAL_PROMPT[step_number] += prompt4B_token
        self.config.TOTAL_TOKEN_PROMPT += prompt4B_token
        self.logger.log(f"Target step {step_number} round 2 thought B prompt token: {prompt4B_token}")

        # Generate B2 response
        thoughtB = await self.assistant_b.a_generate_reply(
            messages=[{'content': prompt4B, 'role': 'user'}]
        )
        if isinstance(thoughtB, dict):
            thoughtB = thoughtB['content']

        # Record B2 generation tokens
        thoughtB_token = len(self.encoding.encode(thoughtB))
        self.config.TOTAL_TOKEN_GENERATION += thoughtB_token
        self.config.TARGET_NORMAL_GENERATION[step_number] += thoughtB_token
        self.logger.log(f"Target step {step_number} round 2 thought B generation token: {thoughtB_token}")

        # Final round - Generate action
        final_prompt = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtA}.\nThe other agent's thought and idea about this step is {thoughtB}.\nPlease summarize and reflect, and then provide your thought and answer for what this step should be.\nAction {step_number}:"
        
        # Record final prompt tokens
        final_prompt_token = len(self.encoding.encode(final_prompt))
        self.config.TARGET_NORMAL_PROMPT[step_number] += final_prompt_token
        self.config.TOTAL_TOKEN_PROMPT += final_prompt_token
        self.logger.log(f"Target step {step_number} final prompt token: {final_prompt_token}")

        # Generate final response
        response = await self.assistant.a_generate_reply(
            messages=[{'content': final_prompt, 'role': 'user'}]
        )
        if isinstance(response, dict):
            response = response['content']

        # Record final generation tokens
        response_token = len(self.encoding.encode(response))
        self.config.TOTAL_TOKEN_GENERATION += response_token
        self.config.TARGET_NORMAL_GENERATION[step_number] += response_token
        self.logger.log(f"Target step {step_number} final generation token: {response_token}")

        return parse_response(response)

def create_agent(agent_type: str, assistant: Union[AssistantAgent, List[AssistantAgent]], logger, encoding=None, config=None) -> BaseAgent:
    """Factory function to create appropriate agent type."""
    if agent_type == 'react':
        return ReActAgent(assistant, logger, encoding, config)
    elif agent_type == 'multi_agent':
        if not isinstance(assistant, list):
            raise ValueError("MultiAgent requires a list of assistants")
        return MultiAgent(assistant, logger, encoding, config)
    elif agent_type == 'cot':
        return CoTAgent(assistant, logger, encoding, config)
    else:  # direct
        return BaseAgent(assistant, logger, encoding, config)

def create_agent_config(model_name, api_key, api_type="openai", base_url=None):
    """Create agent configuration."""
    config = {
        "model": model_name,
        "api_key": api_key,
        "api_type": api_type,
        "cache_seed": None,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 0
    }
    
    if base_url:
        config["base_url"] = base_url
        
    return [config]

def setup_assistants(model_type):
    """Set up base approximation and target agents."""
    if model_type == "deepseek":
        app_config = create_agent_config(
            "deepseek-chat",
            os.environ['DEEPSEEK_API_KEY'],
            api_type="deepseek",
            base_url="https://api.deepseek.com/"
        )
        tar_config = create_agent_config(
            "deepseek-reasoner",
            os.environ['DEEPSEEK_API_KEY'],
            api_type="deepseek",
            base_url="https://api.deepseek.com/" 
        )
    else:
        # For OpenAI models
        app_config = create_agent_config(model_type, os.environ['OPENAI_API_KEY'])
        tar_config = create_agent_config(model_type, os.environ['OPENAI_API_KEY'])
    
    app_assistant = AssistantAgent(
        "assistant",
        llm_config={"config_list": app_config},
        human_input_mode='NEVER'
    )
    
    tar_assistant = AssistantAgent(
        "assistant", 
        llm_config={"config_list": tar_config},
        human_input_mode='NEVER'
    )
    
    return app_assistant, tar_assistant

def setup_multi_agent(model_type, tar_assistant):
    """Set up multi-agent configuration for target agent."""
    if model_type == "deepseek":
        tar_config = create_agent_config(
            "deepseek-reasoner",
            os.environ['DEEPSEEK_API_KEY'],
            api_type="deepseek", 
            base_url="https://api.deepseek.com"
        )
    else:
        # For OpenAI models
        tar_config = create_agent_config(model_type, os.environ['OPENAI_API_KEY'])
    
    tar_assistantB = AssistantAgent(
        "assistant",
        llm_config={"config_list": tar_config},
        human_input_mode='NEVER'
    )
    
    return [tar_assistant, tar_assistantB] 