"""Utility functions for the speculative planning system."""

import asyncio
from datetime import datetime
import time

def parse_response(response: str) -> str:
    """Parse the response from agents to extract the action."""
    if '<' in response and '>' in response and '</' in response:
        all_starts = [i for i in range(len(response)-1) if response[i] == '<' and response[i+1] != '/']
        all_ends = [i for i in range(len(response)-1) if response[i:i+2] == '</']
        start = all_starts[-1]
        end = all_ends[-1]
        return response[start:end].replace('<tool>', '').replace('<', '').replace('>', '')
    elif '**' in response and response.count('**') >= 2:
        start = response.index('**') + len('**')
        response = response[start:]
        end = response.index('**')
        return response[:end].replace('<', '').replace('>', '')
    else:
        return ''

def ordinal(n):
    """Convert a number to its ordinal representation."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def load_data(data_path):
    """Load task descriptions from a file."""
    with open(data_path, "r") as f:
        data = f.read()
    return [t.strip() for t in data.split("\n")]

def judge_to_be_true(s: str, t: str) -> bool:
    """Check if two actions are equivalent."""
    return s == t

def concurrent_calls():
    """Get the number of concurrent tasks."""
    tasks = asyncio.all_tasks()
    pending_tasks = [t for t in tasks if not t.done() and not t.cancelled()]
    return len(pending_tasks)

def get_timestamp():
    """Get current timestamp."""
    return datetime.fromtimestamp(time.time())

def create_prompt(task_description):
    """Create the initial prompt for the agents."""
    tools = """
Available tools are as follows:

(1) <tool>Sentiment Analysis</tool>
(2) <tool>Text Summarization</tool>
(3) <tool>Machine Translation</tool>
(4) <tool>Fill Mask</tool>
(5) <tool>Question Answering</tool>
(6) <tool>Image Classification</tool>
(7) <tool>Object Detection</tool>
(8) <tool>Colorization</tool>
(9) <tool>Image Super-Resolution</tool>
(10) <tool>Image Denoising</tool>
(11) <tool>Image Deblurring</tool>
(12) <tool>Visual Question Answering</tool>
(13) <tool>Image Captioning</tool>
(14) <tool>Text-to-Image Generation</tool>
(15) <tool>TERMINATE</tool>

For each step of the plan, please specify the tool you would like to use. But if you think the task is completed, please use <tool>TERMINATE</tool> to end the conversation.

Please use xml tags to specify the tool when responsing. For example, <tool>Sentiment Analysis</tool> for Sentiment Analysis.
"""
    return f"## Problem: {task_description}\nPlease solve this problem using the following tools step by step:\n{tools}"
