import random 
import numpy as np
import os 
from datetime import date, datetime
import sys
import asyncio
from async_timeout import timeout
import time
from autogen import AssistantAgent
import tiktoken
import signal 
import time 
import config
from functools import partial

class Logger(object):
    def __init__(self, log_path, on=False):
        self.log_path = log_path
        self.on = on
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path = self.log_path[:-4] + '+'+ self.log_path[-4:]
        with open(self.log_path, 'w') as f:
            f.write("")

    def log(self, string, newline=True):
        #if self.on:
        with open(self.log_path, "a") as logf:
            today = date.today()
            today_date = today.strftime("%m/%d/%Y")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            string = today_date + ", " + current_time + ": " + string
            logf.write(string)
            if newline:
                logf.write("\n")

        sys.stdout.write(string)
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()

async def cancel(task):
    start = time.time()
    task.cancel()
    try:
        async with timeout(-1):
            await task
        end = time.time()
    except asyncio.CancelledError:
        end = time.time()
        return 'task cancelled'
    except asyncio.exceptions.TimeoutError:
        end = time.time()
        return 'time out for canceling task'
    except Exception as e:
        end = time.time()
        return 'exception:' + str(e)


class SignalHandler:
    def __init__(self):
        pass

    # # old register handler, doesn't know why it doesn't work :(
    # def exit_handler(self, signum, frame, target_tasks):
    #     to_halt_target_task = [target_task for target_task in target_tasks if target_task.get_name() == f'target_{config.HIL_INTERACTION}' and not target_task.done() and not target_task.cancelled()]
    #     if to_halt_target_task != []:
    #         config.USERINPUT=True
    #         to_halt_target_task[0].cancel()
        
    async def exit_handler(self, target_tasks):
        to_halt_target_task = [target_task for target_task in target_tasks if target_task.get_name() == f'target_{config.HIL_INTERACTION}' and not target_task.done() and not target_task.cancelled()]
        if to_halt_target_task != []:
            config.USERINPUT=True
            to_halt_target_task[0].cancel()
            async with timeout(-1):
                await to_halt_target_task[0]

def register_async_handler(target_tasks):
    loop = asyncio.get_event_loop()
    s = SignalHandler()
    #global sigterm_handler
    loop.add_signal_handler(getattr(signal, 'SIGINT'), lambda: asyncio.create_task(s.exit_handler(target_tasks=target_tasks)))


def exit_handler(signum, frame, target_tasks):
    to_halt_target_task = [target_task for target_task in target_tasks if target_task.get_name() == f'target_{config.HIL_INTERACTION}' and not target_task.done() and not target_task.cancelled()]
    if to_halt_target_task != []:
        config.USERINPUT=True
        to_halt_target_task[0].cancel()

# # old register handler, doesn't know why it doesn't work :(
# def register_handler(target_tasks):
#     s = SignalHandler()
#     signal.signal(signal.SIGINT, partial(s.exit_handler, target_tasks=target_tasks))

def register_handler(target_tasks):
    signal.signal(signal.SIGINT, partial(exit_handler, target_tasks=target_tasks))
