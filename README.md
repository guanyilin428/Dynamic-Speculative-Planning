# Dynamic Speculative Agent Planning


## Abstract
Agents are increasingly employed as user-centric tools for human task delegation, assisting with a wide range of requests by generating thoughts, interacting with user proxies, and producing final action plans. However, agents based on large language models (LLMs) often encounter significant planning latency due to two main factors: the efficiency constraints of the underlying LLMs, exacerbated by their large size and high demand, and the structural complexity of the agents, which necessitates extensive intermediate step generation to produce the final output. This paper introduces a human-centered efficient agent planning method -- interactive speculative planning -- to enhance the efficiency of agent planning while incorporating human interaction to further accelerate the system. Our approach promotes the co-design of the agent system and user interface, emphasizing the importance of an agent system that can seamlessly handle user interactions and interruptions. 
By treating human interruptions as an integral part of the system, we not only make it more user-centric but also accelerate the entire system by providing correct intermediate steps.


## Experiment & Command

### OpenAGI Experiment
The OpenAGI setting uses the agent to generate plan first and then do the execution. Thus here, we focus on the planning step without execution.

In openagi_dyn.py, set the OPENAI_API_KEY and DEEPSEEK_API_KEY:
```
os.environ['OPENAI_API_KEY'] = your_gpt_key
os.environ['DEEPSEEK_API_KEY'] = your_dpsk_key
```

There are four setting that we employs in our experiment:

Setting 1: The approximation agent uses direct-generation-based planning with a GPT-4.1-mini backbone, and the target agent uses ReAct-based planning with a GPT-4.1-mini backbone. 

Setting 2: The approximation agent uses chain-of-thought (CoT)-based planning with a GPT-4.1-mini backbone, and the target agent uses multi-agent-debate (MAD)-based planning with a GPT-4.1-mini backbone. 

Setting 3: The approximation agent uses direct-generation-based planning with a deepseek-chat backbone, and the target agent uses ReAct-based planning with a deepseek-reasoner backbone.

Setting 4: The approximation agent uses chain-of-thought (CoT)-based planning with a deepseek-chat backbone, and the target agent uses multi-agent-debate (MAD)-based planning with a deepseek-reasoner backbone.

- Fix Mode:
```
python -m OpenAGI.openagi_dyn --no-pred --k 2 # choose fix k value
```

- Dynamic Mode:
```
approx_type = "direct" # could be "direct" (setting 1 & 3), "cot" (setting 2 & 4)
target_type = "react" # could be "react" (setting 1 & 3), "multi_agent" (setting 2 & 4)
offset = 0 # choose inference offset for k
tau = 0.5 # choose asymmetric hyperparameter for expectile regression
python openagi_dyn.py --pred --target_type target_type --approx_type approx_type --offset offset --tau tau
```

### TravelPlanner Experiment
The TravelPlanner mainly adopts the code from [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) and integrate the interactive speculative planning code into it.

To run speculative planning on TravelPlanner, you need to first download code and database following instructions in [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) to download data. A different virtual environment to fit TravelPlanner is also necessary.
```
git clone https://github.com/OSU-NLP-Group/TravelPlanner

conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
```

Then put the threes python files under travelplanner_supplement in downloaded TravelPlanner's code under the directory TravelPlanner. 

In tool_agents_sp.py, set the OPENAI_API_KEY and DEEPSEEK_API_KEY:
```
os.environ['OPENAI_API_KEY'] = your_gpt_key
os.environ['DEEPSEEK_API_KEY'] = your_dpsk_key
```

To run the experiment:
- Fix Mode
```
cd TravelPlanner
python sp_travel_planner_dyn.py --no-pred --k 2 # choose fix k value
```

- Dynamic Mode
```
cd TravelPlanner
approx_type = "direct" # could be "direct" (setting 1 & 3), "cot" (setting 2 & 4)
target_type = "react" # could be "react" (setting 1 & 3), "multi_agent" (setting 2 & 4)
offset = 0 # choose inference offset for k
tau = 0.5 # choose asymmetric hyperparameter for expectile regression
python sp_travel_planner_dyn.py --pred --target_type target_type --approx_type approx_type --offset offset --tau tau
```