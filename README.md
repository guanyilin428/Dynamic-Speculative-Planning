# Dynamic Speculative Agent Planning

## Abstract
Dynamic speculative planning aims to enhance decision-making in time-sensitive tasks by predicting and optimizing future outcomes in real-time. Unlike traditional planning approaches that rely on fixed parameters, dynamic speculative planning adapts by continuously adjusting its predictions based on incoming data. This method employs online learning and incremental speculative step prediction to reduce computational costs and time delays, allowing systems to plan and execute actions more efficiently. By utilizing advanced techniques such as expectile regression and cost-latency preference bias, dynamic speculative planning improves both time savings and token usage in complex, dynamic environments. This repository provides tools and algorithms designed to implement dynamic speculative planning strategies, offering insights into their effectiveness across various benchmarks and tasks.

## Experiment & Command
We provide two environments for two separate experiments. Please follow instructions accordingly.


### OpenAGI Experiment
The OpenAGI setting uses the agent to generate plan first and then do the execution. Thus here, we focus on the planning step without execution.

To set up the environment:
```
conda create -n specplan python=3.10
conda activate specplan
pip install -r requirements.txt
```

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
model_type = "gpt-4.1-mini" # could be "gpt-4.1-mini" or "deepseek"
python -m OpenAGI.openagi_dyn --pred --target_type target_type --approx_type approx_type --offset offset --tau tau --model_type model_type
```

### TravelPlanner Experiment
The TravelPlanner mainly adopts the code from [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) and integrate the dynamic speculative planning code into it.

To run speculative planning on TravelPlanner, you need to first download code and database following instructions in [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) to download data. A different virtual environment to fit TravelPlanner is also necessary.
```
git clone https://github.com/OSU-NLP-Group/TravelPlanner

conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
pip install -r TravelPlanner/requirements.txt
```

Then put the async_online_utils.py and sp_travel_planner_dyn.py under travelplanner_supplement and predictor.py and util.py under Dynamic-Speculative-Planning in downloaded TravelPlanner's code under the directory TravelPlanner. Put tool_agents_sp.py under travelplanner_supplement in TravelPlanner/agents directory.

In tool_agents_sp.py and sp_travel_planner_dyn.py, set the OPENAI_API_KEY and DEEPSEEK_API_KEY:
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
model_type = "gpt-4.1-mini" # could be "gpt-4.1-mini", "deepseek-chat"
python sp_travel_planner_dyn.py --pred --target_type target_type --approx_type approx_type --offset offset --tau tau --model_type model_type
```