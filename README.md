# Dynamic Speculative Agent Planning
Dynamic Speculative Planning (DSP) is an lightweight online reinforcement learning framework for accelerating LLM-based agents. This repository hosts the code and data for this paper: [Dynamic Speculative Agent Planning](https://arxiv.org/abs/2509.01920)

## Table of Contents
- [Experiment & Command](#experiment--command)
  - [OpenAGI Experiment](#openagi-experiment)
  - [TravelPlanner Experiment](#travelplanner-experiment)
- [Citation](#citation)

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

We use the following shorthand:  
- `Direct` = direct-generation  
- `CoT` = chain-of-thought  
- `MAD` = multi-agent-debate  

| **Setting** | **Approximation Agent** | **Target Agent** |
|-------------|--------------------------|------------------|
| **1** | Direct (GPT-4.1-mini) | ReAct (GPT-4.1-mini) |
| **2** | CoT (GPT-4.1-mini) | MAD (GPT-4.1-mini) |
| **3** | Direct (deepseek-chat) | ReAct (deepseek-reasoner) |
| **4** | CoT (deepseek-chat) | MAD (deepseek-reasoner) |

You may also configure your own approximation–target combinations or plug in other APIs if interested.

- Fix Mode:
```
python -m OpenAGI.runner --no-pred --k 2 # choose fix k value
```

- Dynamic Mode:
```
approx_type = "direct" # could be "direct" (setting 1 & 3), "cot" (setting 2 & 4)
target_type = "react" # could be "react" (setting 1 & 3), "multi_agent" (setting 2 & 4)
offset = 0 # choose inference offset for k
tau = 0.5 # choose asymmetric hyperparameter for expectile regression
model_type = "gpt-4.1-mini" # could be "gpt-4.1-mini" or "deepseek"
python -m OpenAGI.runner --pred --target_type target_type --approx_type approx_type --offset offset --tau tau --model_type model_type
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

Put tool_agents_sp.py from travelplanner_supplement/ into TravelPlanner/agents/.
Then, put other files from travelplanner_supplement/ and predictor.py, util.py from Dynamic-Speculative-Planning/ into the TravelPlanner/ root directory.

In tool_agents_sp.py and runner.py, set the OPENAI_API_KEY and DEEPSEEK_API_KEY:
```
os.environ['OPENAI_API_KEY'] = your_gpt_key
os.environ['DEEPSEEK_API_KEY'] = your_dpsk_key
```

To run the experiment:
- Fix Mode
```
cd TravelPlanner
python runner.py --no-pred --k 2 # choose fix k value
```

- Dynamic Mode
```
cd TravelPlanner
approx_type = "direct" # could be "direct" (setting 1 & 3), "cot" (setting 2 & 4)
target_type = "react" # could be "react" (setting 1 & 3), "multi_agent" (setting 2 & 4)
offset = 0 # choose inference offset for k
tau = 0.5 # choose asymmetric hyperparameter for expectile regression
model_type = "gpt-4.1-mini" # could be "gpt-4.1-mini", "deepseek-chat"
python runner.py --pred --target_type target_type --approx_type approx_type --offset offset --tau tau --model_type model_type
```

## Citation

If you have any further questions, please feel free to contact us. And if you find our work helpful, please cite our paper:

```bibtex
@article{guan2025dynamic,
  title={Dynamic Speculative Agent Planning},
  author={Guan, Yilin and Hua, Wenyue and Lan, Qingfeng and Fei, Sun and Ding, Dujian and Acharya, Devang and Wang, Chi and Wang, William Yang},
  journal={arXiv preprint arXiv:2509.01920},
  year={2025}
}
```
