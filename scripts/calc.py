import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict


#### plot performance vs. cost
seeds = 1

bins = np.array([float('-inf'), 0.0, 0.2, 0.4, 0.6, 0.8, float('inf')])
# datasets = ['healthbench_hard']
datasets = ['browsecomp', 'hle_text', 'deepsearchqa', 'gaia', 'healthbench_hard']
# datasets = ['browsecomp']

tag = 'v2_300'
models = [
    'o3',
    'react-o3-1',
    'react-o3-5',
    'react-o3-10',
    'search-o1-tool-o3-1',
    'search-o1-tool-o3-5',
    'search-o1-tool-o3-10',
    'search-o1-tool-o3-25',
    'search-o1-tool-o3-50',
    'search-o1-tool-o3-100',
    'hf-odr-o3',
    'gpt-researcher-o3',
    'slim-o3-10',
    'slim-o3-25',
    'slim-o3-50',
    'slim-o3-100',
    'slim-o3-150',

    'o4-mini',
    'react-o4-mini-1',
    'react-o4-mini-5',
    'react-o4-mini-10',
    'search-o1-tool-o4-mini-1',
    'search-o1-tool-o4-mini-5',
    'search-o1-tool-o4-mini-10',
    'search-o1-tool-o4-mini-25',
    'search-o1-tool-o4-mini-50',
    'hf-odr-o4-mini',
    'gpt-researcher-o4-mini',
    'slim-o4-mini-10',
    'slim-o4-mini-25',
    'slim-o4-mini-50',
    'slim-o4-mini-100',
    'slim-o4-mini-150',

    'claude-4-sonnet',
    'react-claude-4-sonnet-1',
    'react-claude-4-sonnet-5',
    'react-claude-4-sonnet-10',
    'search-o1-tool-claude-4-sonnet-1',
    'search-o1-tool-claude-4-sonnet-5',
    'search-o1-tool-claude-4-sonnet-10',
    'search-o1-tool-claude-4-sonnet-25',
    'search-o1-tool-claude-4-sonnet-50',
    'hf-odr-claude-4-sonnet',
    'gpt-researcher-claude-4-sonnet',
    'slim-claude-4-sonnet-10',
    'slim-claude-4-sonnet-25',
    'slim-claude-4-sonnet-50',
    'slim-claude-4-sonnet-100',
    'slim-claude-4-sonnet-150',

    'gpt-oss-120b',
    'search-o1-tool-gpt-oss-120b-50',
    'slim-orig-gpt-oss-120b-150',
    'react-gpt-oss-120b-10',

    'minimax-m2.5',
    'react-minimax-m2.5-10',
    'search-o1-tool-minimax-m2.5-50',
    'slim-orig-minimax-m2.5-150',

    'glm-4.7-flash',
    'react-glm-4.7-flash-10',
    'search-o1-tool-glm-4.7-flash-50',
    'slim-orig-glm-4.7-flash-150',

    'qwen3.5-122b',
    'react-qwen3.5-122b-10',
    'search-o1-tool-qwen3.5-122b-50',
    'slim-orig-qwen3.5-122b-150',

    'tongyi-deepresearch-30b-a3b',
    'react-tongyi-deepresearch-30b-a3b-10',
    'search-o1-tool-tongyi-deepresearch-30b-a3b-50',
    'slim-orig-tongyi-deepresearch-30b-a3b-150',
]

BASE_MODELS = ['o3', 'o4-mini', 'claude-4-sonnet', 'gpt-oss-120b', 'tongyi-deepresearch-30b-a3b', 'minimax-m2.5', 'glm-4.7-flash', 'qwen3.5-122b']

def get_base_model(model_name):
    for base in BASE_MODELS:
        if base in model_name:
            return base
    raise ValueError(f"Unknown base model for: {model_name}")

ms = []
for m in models:
    d = {'model': m, 'tag': tag, 'base_model': get_base_model(m)}
    if 'react' in m:
        d['model_class'] = 'react'
    elif 'search-o1-tool' in m:
        d['model_class'] = 'search-o1'
    elif 'slim' in m:
        d['model_class'] = 'slim'
    elif 'hf-odr' in m:
        d['model_class'] = 'hf-odr'
    elif 'gpt-researcher' in m:
        d['model_class'] = 'gpt-researcher'
    else:
        d['model_class'] = 'vanilla'
    ms.append(d)
    
print(ms)

path_template = "simple-evals/outputs/{model}/{dataset}_{model}_{tag}_{seed}_allresults.json"

# pre-defined contraints
# prices from openai, anthropic, togetherai, serper, and firecrawl standard packages respectively (unless otherwise specified)
def get_money_costs(base_model):
    if base_model == 'o3':
        money_costs = {'input_token': 2.0/1e6, 'output_token': 8.0/1e6,}
    elif base_model == 'o4-mini':
        money_costs = {'input_token': 1.1/1e6, 'output_token': 4.4/1e6,}
    elif base_model == 'claude-4-sonnet':
        money_costs = {'input_token': 3.0/1e6, 'output_token': 15.0/1e6,}
    elif base_model == 'gpt-oss-120b': 
        money_costs = {'input_token': 0.15/1e6, 'output_token': 0.6/1e6}
    elif base_model == 'tongyi-deepresearch-30b-a3b': # https://openrouter.ai/alibaba/tongyi-deepresearch-30b-a3b
        money_costs = {'input_token': 0.09/1e6, 'output_token': 0.45/1e6}
    elif base_model == 'minimax-m2.5':
        money_costs = {'input_token': 0.3/1e6, 'output_token': 1.2/1e6}
    elif base_model == 'glm-4.7-flash': # https://openrouter.ai/z-ai/glm-4.7-flash
        money_costs = {'input_token': 0.07/1e6, 'output_token': 0.4/1e6}
    elif base_model == 'qwen3.5-122b':
        money_costs = {'input_token': 0.4/1e6, 'output_token': 3.2/1e6}
    else:
        assert False
        
    money_costs.update({'search': 0.5/1e3, "browse": 83/1e5})
    return money_costs

def get_tokens(usages):
    total_tokens = 0
    
    if usages is None:
        # this happens when there is an error 
        return 0, 0, 0
        
    if isinstance(usages, dict):
        input_tokens = usages.get('input_tokens', usages.get('prompt_tokens'))
        output_tokens = usages.get('output_tokens', usages.get('completion_tokens'))
        total_tokens = input_tokens + 4*output_tokens
    else:
        input_tokens = 0
        output_tokens = 0
        for usage in usages:
            if "data" in usage:
                usage = usage['data']
            i = usage.get("input_tokens", usage.get('prompt_tokens'))
            o = usage.get("output_tokens", usage.get('completion_tokens'))
            if i is None or o is None:
                # this should only happen in the memory step for hf-odr
                continue
            if isinstance(i, str):
                i = int(i.replace(',', ''))
            if isinstance(o, str):
                o = int(o.replace(',', ''))
            input_tokens += i
            output_tokens += o
        total_tokens = input_tokens + 4*output_tokens
    return input_tokens, output_tokens, total_tokens

def get_tokens_caching(usages):
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for i, usage in enumerate(usages):
        if i > 0 and usage['input_tokens'] > usages[i-1]['total_tokens']:
            prev_reasoning = usages[i-1]['output_reasoning_tokens']
            prev_reasoning = prev_reasoning if prev_reasoning is not None else 0

            input_tokens += usage['input_tokens'] - (usages[i-1]['input_tokens'] + usages[i-1]['output_tokens'] - prev_reasoning) 
            output_tokens += usage['output_tokens']
            total_tokens += usage['input_tokens'] - (usages[i-1]['input_tokens'] + usages[i-1]['output_tokens'] - prev_reasoning) + 4*usage['output_tokens']
        else:
            input_tokens += usage['input_tokens']
            output_tokens += usage['output_tokens']
            total_tokens += usage['input_tokens'] + 4*usage['output_tokens']
    return input_tokens, output_tokens, total_tokens

def react_calc(conversation, metadata, money_costs):
    # we need to count the number of atomic tool calls
    counts = defaultdict(int)
    for c in conversation:
        if c['role'].startswith('tool_call'):
            counts[c['role'].split()[1]] += 1

    if 'error' in metadata['response_metadata']:
        # this means that there was an error
        input_tokens = output_tokens = total_tokens = 0
    elif "usage" in metadata and metadata['usage'] is not None:
        input_tokens, output_tokens, total_tokens = get_tokens_caching(metadata['usage'])
    else:
        input_tokens, output_tokens, total_tokens = get_tokens_caching(metadata['response_metadata']['usage'])
    
    money = money_costs['search'] * counts['search'] + money_costs['browse'] * counts['visit'] + input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token']
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_tools': sum(counts.values()), 'total_money': money, **dict(counts)}

def slim_calc(conversation, metadata, money_costs):
    counts = defaultdict(int)
    for c in conversation:
        if c['role'].startswith('tool_call'):
            tool_name = c['role'].split()[1]
            n = 1
            if tool_name == 'search' and isinstance(c['content'], dict):
                if isinstance(c['content'].get("query"), list):
                    n = len(c['content']['query'])
            elif tool_name == 'visit' and isinstance(c['content'], dict):
                if isinstance(c['content'].get("url"), list):
                    n = len(c['content']['url'])
            
            counts[tool_name] += n

    if 'error' in metadata['response_metadata']:
        # this means that there was an error
        input_tokens = output_tokens = total_tokens = 0
    else:
        if "summ_usages" in metadata['response_metadata']:
            input_tokens, output_tokens, total_tokens = get_tokens_caching(metadata['response_metadata']['agent_usages'])
            summ_input_tokens, summ_output_tokens, summ_total_tokens = get_tokens(metadata['response_metadata']['summ_usages'])
            total_tokens += summ_total_tokens
            input_tokens += summ_input_tokens
            output_tokens += summ_output_tokens
        else:
            input_tokens, output_tokens, total_tokens = get_tokens_caching(metadata['response_metadata']['usage'])
    
    # in theory, if you visit a page multiple time, only the first one should count towards the cost...
    money = money_costs['search'] * counts['search'] + money_costs['browse'] * counts['visit'] + input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token']
    # the time cost is a little tricky... 
    # time = metadata['response_metadata']['latency']
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_tools': sum(counts.values()), 'total_money': money, **dict(counts)}

def vanilla_calc(conversation, metadata, money_costs):
    if "usage" in metadata:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['usage'])
    else:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['response_metadata']['usage'])
    money = input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token']
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_money': money, 'total_tools': 0}

def search_o1_calc(converation, metadata, money_costs):
    # with prefix calcualtion
    if 'error' in metadata['response_metadata']:
        # this means that there was an error
        input_tokens = output_tokens = total_tokens = 0
        money = 0
        searches = browses = 0
    else:
        # generation can be cached but not the reasoning/summarization part
        gen_input_tokens, gen_output_tokens, gen_total_tokens = get_tokens_caching(metadata['response_metadata']['generation_usage'])
        reasoning_input_tokens, reasoning_output_tokens, reasoning_total_tokens = get_tokens(metadata['response_metadata']['reasoning_usage'])
        input_tokens = gen_input_tokens + reasoning_input_tokens
        output_tokens = gen_output_tokens + reasoning_output_tokens
        total_tokens = gen_total_tokens + reasoning_total_tokens
        search_results = metadata['response_metadata']['all_search_results']
        searches = len(search_results)
        browses = sum([len(s) for s in search_results])
        money = money_costs['search'] * searches + money_costs['browse'] * browses + input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token']
        
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_money': money, 'total_tools': searches + browses}

def gptr_calc(conversation, metadata, money_costs):
    if 'error' in metadata['response_metadata']:
        # this means that there was an error
        input_tokens = output_tokens = total_tokens = 0
    elif 'usage' in metadata:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['usage'])
    else:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['response_metadata']['usage'])

    searches = browses = 0
    
    for c in conversation:
        content = c['content']
        if "browsing the web to learn more about" in content.lower():
            searches += 1
        elif "scraping content from " in content.lower():
            n = int(content.strip().split()[-2])
            browses += n

    money = input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token'] + money_costs['search'] * searches + money_costs['browse'] * browses
    tools = browses + searches
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_money': money, 'total_tools': tools}

def hfodr_calc(conversation, metadata, money_costs):
    if 'error' in metadata['response_metadata']:
        # this means that there was an error
        input_tokens = output_tokens = total_tokens = 0
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_money': 0, 'total_tools': 0}
    elif 'usage' in metadata:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['usage'])
    else:
        input_tokens, output_tokens, total_tokens = get_tokens(metadata['response_metadata']['usage'])

    counts = defaultdict(int)
    for tool in metadata['response_metadata']['manager_metadata'].get('tool_calls', []) + metadata['response_metadata']['browser_metadata'].get('tool_calls', []):
        counts[tool['function']['name']] += 1

    # {'find_next', 'visit_page', 'find_on_page_ctrl_f', 'find_archived_url', 'python_interpreter', 'page_up', 'inspect_file_as_text', 'final_answer', 'page_down', 'web_search'}
    searches = counts['web_search']
    browses = counts['visit_page']
        
    money = input_tokens * money_costs['input_token'] + output_tokens * money_costs['output_token'] + money_costs['search'] * searches + money_costs['browse'] * browses
    tools = browses + searches + counts['python_interpreter'] + counts['find_archived_url']
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens, 'total_money': money, 'total_tools': tools}

def parse_turns(idx, model_dict, dataset, conversation, metadata, money_costs):
    model = model_dict['model']
    data = []

    d = {"dataset": dataset, "model": model}
    if "id" in metadata:
        d['qid'] = metadata["id"]
    else:
        d['qid'] = idx

    if model_dict['model_class'] in ['react']:
        d.update(react_calc(conversation, metadata, money_costs))
    elif model_dict['model_class'] == 'slim':
        d.update(slim_calc(conversation, metadata, money_costs))
    elif model_dict['model_class'] == 'vanilla':
        d.update(vanilla_calc(conversation, metadata, money_costs))
    elif model_dict['model_class'] == 'search-o1':
        d.update(search_o1_calc(conversation, metadata, money_costs))
    elif model_dict['model_class'] == 'gpt-researcher':
        d.update(gptr_calc(conversation, metadata, money_costs))
    elif model_dict['model_class'] == 'hf-odr':
        d.update(hfodr_calc(conversation, metadata, money_costs))
    else:
        assert False, "should have a calculation function"
        
    if 'healthbench' in dataset:
        s = metadata['score']
        idx = np.digitize(np.array([s]), bins)[0]
        # d['score'] = f"[{bins[idx-1]}, {bins[idx]})"
        d['score'] = s * 100
    elif 'deepsearchqa' in dataset:
        d['score'] = metadata['all_correct'] * 100
    elif 'gaia' in dataset:
        d['score'] = 0 if metadata['correctness'] == "Incorrect" else 100
    else:
        d['score'] = 100 if metadata['correctness'] == "yes" else 0

    data.append(d)
        
    return data

df = []

for model_dict in ms:
    for dataset in datasets:
        for seed in range(seeds):
            print(model_dict, dataset)
            model_class = model_dict["model_class"]
            model = model_dict['model']
            money_costs = get_money_costs(model_dict['base_model'])
            tool_names = set()
            path = path_template.format(model=model, dataset=dataset, tag=model_dict['tag'], seed=seed)
            if not os.path.exists(path):
                print(f"Skipping {path} (not found)")
                continue
            with open(path) as f:
                data = json.load(f)

            temp = []
            for idx, (convo, metadata) in enumerate(zip(data['convos'], data['metadata']['example_level_metadata'])):
                # healthbench needs separate handling for correctness...
                dat = parse_turns(idx, model_dict, dataset, convo, metadata, money_costs)
                temp += dat
            # take the mean of the score and token cost
            # print([x['score'] for x in temp])
            df.append({
                "model": model, "model_class": model_class, "dataset": dataset, 
                "tokens": np.mean([x['total_tokens'] for x in temp]), 
                "cost": np.mean([x['total_money'] for x in temp]), 
                "tools": np.mean([x['total_tools'] for x in temp]), 
                "score": np.mean([x['score'] for x in temp]),
                "seed": seed,
            })

df = pd.DataFrame(df)
print(df.to_csv(index=False))

# pivot table: rows = model, columns = (dataset, metric), averaged over seed
pivot = pd.pivot_table(df, index='model', columns='dataset', values=['tokens', 'cost', 'tools', 'score'], aggfunc='mean', sort=False)
pivot = pivot.swaplevel(axis=1)
pivot = pivot.reindex(columns=pd.MultiIndex.from_product([datasets, ['tokens', 'cost', 'tools', 'score']]))
print("\n=== Pivot Table ===")
print(pivot.to_csv())