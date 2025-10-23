import os
import random
import json
import threading
import re
import csv
import pandas as pd

import litellm
from litellm import completion
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Literal
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import argparse

litellm.enable_json_schema_validation = True

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
path = args.path

print(path)

def get_search_queries(convo, metadata):
    prompt = ""
    if "search-o1-tool" in path:
        for c in convo:
            if "tool_call" in c['role']:
                prompt += f"Search query: {c['content']}\n\n"
    
    elif "/react-" in path:
        for c in convo:
            if "tool_call" in c['role']:
                prompt += f"Search query: {c['content']}\n\n"
    
    elif "hf-odr" in path:
        for m in metadata['response_metadata']['browser_metadata']['tool_calls']:
            if m['function']['name'] == 'web_search':
                prompt += f"Search query: {m['function']['arguments']['query']}\n\n"
    
    elif "gpt-researcher" in path:
        for c in convo:
            if "browsing the web to learn more about" in c['content'].lower():
                query = c['content'].split("the web to learn more about the task:")[-1].strip('...')
                prompt += f"Search query: {query}\n\n"
    
    elif 'slim' in path or 'drreact' in path:
        for c in convo:
            if "tool_call search" in c['role']:
                query = json.loads(c['content'])['query']
                prompt += f"Search query: {query}\n\n"

    else:
        assert False, "Unknown model"

    return prompt
     
def completion_wrapper(messages, response_format):
    trial = 0
    while trial < 3:
        try: 
            response = completion(
                model='azure/o3',
                messages=messages,
                max_tokens=30000,
                response_format=response_format,
            )
            return json.loads(response['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error: {e}")
            trial += 1
    return None


CONFIRMATION_BIAS_SYSTEM_MESSAGE = """You are a helpful assistant that can analyze the trajectory of an information-seeking agent. You are given a question-answer pair and the search history of an agent that tried to answer the question. You should analyze the search history and determine if the agent spends more than half of the tool calls searching for the same incorrect answer. That is, the agent continues searching for the same topic even though it's not the correct answer to the question, and spends half or more of its tool calls on these searches. Output your final conclusion with your reasoning and a single word: 'yes' if the agent spends more than half of its tool calls on the same incorrect answer or 'no' if the agent does not.

Reasoning: explain what the agent did, and if it did or did not focus its searches on a wrong answer.

Conclusion: "yes" or "no"."""

class ConfirmationBias(BaseModel):
    reasoning: str
    conclusion: Literal["yes", "no"]
    strict: Literal[True]

def confirmation_bias_analysis(convo, question, answer, metadata):
    prompt = get_search_queries(convo, metadata)
    prompt += f"\n\nQuestion: {question}"
    prompt += f"\nCorrect Answer: {answer}"
    messages = [
        {"role": "system", "content": CONFIRMATION_BIAS_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    return completion_wrapper(messages, ConfirmationBias)


UNFOCUSED_SEARCH_SYSTEM_MESSAGE = """You are a helpful assistant that can analyze the trajectory of an information-seeking agent. You are given a question-answer pair and the search history of an agent that tried to answer the question. You should analyze the search history and determine if the search queries do not help the agent narrow down the search space. Consider the following cases:
1. The agent searches for information relevant to the question and answer, but it's not specific enough to yield helpful results. 
2. The agent searches for queries that are not sufficiently relevant or specific to the question and answer, which does not narrow down the search space enough.
3. The agent explores the search space with diverse queries but does not use enough tool calls to properly narrow down the search space by either eliminating wrong answers or verifying the correct answer.
All of these cases are considered to be unfocused search. You should consider the whole trajectory of the agent, and not just some of the tool calls---only consider the trajectory to be unfocused if more than half of the searches are unfocused.

Output your final conclusion with your reasoning and a single word: 'yes' if the searches are unfocused or 'no' if the searches are focused enough.

Reasoning: explain what the agent did, and if it did or did not use tool calls to properly narrow down the search space.

Conclusion: "yes" or "no"."""

class UnfocusedSearch(BaseModel):
    reasoning: str
    conclusion: Literal["yes", "no"]
    strict: Literal[True]

def unfocused_search_analysis(convo, question, answer, metadata):
    prompt = get_search_queries(convo, metadata)
    prompt += f"\n\nQuestion: {question}"
    prompt += f"\nCorrect Answer: {answer}"
    messages = [
        {"role": "system", "content": UNFOCUSED_SEARCH_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    return completion_wrapper(messages, UnfocusedSearch)


def inefficient_tool_analysis(convo, question, answer, metadata):
    # here we simply check if the agent does not find new information due to the lack of new urls
    # we consider a model to be inefficient with its tool calls if half or more of its searches calls do not yield any new urls.
    found_urls = set()
    if "search-o1-tool" in path:
        search_results = metadata['response_metadata']['all_search_results']
        urls = [[x['url'] for x in y] for y in search_results]
        num_tools = len(urls)
        count = 0
        for url_list in urls:
            if all([url in found_urls for url in url_list]):
                count += 1
            for url in url_list:
                found_urls.add(url)
        
    elif "/react-" in path:
        num_tools = 0
        count = 0
        for c in convo:
            if c['role'] == 'tool':
                num_tools += 1
                pattern = r"<URL (\d+): (.*)>"
                matches = re.findall(pattern, c['content'])
                urls = [match[1] for match in matches]

                # the other pattern is when there is an error
                pattern = r"<URL (.*)>\n<Error"
                matches = re.findall(pattern, c['content'])
                urls += [match for match in matches]

                if all([url in found_urls for url in urls]):
                    count += 1
                for url in urls:
                    found_urls.add(url)
    
    elif "hf-odr" in path:
        num_tools = 0
        count = 0
        for m in metadata['response_metadata']['browser_metadata']['tool_calls']:
            if m['function']['name'] == 'web_search':                    
                num_tools += 1

        for c in convo:
            if c['role'] == 'tool-response':
                for m in c['content'].split('\n---\n'):
                    pattern = r"(\d+). \[(.*)\]\((.*)\)"
                    matches = re.findall(pattern, m)
                    urls = [match[2] for match in matches]
                    if all([url in found_urls for url in urls]):
                        count += 1
                    for url in urls:
                        found_urls.add(url)
    
    elif 'slim' in path or 'drreact' in path:
        num_tools = 0
        count = 0
        for c in convo:
            if c['role'] == 'tool' and c['content'].startswith("The search engine returned"):
                num_tools += 1
                pattern = r'<URL: (.*)>'
                matches = re.findall(pattern, c['content'])
                urls = [match for match in matches]
                if all([url in found_urls for url in urls]):
                    count += 1
                for url in urls:
                    found_urls.add(url)
    
    else:
        assert False, "Unknown model"

        
    if count >= num_tools / 2:
        return {"conclusion": "yes", "count": count, "total": num_tools}
    else:
        return {"conclusion": "no", "count": count, "total": num_tools}

def get_search_o1_content(results):
    prompt = ""
    for i, result in enumerate(results):
        prompt += json.dumps({
            'id': i + 1,
            'title': result.get('title', ''),
            'url': result.get('link', ''),
            'site_name': result.get('source', ''),
            'date': result.get('date', ''),
            'snippet': result.get('snippet', ''),
            'context': result.get('context', '')
        })
    return prompt


def get_hfodr_content(content):
    prompt = ""
    summary = content.split('<summary_of_work>')[-1].split('</summary_of_work>')[0].strip()
    for m in summary.split('\n---\n'):
        if m.startswith("""[{'type': 'text', 'text': "Observation:\\n##"""):
            prompt += m
    return prompt

ANSWER_FOUND_SYSTEM_MESSAGE = """You are a helpful assistant that can analyze the trajectory of an information-seeking agent. You are given a question-answer pair and a list of webpages. You should analyze the web contents and determine if it contains the correct answer. The correct answer is considered to be found if there are some context in the search results that is either a direct or near-exact match to the correct answer. Output your final conclusion with your reasoning and a single word: 'yes' if the content contains the correct answer or 'no' if the content does not contain the correct answer.

Reasoning: explain if the web content contains the correct answer.

Conclusion: "yes" or "no"."""
class AnswerFound(BaseModel):
    reasoning: str
    conclusion: Literal["yes", "no"]
    strict: Literal[True]

def answer_found_analysis(convo, question, answer, metadata):
    # here if we check if the agent actually encountered the correct answer somewhere in its trajectory but for some reason did not use it.
    if "search-o1-tool" in path:
        search_results = metadata['response_metadata']['all_search_results']
        for results in tqdm(search_results, leave=False, desc="Answer Found Analysis"):
            prompt = "Webpages:\n\n"
            prompt += get_search_o1_content(results)
            prompt += f"\n\nQuestion: {question}"
            prompt += f"\nCorrect Answer: {answer}"
            messages = [
                {"role": "system", "content": ANSWER_FOUND_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            result = completion_wrapper(messages, AnswerFound)
            if result is not None and result['conclusion'] == "yes":
                # early stop if there is correct answer
                return result
    
    elif "/react-" in path:
        for c in convo:
            if c['role'] == 'tool':
                prompt = c['content']
                prompt += f"\n\nQuestion: {question}"
                prompt += f"\nCorrect Answer: {answer}"

                messages = [
                    {"role": "system", "content": ANSWER_FOUND_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, AnswerFound)
                if result is not None and result['conclusion'] == "yes":
                    return result
                
    elif "hf-odr" in path:
        for c in convo:
            if c['role'] == 'tool-response' and "your managed agent 'search_agent'" in c['content']:
                prompt = get_hfodr_content(c['content'])
                prompt += f"\n\nQuestion: {question}"
                prompt += f"\nCorrect Answer: {answer}"
                messages = [
                    {"role": "system", "content": ANSWER_FOUND_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, AnswerFound)
                if result is not None and result['conclusion'] == "yes":
                    return result
    
    elif 'slim' in path or 'drreact' in path:
        for c in convo:
            if c['role'] == 'tool':
                prompt = c['content']
                prompt += f"\n\nQuestion: {question}"
                prompt += f"\nCorrect Answer: {answer}"
                messages = [
                    {"role": "system", "content": ANSWER_FOUND_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, AnswerFound)
                if result is not None and result['conclusion'] == "yes":
                    return result
    
    return {"conclusion": "no"}



HALLUCINATION_SYSTEM_MESSAGE = """You are a helpful assistant that can analyze the trajectory of an information-seeking agent. You are given a list of webpages and a list of claims made by the agent. You should analyze the web contents to determine if each claim is supported by the web content.
A claim is supported by the web content if its factual information is mostly supported by the web content, and is not contradicted by the web content. Output your final conclusion with a list of claims that are supported by the web content. Output the list in the form of a json list, and you only need to write the index of the supported claims in the list and nothing else.

Output the index of the supported claims in the form of a json list. Use 0-based indexing."""

ATOMIC_CLAIMS_SYSTEM_MESSAGE = """Read the given explanation and generate a list of atomic claims that are supported by the explanation. Atomic claims that are basic facts that cannot be further broken down. Generate at most 10 claims for the explanation. If there are more than 10 possible claims, focus on the most important claims made about the final answer.

Use the following as an example:

Explanation:  \nSearching UFCStats for featherweight bouts where the loser landed 14 of 83 significant strikes (16.87 %) and went 0-for-4 on takedowns returns the fight Myles Jury vs. Ricky Glenn at UFC 219: Cyborg vs Holm (30 Dec 2017).  \n• Ricky Glenn (nickname “The Gladiator”—a synonym for swordsman) was the loser: sig. strikes 14/83 (16.87 %), takedowns 0/4.  \n• Both fighters (Jury 29, Glenn 28) were under 35 and are American.  \n• The referee was John McCarthy, whose first event for the UFC was in 1994.  \nThus, the MMA event is UFC 219: Cyborg vs Holm.\n\nExact Answer: UFC 219: Cyborg vs Holm\n\nConfidence: 75%

Atomic Claims:
- Ricky Glenn was the loser
- Ricky Glenn was nicknamed "The Gladiator"
- The sig. strike rate of Ricky Glenn was 14/83 (16.87%)
- The takedown rate of Ricky Glenn was 0/4
- Jury was age 29
- Glenn was age 28
- Jury is American
- Glenn is American
- The referee was John McCarthy
- John McCarthy's first event for the UFC was in 1994

Output the atomic claims in the form of a json list."""

class AtomicClaims(BaseModel):
    atomic_claims: List[str]
    strict: Literal[True]

def get_atomic_claims(response_text):
    messages = [
        {"role": "system", "content": ATOMIC_CLAIMS_SYSTEM_MESSAGE},
        {"role": "user", "content": response_text},
    ]
    response = completion_wrapper(messages, AtomicClaims)
    if response is None:
        return []
    return response['atomic_claims']

class Hallucination(BaseModel):
    supported_claims: List[int]
    strict: Literal[True]

def hallucination_analysis(convo, question, answer, metadata):
    # this is a bit more complicated. essentially we adopt the evaluation from ALCE/HELMET where we check for attribution
    # we first break the final answer into its atomic claims, then we iterate through the search results and check if each claim is support by the search results
    response_text = metadata['response_text']
    atomic_claims = get_atomic_claims(response_text)
    if len(atomic_claims) == 0:
        return None
    supported_claims = [False for _ in atomic_claims]

    if "search-o1-tool" in path:
        search_results = metadata['response_metadata']['all_search_results']
        for results in tqdm(search_results, leave=False, desc="Hallucination Analysis"):
            prompt = "Webpages:\n\n"
            prompt += get_search_o1_content(results)
            prompt += f"\n\nAtomic Claims:"
            prompt += '\n'.join([f"{idx}: {claim}" for idx, claim in enumerate(atomic_claims)])
            messages = [
                {"role": "system", "content": HALLUCINATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            result = completion_wrapper(messages, Hallucination)
            if result is None:
                continue
            supports = result['supported_claims']
            for claim in supports:
                supported_claims[claim] = True
            if all(supported_claims):
                return {'atomic_claims': atomic_claims, 'supported_claims': supported_claims, "conclusion": "yes"}
    
    elif "/react-" in path:
        for c in convo:
            if c['role'] == 'tool':
                prompt = c['content']
                prompt += f"\n\nAtomic Claims:"
                prompt += '\n'.join([f"{idx}: {claim}" for idx, claim in enumerate(atomic_claims)])
                
                messages = [
                    {"role": "system", "content": HALLUCINATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, Hallucination)
                if result is None:
                    continue
                supports = result['supported_claims']
                for claim in supports:
                    supported_claims[claim] = True
                if all(supported_claims):
                    return {'atomic_claims': atomic_claims, 'supported_claims': supported_claims, "conclusion": "yes"}
    
    elif "hf-odr" in path:
        for c in convo:
            if c['role'] == 'tool-response' and "your managed agent 'search_agent'" in c['content']:
                prompt = get_hfodr_content(c['content'])
                prompt += f"\n\nAtomic Claims:"
                prompt += '\n'.join([f"{idx}: {claim}" for idx, claim in enumerate(atomic_claims)])
                messages = [
                    {"role": "system", "content": HALLUCINATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, Hallucination)
                if result is None:
                    continue
                supports = result['supported_claims']
                for claim in supports:
                    supported_claims[claim] = True
                if all(supported_claims):
                    return {'atomic_claims': atomic_claims, 'supported_claims': supported_claims, "conclusion": "yes"}
    
    elif 'slim' in path or 'drreact' in path:
        for c in convo:
            if c['role'] == 'tool':
                prompt = c['content']
                prompt += f"\n\nAtomic Claims:"
                prompt += '\n'.join([f"{idx}: {claim}" for idx, claim in enumerate(atomic_claims)])
                messages = [
                    {"role": "system", "content": HALLUCINATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
                result = completion_wrapper(messages, Hallucination)
                if result is None:
                    continue
                supports = result['supported_claims']
                for claim in supports:
                    supported_claims[claim] = True
                if all(supported_claims):
                    return {'atomic_claims': atomic_claims, 'supported_claims': supported_claims, "conclusion": "yes"}

    else:
        assert False, "Unknown model"

    return {'atomic_claims': atomic_claims, "supported_claims": supported_claims, "conclusion": "no"}


GIVE_UP_SYSTEM_PROMPT = """You are a helpful assistant that can analyze the final output of an information-seeking agent. You are to check if the agent decides that it cannot find the correct answer. For example, if the explanation states that it cannot find enough relevant information to answer the question, or if the response is simply empty or "I don't know", then the agent did not attempt to answer the question. Output your final conclusion with a single word "yes" if the agent decides it did not find enough information to answer the question or "no" otherwise.

Conclusion: "yes" or "no"."""

class GiveUp(BaseModel):
    conclusion: Literal["yes", "no"]
    strict: Literal[True]

def give_up_analysis(convo, question, answer, metadata):
    response_text = metadata['response_text']
    messages = [
        {"role": "system", "content": GIVE_UP_SYSTEM_PROMPT},
        {"role": "user", "content": response_text},
    ]
    result = completion_wrapper(messages, GiveUp)
    if result is None:
        return None
    return result


with open(path, 'r') as f:
    data = json.load(f)
       

def determine_outcome(convo, metadata):
    if "search-o1-tool" in path:
        budget = int(path.split("/")[1].split('-')[-1])
        if 'usage' in metadata and metadata.get('usage') is None:
            return "error"
        elif "ContentPolicyViolationError" in metadata['response_text']:
            return "error"
        elif metadata['response_metadata']['generation_usage'][-1]['input_tokens'] < metadata['response_metadata']['generation_usage'][-2]['input_tokens']:
            return "exceed_context"
        elif metadata['response_metadata']['iterations'] >= budget:
            return "exceed_budget"
        elif metadata['response_metadata']['iterations'] == 0:
            return "no_tools"
        elif metadata['response_metadata']['iterations'] < budget:
            return "early_stop"
        else:
            assert False, "Unknown outcome"
    
    elif "/react-" in path:
        budget = int(path.split("/")[1].split('-')[-1])
        num_tools = len([c for c in convo if c['role'] == 'tool'])
        if 'usage' in metadata and metadata.get('usage') is None:
            return 'error'
        elif "ContentPolicyViolationError" in metadata['response_text']:
            return "error"
        elif metadata['response_metadata']['usage'][-1]['input_tokens'] < metadata['response_metadata']['usage'][-2]['input_tokens']:
            return "exceed_context"
        elif num_tools >= budget:
            return "exceed_budget"
        elif num_tools == 0:
            return "no_tools"
        elif num_tools < budget:
            return "early_stop"
        else:
            assert False, "Unknown outcome"

    elif "hf-odr" in path:
        num_tools = len(metadata['response_metadata']['manager_metadata']['tool_calls'])
        budget = 20 # default for hf odr
        if 'usage' in metadata and metadata.get('usage') is None:
            return 'error'
        elif "ContentPolicyViolationError" in metadata['response_text']:
            return "error"
            # note that hf odr cannot run out of context window, at least for the manager agent
        elif num_tools >= budget:
            return "exceed_budget"
        elif num_tools == 0:
            return "no_tools"
        elif num_tools < budget:
            return "early_stop"
        else:
            assert False, "Unknown outcome"

    elif "gpt-researcher" in path:
        # note that gpt-researcher does not exceed context window and does not have a budget
        if 'usage' in metadata and metadata.get('usage') is None:
            return 'error'
        elif "ContentPolicyViolationError" in metadata['response_text']:
            return "error"
        else:
            return "early_stop"

    elif 'slim' in path or 'drreact' in path:
        # budget is the first int after a -
        budget = path.split("/")[1]
        pattern = r'-(\d+)'
        matches = re.findall(pattern, budget)
        budget = int(matches[0])

        num_tools = len([c for c in convo if c['role'] == 'tool'])
        if metadata['response_metadata'].get("error") is not None:
            if "ContextWindowExceededError" in metadata['response_metadata']['error']:
                return "exceed_context"
            else:
                return "error"
        elif num_tools >= budget:
            return "exceed_budget"
        elif num_tools == 0:
            return "no_tools"
        elif num_tools < budget:
            return "early_stop"
        else:
            assert False, "Unknown outcome"

    else:
        assert False, "Unknown model"
            

# take a random sample for humna analysis
rng = random.Random(0)
idxs = rng.sample(range(len(data['convos'])), 15)
to_check_outputs = []
for idx in idxs:
    convo = data['convos'][idx]
    metadata = data['metadata']['example_level_metadata'][idx]
    output_data = {
        "idx": idx,
        "question": metadata['question'],
        "answer": metadata['answer'],
        "correctness": metadata['correctness'],
        "response_text": metadata['response_text'],
        "queries": get_search_queries(convo, metadata),
    }
    contents = []

    if "search-o1-tool" in path:
        for results in tqdm(metadata['response_metadata']['all_search_results'], leave=False, desc="Answer Found Analysis"):
            contents.append(get_search_o1_content(results))
    elif "/react-" in path:
        for c in convo:
            if c['role'] == 'tool':
                contents.append(c['content'])
    elif "hf-odr" in path:
        for c in convo:
            if c['role'] == 'tool-response' and "your managed agent 'search_agent'" in c['content']:
                contents.append(get_hfodr_content(c['content']))
    elif 'slim' in path or 'drreact' in path:
        for c in convo:
            if c['role'] == 'tool':
                contents.append(c['content'])

    output_data['contents'] = "\n\n".join(contents)
    to_check_outputs.append(output_data)
    # import pdb; pdb.set_trace()

human_path = 'human_analysis/' + path.split('/')[-1].replace('.json', '.csv')
df = pd.DataFrame(to_check_outputs)
df.to_csv(human_path, index=False)

# with open(human_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     for output_data in to_check_outputs:
#         writer.writerow(output_data.values())
    
# import pdb; pdb.set_trace()
    
output_file = path.replace('.json', '_analyzed.json')
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        all_outputs = json.load(f)
    all_outputs.pop("idx", "")
    print(f"Loaded {len(all_outputs)} samples")
else:
    all_outputs = {}

print(f"Starting analysis with {len(data['convos'])} samples")
file_lock = threading.Lock()

def process_sample(idx, convo, metadata):
    correct_answer = metadata['answer']
    question = metadata['question']
    output_data = {
        "idx": idx,
    }
    if metadata['correctness'] == 'yes':
        output_data['outcome'] = 'correct'
        return idx, output_data

    # we need to determine what the outcome is: ran out of context window, ran out of tool budget, or early stop
    output_data['outcome'] = determine_outcome(convo, metadata)
    if output_data['outcome'] == 'error':
        return idx, output_data
    
    output_data['confirmation_bias'] = confirmation_bias_analysis(convo, question, correct_answer, metadata)
    output_data['unfocused_search'] = unfocused_search_analysis(convo, question, correct_answer, metadata)
    # look at the urls
    if "gpt-researcher" not in path:
        output_data['inefficient_tool'] = inefficient_tool_analysis(convo, question, correct_answer, metadata)

    if output_data['outcome'] != "exceed_context":
        # only look at the final output 
        output_data['give_up'] = give_up_analysis(convo, question, correct_answer, metadata)
        if "gpt-researcher" not in path:
            # look at the final output and all content
            output_data['answer_ignored'] = answer_found_analysis(convo, question, correct_answer, metadata)
            if output_data['give_up'] is not None and output_data['give_up']['conclusion'] == "no":
                # look at the final output and all content
                output_data['hallucination'] = hallucination_analysis(convo, question, correct_answer, metadata)
    
    return idx, output_data

def process_sample_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    idx, output_data = process_sample(*args)
    with file_lock:
        all_outputs[str(idx)] = output_data
        with open(output_file, 'w') as f:
            json.dump(all_outputs, f, indent=4)
    return idx, output_data

args_list = list(zip(range(len(data['convos'])), data['convos'], data['metadata']['example_level_metadata']))
args_list = [a for a in args_list if str(a[0]) not in all_outputs]
print(f"Processing {len(args_list)} samples")

with ThreadPool(min(8, len(data['convos']))) as pool:
    for result in tqdm(pool.imap(process_sample_wrapper, args_list), total=len(args_list), desc="Processing samples"):
        idx, output_data = result

with open(output_file, 'w') as f:
    json.dump(all_outputs, f, indent=4)

