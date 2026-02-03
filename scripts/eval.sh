models=(
    #react-url-o3-100
    # drreact-kimi-k2-instruct-100
    #o4-mini
    #gpt-researcher-o4-mini

    #search-o1-tool-claude-4-sonnet-10
    #search-o1-tool-claude-4-sonnet-50

    #gpt-researcher-o3
    #gpt-researcher-o4-mini
    #gpt-researcher-claude-4-sonnet ##still need to run this but let's put it off for later
    #drreact-summ-o3-150-50
    #drreact-summ-o3-200-50

    hf-odr-o3
    hf-odr-o4-mini
    hf-odr-claude-4-sonnet

    #search-o1-tool-o3-25
    # search-o1-tool-o4-mini-10
    #search-o1-tool-claude-4-sonnet-25

    #react-o3-10-k5-10k
    #react-o3-10-k10-3k
    #react-o3-10-k5-3k
    #slim-o3-99-50
    #slim-o3-150-50

    #slim-o4-mini-150-50
    #slim-o4-mini-100
    #slim-o4-mini-50
    #slim-o4-mini-25
    #slim-o4-mini-10

    #slim-o3-token-100-32k
    #slim-o3-token-100-64k
    #search-o1-tool-o4-mini-100

    #slim-o3-150
    #gpt-oss-120b

    #slim-gpt-oss-120b-150
    #search-o1-tool-gpt-oss-120b-10
    #search-o1-tool-gpt-oss-120b-50

    #tongyi-deepresearch-30b-a3b
    ##slim-tongyi-deepresearch-30b-a3b-150
    #search-o1-tool-tongyi-deepresearch-30b-a3b-50

    #slim-o3-200

    #slim-o4-mini-100
    #o3
    #slim-o3-100
    #slim-o3-100-25
    #slim-o3-token-100-32k
    #slim-o3-token-100-64k
    #slim-o3-100-newline-bm25
    #slim-o3-100-words_100-rouge
    #slim-o3-100-words_100-bm25
    #slim-o3-100-k10-3k
    #slim-o3-100-k10-20k
    #slim-o3-100-k20-3k
    #slim-o3-100-k20-10k
    #slim-o3-100-k20-20k
)
n=3
seed=0
n=300

for model in "${models[@]}"; do
    # no debug, final runs
    for seed in {0..0}; do
        #python -m simple-evals.simple_evals --eval browsecomp,hle_text --model $model --output-dir simple-evals/outputs/${model} --n-threads 8 --tag "v2_${n}_${seed}" --examples $n --model_seed $seed --checkpoint-interval 5
        uv run python -m simple-evals.simple_evals --eval deepsearchqa --model $model --output-dir simple-evals/outputs/${model} --n-threads 2 --tag "v2_${n}_${seed}" --examples $n --model_seed $seed --checkpoint-interval 5
        echo ''
    done
done

#python simple-evals/scripts/collect_results.py \
    #--models gpt-4.1,gpt-4.1-web-search,claude-4-sonnet,claude-4-sonnet-web-search,react-web-claude-4-sonnet,o3,react-url-o3,react-url-o3-100,o4-mini,jf-odr,gpt-researcher,react-o4-mini,react-web-o4-mini,react-url-o4-mini,qwen2.5-7b,react-web-qwen2.5-7b,search-r1-qwen2.5-7b,search-r1-qwen2.5-7b-em-ppo,qwen2.5-7b-it,react-web-qwen2.5-7b-it,search-r1-qwen2.5-7b-it,search-r1-qwen2.5-7b-it-em-ppo,qwen3-8b,react-web-qwen3-8b,search-r1-qwen3-8b,hosted_vllm-qwen3-32b,react_vllm-qwen3-32b \
    #--evals hle_text,browsecomp,healthbench_hard \
    #--output-dir simple-evals/outputs --tag v1_${n} --output-csv simple-evals/outputs/leaderboard.csv --seeds 3

