import argparse
from datetime import datetime
import gc
import json
import os
import random
import requests
import shutil
import signal
import sys
import time

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

# Make local Safe-RLHF package importable for workers if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'safe-rlhf'))
from datasets import load_dataset

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4  # 4 engines like countdown
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/alignment/es-ft-experiment"

# Minimal local prompt formatting to avoid importing Safe-RLHF on drivers/workers
PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'

def format_prompt_local(user_input: str, eos_token: str) -> str:
    # We only need a single-turn user prompt followed by ASSISTANT:
    return ''.join([PROMPT_BEGIN, PROMPT_USER.format(input=user_input), PROMPT_ASSISTANT])

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Alignment (PKU objective) with multi-engine NCCL sync and external scorer API"
    )
    parser.add_argument("--policy_model_path", type=str, default="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of tokens allowed to be generated')
    
    # Scorer API configuration
    parser.add_argument("--scorer_url", type=str, default="http://localhost:8000", help="URL of the scorer API service")
    
    # Dataset and training args
    parser.add_argument('--lambda_cost', type=float, default=0.5)
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--eval_samples', type=int, default=500)
    parser.add_argument('--train_jsonl', type=str, default='alignment/data/custom_train_250.jsonl')
    parser.add_argument('--eval_jsonl', type=str, default='alignment/data/custom_eval_500.jsonl')
    parser.add_argument('--batch_size', type=int, default=200, help='Number of prompts to sample per iteration for ES fitness')
    parser.add_argument('--standardize_within_batch', action='store_true', default=True,
                        help='Standardize reward and cost within mini-batch before combining')
    parser.add_argument('--safe_first', action='store_true', default=False,
                        help='Apply safe-first gating: zero or downweight reward when cost>0')
    parser.add_argument('--safe_first_factor', type=float, default=0.0,
                        help='Multiplier for reward when cost>0 (0.0 => hard gate)')
    parser.add_argument('--lambda_adapt', action='store_true', default=False,
                        help='Enable Lagrangian lambda adaptation')
    parser.add_argument('--lambda_lr', type=float, default=0.01, help='Learning rate for ln(lambda) update')
    parser.add_argument('--lambda_min', type=float, default=1e-4)
    parser.add_argument('--lambda_max', type=float, default=10.0)
    parser.add_argument('--lambda_pos_cost_only', action='store_true', default=True,
                        help='Use mean(max(C,0)) as J_C for lambda update')
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--wandb_project', type=str, default='alignment_accl')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--scorer_batch_size', type=int, default=16)
    parser.add_argument('--save_every', type=int, default=100, help='Save HF-format latest checkpoint every N iters (0=off)')
    parser.add_argument("--global_seed", type=int, help="Global random seed")
    
    args = parser.parse_args()
    
    # Optional: scope host visibility; vLLM actors will ignore it and pick device from PG
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args

class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        # Let Ray/PG determine the actual visible device in the actor
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name):
    # Strict 1-GPU isolation via PGs (exactly like countdown)
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs

def call_scorer_api(scorer_url, texts, batch_size=16):
    """Call the external scorer API to get rewards and costs."""
    try:
        # For alignment task, texts are already full (prompt + response)
        # We need to extract prompts and responses for the API
        # The format is: BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}
        prompts = []
        responses = []
        for text in texts:
            # Split on ASSISTANT: to separate prompt and response
            parts = text.split('ASSISTANT:')
            if len(parts) >= 2:
                # Everything before ASSISTANT: is the prompt part
                prompt = parts[0] + 'ASSISTANT:'
                # Everything after is the response
                response = ''.join(parts[1:])
                prompts.append(prompt)
                responses.append(response)
            else:
                # Fallback if format is unexpected
                prompts.append(text)
                responses.append("")
        
        # Call the API in batches if needed
        all_rewards = []
        all_costs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            
            payload = {
                "prompts": batch_prompts,
                "responses": batch_responses
            }
            
            # Use the batch endpoint for efficiency
            response = requests.post(
                f"{scorer_url}/score_batch",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            all_rewards.extend(result["rewards"])
            all_costs.extend(result["costs"])
        
        return {
            "per_sample_reward": all_rewards,
            "per_sample_cost": all_costs,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "mean_cost": float(np.mean(all_costs)) if all_costs else 0.0,
            "safe_count": sum(1 for c in all_costs if c <= 0),
            "num_samples": len(all_costs),
            "safe_ratio": sum(1 for c in all_costs if c <= 0) / len(all_costs) if all_costs else 0.0,
        }
    except requests.exceptions.RequestException as e:
        print(f"Error calling scorer API: {e}")
        # Return zeros if the API call fails
        return {
            "per_sample_reward": [0.0] * len(texts),
            "per_sample_cost": [0.0] * len(texts),
            "mean_reward": 0.0,
            "mean_cost": 0.0,
            "safe_count": 0,
            "num_samples": len(texts),
            "safe_ratio": 0.0,
        }

def evaluate_alignment_handle(llm, prompts, max_tokens):
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        seed=42,
        max_tokens=max_tokens,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()

def build_full_texts(prompts, outputs):
    full_texts = []
    for p, out in zip(prompts, outputs):
        response = out.outputs[0].text
        full_texts.append(p + response)
    return full_texts

def run_final_evaluation(llm, eval_prompts, scorer_url, args):
    print(f"\nRunning final evaluation on {len(eval_prompts)} samples...")

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=args.max_new_tokens,
    )
    outputs = ray.get(llm.generate.remote(eval_prompts, sampling_params, use_tqdm=True))

    full_texts = build_full_texts(eval_prompts, outputs)
    eval_scores = call_scorer_api(scorer_url, full_texts, batch_size=args.scorer_batch_size)

    rewards = np.asarray(eval_scores["per_sample_reward"], dtype=np.float32)
    costs = np.asarray(eval_scores["per_sample_cost"], dtype=np.float32)

    # Safe-first and optional standardization (to mirror training combine)
    if args.safe_first:
        r_eff = np.where(costs > 0.0, args.safe_first_factor * rewards, rewards)
    else:
        r_eff = rewards
    if args.standardize_within_batch:
        r_mean, r_std = float(r_eff.mean()), float(r_eff.std()) + 1e-8
        c_mean, c_std = float(costs.mean()), float(costs.std()) + 1e-8
        r_tilde = (r_eff - r_mean) / r_std
        c_tilde = (costs - c_mean) / c_std
    else:
        r_tilde = r_eff
        c_tilde = costs
    combined = r_tilde - args.lambda_cost * c_tilde

    per_sample_results = []
    for r, c, comb in zip(rewards.tolist(), costs.tolist(), combined.tolist()):
        per_sample_results.append({
            "reward": float(r),
            "cost": float(c),
            "combined": float(comb),
        })

    return {
        "mean_reward": float(rewards.mean()) if rewards.size > 0 else 0.0,
        "mean_cost": float(costs.mean()) if costs.size > 0 else 0.0,
        "mean_combined": float(combined.mean()) if combined.size > 0 else 0.0,
        "total_samples": len(full_texts),
        "per_sample_results": per_sample_results,
    }

def main(args):
    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Verify GPU availability (driver-level view)
    resources = ray.cluster_resources()
    gpu_count = resources.get('GPU', 0)
    required_gpus = args.num_engines  # Only engines need GPUs (scorer is external)

    print("=" * 80)
    print(f"Ray Cluster Resources:")
    print(f"  GPUs: {gpu_count}")
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  Memory: {resources.get('memory', 0) / (1024**3):.2f} GB")
    print(f"  Required: {required_gpus} GPUs for {args.num_engines} engines")
    print(f"  Scorer API: {args.scorer_url}")
    if gpu_count < required_gpus:
        print(f"\n❌ ERROR: Need {required_gpus} GPUs but only {gpu_count} available!")
        print(f"   Adjust --num_engines or ensure sufficient GPUs are allocated.")
        ray.shutdown()
        sys.exit(1)
    print(f"✅ Found {gpu_count} GPUs - sufficient for training")
    print("=" * 80)

    # Check scorer API availability
    try:
        response = requests.get(f"{args.scorer_url}/health", timeout=5)
        response.raise_for_status()
        print(f"✅ Scorer API is available at {args.scorer_url}")
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Scorer API at {args.scorer_url} is not responding: {e}")
        print("Make sure the scorer service is running before continuing.")
        ray.shutdown()
        sys.exit(1)

    # Logging
    logging_dir = f"{args.experiment_dir}/alignment_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)
    if args.wandb_project and wandb is not None:
        default_name = args.wandb_run_name or f"accl_api_{os.path.basename(args.policy_model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=default_name, config={
            'policy_model_path': args.policy_model_path,
            'scorer_url': args.scorer_url,
            'sigma': args.sigma,
            'alpha': args.alpha,
            'population_size': args.population_size,
            'num_engines': args.num_engines,
            'lambda_cost_init': args.lambda_cost,
            'lambda_adapt': args.lambda_adapt,
            'safe_first': args.safe_first,
            'standardize_within_batch': args.standardize_within_batch,
            'batch_size': args.batch_size,
            'scorer_batch_size': args.scorer_batch_size,
            'eval_every': args.eval_every,
        })

    # Prepare an HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # Resolve base model path: use provided local path if directory exists; otherwise, fetch and save
    if os.path.isdir(args.policy_model_path):
        base_model_path = args.policy_model_path
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.policy_model_path, torch_dtype=torch.float16
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path)
        base_model_path = f"{model_saves_dir}/base_model"
        if os.path.exists(base_model_path):
            shutil.rmtree(base_model_path)
        os.makedirs(base_model_path, exist_ok=True)
        tokenizer.save_pretrained(base_model_path)
        base_model.save_pretrained(base_model_path)
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load policy tokenizer for prompt formatting
    policy_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    def save_hf_latest_from_engine(engine, iteration: int):
        """Save current engine weights into HF-format 'latest' directory."""
        tmp_path = f"{model_saves_dir}/tmp_iter_{iteration}.pth"
        # dump raw state_dict from engine to tmp
        ray.get(engine.collective_rpc.remote("save_self_weights_to_disk", args=(tmp_path,)))
        # load base model on CPU and load state dict
        mdl = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to("cpu")
        state = torch.load(tmp_path, map_location="cpu")
        mdl.load_state_dict(state, strict=True)
        # write to latest_hf
        latest_dir = f"{model_saves_dir}/latest_hf"
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        os.makedirs(latest_dir, exist_ok=True)
        policy_tokenizer.save_pretrained(latest_dir)
        mdl.save_pretrained(latest_dir)
        with open(os.path.join(latest_dir, "ckpt_meta.json"), "w") as f:
            json.dump({"iteration": iteration, "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')}, f)
        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        del mdl, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load prompts from custom JSONL if available; otherwise fall back to PKU dataset
    def load_prompts_from_jsonl(path: str, n: int):
        try:
            prompts_raw = []
            with open(path, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    if 'prompt' in obj:
                        prompts_raw.append(obj['prompt'])
            if not prompts_raw:
                return None
            rng = random.Random(args.global_seed or 42)
            if n < len(prompts_raw):
                prompts_raw = rng.sample(prompts_raw, n)
            prompts = []
            for user_prompt in prompts_raw:
                conv = format_prompt_local(user_prompt, policy_tokenizer.eos_token or '</s>')
                if not conv.endswith(PROMPT_ASSISTANT):
                    conv = conv + PROMPT_ASSISTANT
                prompts.append(conv)
            return prompts
        except Exception:
            return None

    def sample_prompts(split: str, n: int, jsonl_path: str | None = None):
        if jsonl_path and os.path.exists(jsonl_path):
            loaded = load_prompts_from_jsonl(jsonl_path, n)
            if loaded is not None:
                return loaded
        ds = load_dataset('PKU-Alignment/PKU-SafeRLHF', split=split)
        total = len(ds)
        indices = list(range(total))
        if n < total:
            rng = random.Random(args.global_seed or 42)
            indices = rng.sample(indices, n)
        prompts = []
        for idx in indices[:n]:
            raw = ds[int(idx)]
            user_prompt = raw['prompt']
            conv = format_prompt_local(user_prompt, policy_tokenizer.eos_token or '</s>')
            if not conv.endswith(PROMPT_ASSISTANT):
                conv = conv + PROMPT_ASSISTANT
            prompts.append(conv)
        return prompts

    train_prompts = sample_prompts('train', args.train_samples, args.train_jsonl)
    eval_prompts = sample_prompts('test', args.eval_samples, args.eval_jsonl)

    # Launch engines (exactly like countdown)
    print(f"\nLaunching {args.num_engines} vLLM engines with placement groups...")
    engines, pgs = launch_engines(args.num_engines, base_model_path)
    print(f"✅ Submitted {len(engines)} engines")

    # Initialize lambda for Lagrangian adaptation
    current_lambda = float(args.lambda_cost)

    # Init inter-engine communicator once
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Training loop (exactly like original but with API calls)
    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []
        # For logging distributions
        all_rewards_samples = []
        all_cost_samples = []
        total_safe_count = 0
        total_sampled = 0

        # Choose a deterministic subset of prompts for this generation
        bs = min(args.batch_size, len(train_prompts))
        rng = random.Random((args.global_seed or 42) * 100000 + i)
        subset_indices = rng.sample(range(len(train_prompts)), bs) if bs < len(train_prompts) else list(range(len(train_prompts)))
        curr_prompts = [train_prompts[idx] for idx in subset_indices]

        # Kick off an eval on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Add exploration noise
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, start_ts = evaluate_alignment_handle(llm, curr_prompts, args.max_new_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            full_texts = build_full_texts(curr_prompts, outputs)
            
            # Call scorer API instead of Ray actor
            metrics = call_scorer_api(args.scorer_url, full_texts, batch_size=args.scorer_batch_size)
            elapsed = time.time() - meta["start_ts"]

            # Combine with optional safe-first and within-batch standardization
            rewards_np = np.asarray(metrics["per_sample_reward"], dtype=np.float32)
            costs_np = np.asarray(metrics["per_sample_cost"], dtype=np.float32)
            all_rewards_samples.extend(rewards_np.tolist())
            all_cost_samples.extend(costs_np.tolist())
            total_safe_count += int(np.sum(costs_np <= 0.0))
            total_sampled += int(costs_np.size)

            if args.safe_first:
                r_eff = np.where(costs_np > 0.0, args.safe_first_factor * rewards_np, rewards_np)
            else:
                r_eff = rewards_np

            if args.standardize_within_batch:
                r_mean, r_std = float(r_eff.mean()), float(r_eff.std()) + 1e-8
                c_mean, c_std = float(costs_np.mean()), float(costs_np.std()) + 1e-8
                r_tilde = (r_eff - r_mean) / r_std
                c_tilde = (costs_np - c_mean) / c_std
            else:
                r_tilde = r_eff
                c_tilde = costs_np

            combined_arr = r_tilde - current_lambda * c_tilde
            mean_combined = float(combined_arr.mean()) if combined_arr.size > 0 else 0.0

            seeds_perf[meta["seed"]] = {
                **metrics,
                "mean_combined": mean_combined,
            }
            results_this_gen.append({"seed": meta["seed"], "avg_combined": mean_combined, "time": elapsed})

            llm = meta["engine"]
            # Remove exploration noise
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            handle, start_ts = evaluate_alignment_handle(llm, curr_prompts, args.max_new_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            if args.verbose:
                print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Optional Lagrangian lambda adaptation using baseline (unperturbed) policy
        if args.lambda_adapt:
            base_handle, _ = evaluate_alignment_handle(engines[0], curr_prompts, args.max_new_tokens)
            base_outputs = ray.get(base_handle)
            base_texts = build_full_texts(curr_prompts, base_outputs)
            base_scores = call_scorer_api(args.scorer_url, base_texts, batch_size=args.scorer_batch_size)
            base_costs = np.asarray(base_scores["per_sample_cost"], dtype=np.float32)
            jc = float(np.mean(np.maximum(base_costs, 0.0))) if args.lambda_pos_cost_only else float(np.mean(base_costs))
            ln_lambda = float(np.log(max(current_lambda, 1e-12)))
            ln_lambda += args.lambda_lr * current_lambda * jc
            current_lambda = float(np.clip(np.exp(ln_lambda), args.lambda_min, args.lambda_max))
            writer.add_scalar("lambda/J_C", jc, i)
            writer.add_scalar("lambda/value", current_lambda, i)

        # Normalize combined objective
        all_means = [v["mean_combined"] for v in seeds_perf.values()]
        mean_combined = float(np.mean(all_means)) if all_means else 0.0
        std_combined = float(np.std(all_means)) if all_means else 0.0
        min_combined = float(np.min(all_means)) if all_means else 0.0
        max_combined = float(np.max(all_means)) if all_means else 0.0

        print(f"Mean combined: {mean_combined}, std: {std_combined}, min: {min_combined}, max: {max_combined}")
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["mean_combined"] - mean_combined) / (std_combined + 1e-8)
            if args.verbose:
                print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("combined/mean", mean_combined, i)
        writer.add_scalar("combined/std", std_combined, i)
        writer.add_scalar("combined/min", min_combined, i)
        writer.add_scalar("combined/max", max_combined, i)
        writer.add_scalar("helpfulness/mean", float(np.mean([v["mean_reward"] for v in seeds_perf.values()])), i)
        writer.add_scalar("cost/mean", float(np.mean([v["mean_cost"] for v in seeds_perf.values()])), i)
        if total_sampled > 0:
            writer.add_scalar("safety/safe_ratio", float(total_safe_count) / float(total_sampled), i)
        try:
            writer.add_histogram("helpfulness/dist", np.asarray(all_rewards_samples, dtype=np.float32), i)
            writer.add_histogram("cost/dist", np.asarray(all_cost_samples, dtype=np.float32), i)
        except Exception:
            pass
        if args.wandb_project and wandb is not None:
            log_payload = {
                'combined/mean': mean_combined,
                'combined/std': std_combined,
                'combined/min': min_combined,
                'combined/max': max_combined,
                'helpfulness/mean': float(np.mean([v["mean_reward"] for v in seeds_perf.values()])),
                'cost/mean': float(np.mean([v["mean_cost"] for v in seeds_perf.values()])),
            }
            if total_sampled > 0:
                log_payload['safety/safe_ratio'] = float(total_safe_count) / float(total_sampled)
            try:
                log_payload['helpfulness/dist'] = wandb.Histogram(np.asarray(all_rewards_samples, dtype=np.float32))
                log_payload['cost/dist'] = wandb.Histogram(np.asarray(all_cost_samples, dtype=np.float32))
            except Exception:
                pass
            wandb.log(log_payload, step=i)

        # ES update on engine 0 (exactly like countdown)
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        perturb_start = time.time()
        handles = []
        for seed, coeff in per_seed_coeffs:
            handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))
        ray.get(handles)
        if args.verbose:
            print(f"Applied perturbations in {time.time() - perturb_start}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, i)

        # Broadcast updated weights from engine 0 to all engines (avoid CPU copies)
        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        if args.verbose:
            print(f"Broadcasted updated weights in {time.time() - broadcast_start}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, i)

        total_iter_end = time.time()
        writer.add_scalar("time/iteration", total_iter_end - total_iter_start, i)
        print(f"wall clock time for iteration {i}: {total_iter_end - total_iter_start}s")
        print(f"=== Generation {i} finished ===\n")

        # Periodic evaluation during training
        if args.eval_every > 0 and ((i + 1) % args.eval_every == 0):
            eval_results_i = run_final_evaluation(engines[0], eval_prompts, args.scorer_url, args)
            writer.add_scalar("eval/mean_reward", eval_results_i["mean_reward"], i)
            writer.add_scalar("eval/mean_cost", eval_results_i["mean_cost"], i)
            writer.add_scalar("eval/mean_combined", eval_results_i["mean_combined"], i)
            if args.wandb_project and wandb is not None:
                wandb.log({
                    'eval/mean_reward': eval_results_i['mean_reward'],
                    'eval/mean_cost': eval_results_i['mean_cost'],
                    'eval/mean_combined': eval_results_i['mean_combined'],
                    'eval/total_samples': eval_results_i['total_samples'],
                }, step=i)

        # Periodic HF-format checkpoint (latest only)
        if args.save_every > 0 and ((i + 1) % args.save_every == 0):
            try:
                save_hf_latest_from_engine(engines[0], i + 1)
                if args.wandb_project and wandb is not None:
                    wandb.log({'checkpoint/iteration': i + 1}, step=i)
            except Exception as e:
                print(f"Warning: failed to save HF latest checkpoint at iter {i+1}: {e}")

    # FINAL EVAL
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    eval_results = run_final_evaluation(engines[0], eval_prompts, args.scorer_url, args)

    print(f"\nMean Reward: {eval_results['mean_reward']:.4f}")
    print(f"Mean Cost: {eval_results['mean_cost']:.4f}")
    print(f"Mean Combined: {eval_results['mean_combined']:.4f}")
    print("="*80 + "\n")
    if args.wandb_project and wandb is not None:
        wandb.log({
            'eval/mean_reward': eval_results['mean_reward'],
            'eval/mean_cost': eval_results['mean_cost'],
            'eval/mean_combined': eval_results['mean_combined'],
            'eval/total_samples': eval_results['total_samples'],
        })

    # Save evaluation results to JSON
    logs_dir = f"{logging_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)

    eval_output = {
        "metadata": {
            "policy_model_path": args.policy_model_path,
            "num_iterations": args.num_iterations,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "scorer_url": args.scorer_url,
            "lambda_cost": args.lambda_cost,
        },
        "metrics": {
            "mean_reward": eval_results["mean_reward"],
            "mean_cost": eval_results["mean_cost"],
            "mean_combined": eval_results["mean_combined"],
            "total_samples": eval_results["total_samples"],
        },
        "per_sample_results": eval_results["per_sample_results"],
    }

    eval_json_path = f"{logs_dir}/final_eval_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"Evaluation results saved to {eval_json_path}\n")

    # Save final HF-format checkpoint to 'latest_hf'
    try:
        save_hf_latest_from_engine(engines[0], args.num_iterations)
        print("Final HF-format checkpoint saved to latest_hf.")
    except Exception as e:
        print(f"Warning: failed to save final HF-format checkpoint: {e}")

    # Save final model weights (all engines are in sync; save from engine 0)
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",)
        )
    )
    print(f"Final model weights saved to {final_model_path}.")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    main(args)