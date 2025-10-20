import argparse
from datetime import datetime
import gc
import json
import os
import random
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
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown.countdown_task import reward_function

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "es-ft-experiment"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task with vLLM"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--sigma", type=float, default=SIGMA, help="Exploration noise scale"
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="ES learning rate")
    parser.add_argument(
        "--population_size", type=int, default=POPULATION_SIZE, help="Population size"
    )
    parser.add_argument(
        "--num_engines",
        type=int,
        default=NUM_ENGINES,
        help="Number of vLLM engines (GPUs)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=NUM_ITERATIONS,
        help="Number of ES iterations",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=EXPERIMENT_DIR,
        help="Directory to save experiment logs and models",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES setting (optional)",
    )

    # set cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().cuda_devices

    return parser.parse_args()


class ESWrappedLLM(LLM):
    """vLLM LLM with determinism knobs for Ray execution."""

    def __init__(self, *args, **kwargs):
        # Let Ray/PG assign the visible GPU to each actor
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines=4, model_name="Qwen/Qwen2.5-3B-Instruct"):
    # One placement group (1 GPU) per engine for strict isolation
    pgs = [
        placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        for _ in range(num_engines)
    ]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    # Create one vLLM actor per GPU/PG
    engines = [
        ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=strategy,
        )(ESWrappedLLM).remote(
            model=model_name,
            enforce_eager=False,
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            dtype="float16",
            enable_prefix_caching=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


def evaluate_countdown_handle(llm, task_datas):
    """Return a generation handle so we can schedule round-robin."""
    start = time.time()
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1024,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, start


def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
    }


def main(args):
    # We don't try to connect to an external Ray cluster
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Logging
    logging_dir = f"{args.experiment_dir}/countdown_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare an initial HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load data
    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    # Start persistent engines
    engines, pgs = launch_engines(
        num_engines=args.num_engines, model_name=base_model_path
    )

    # Helper to cleanly shutdown at the end
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

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} ===")
        total_iter_start = time.time()

        # Snapshot the current baseline on each engine for this iteration
        ray.get(
            [llm.collective_rpc.remote("save_self_initial_weights") for llm in engines]
        )

        # Random seeds for population
        seeds = [random.randint(0, 1000000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduler
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off one job per engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Exploration perturbation (coeff=1.0)
            ray.get(
                llm.collective_rpc.remote(
                    "perturb_self_weights", args=(seed, args.sigma, 1.0, False)
                )
            )
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        # Process jobs; when an engine is free, schedule the next seed
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {
                    "seed": meta["seed"],
                    "avg_reward": metrics["avg_reward"],
                    "time": elapsed,
                }
            )

            llm = meta["engine"]
            # Remove the exploration perturbation
            ray.get(
                llm.collective_rpc.remote(
                    "restore_self_weights", args=(meta["seed"], args.sigma)
                )
            )

            # Schedule next job on this engine if any remain
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(
                llm.collective_rpc.remote(
                    "perturb_self_weights", args=(next_seed, args.sigma, 1.0, False)
                )
            )
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Normalize rewards
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        print(
            f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}"
        )
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (
                seeds_perf[k]["avg_reward"] - mean_reward
            ) / (std_reward + 1e-8)
            print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        # Log generation statistics
        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)

        # Bring all engines to the exact same baseline
        ray.get(
            [
                llm.collective_rpc.remote("restore_self_initial_weights")
                for llm in engines
            ]
        )

        # Seed-by-seed ES update, broadcasted to all engines in the same order
        # coeff = (ALPHA / POPULATION_SIZE) * norm_reward
        per_seed_coeffs = [
            (
                seed,
                (args.alpha / args.population_size)
                * float(seeds_perf[seed]["norm_reward"]),
            )
            for seed in seeds
        ]
        for seed, coeff in per_seed_coeffs:
            # Use sigma_or_scale=1.0 so scale = coeff
            ray.get(
                [
                    llm.collective_rpc.remote(
                        "perturb_self_weights", args=(seed, 1.0, coeff, False)
                    )
                    for llm in engines
                ]
            )

        # Re-snapshot the new baseline for the next iteration
        ray.get(
            [llm.collective_rpc.remote("save_self_initial_weights") for llm in engines]
        )

        # End of iteration logging
        for res_idx, res in enumerate(results_this_gen):
            print(
                f"IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s"
            )
        total_iter_end = time.time()
        writer.add_scalar("time/iteration", total_iter_end - total_iter_start, i)
        print(f"=== Generation {i} finished ===\n")

    # save the model weights after training
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    # Save model weights from the first engine (all engines are synchronized)
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
