import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ['NCCL_TIMEOUT'] = '86400'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ["NCCL_P2P_DISABLE"] = "1"   # Disable NCCL point-to-point support

import argparse
import re
import gc
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import datetime


def parse_output_without_sentence(item_id_pattern, summarization_pattern, reasoning_pattern, generated_text):
    item_id = re.search(item_id_pattern, generated_text)
    summarization = re.search(summarization_pattern, generated_text)
    reasoning = re.search(reasoning_pattern, generated_text)

    sections = {
        "item_id": item_id.group(1) if item_id else None,
        "summarization": summarization.group(1) if summarization else None,
        "reasoning": reasoning.group(1) if reasoning else None
    }

    return sections


def main_worker(gpu, world_size, args):
    try:
        # Set the current device.
        torch.cuda.set_device(gpu)

        timeout = datetime.timedelta(seconds=360000)
        # Initialize the process group for distributed communication.
        dist.init_process_group(
            backend='nccl',
            init_method=args.init_method,  # e.g., "tcp://127.0.0.1:23456"
            world_size=world_size,
            rank=gpu,
            timeout=timeout
        )

        # Load system prompt and item prompts.
        with open(args.item_system_prompt_path, 'r') as f:
            system_prompt = f.read()

        with open(args.fashion_item_prompt_path, 'rb') as f:
            item_prompt_dict = pickle.load(f)

        # Prepare the model and tokenizer on the current GPU.
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to the EOS token
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
        # processor.pad_token_id = processor.eos_token_id  # Set pad_token_id to the EOS token
        # processor.padding_side = 'left'
        # processor.pad_token = processor.eos_token
        # processor.pad_token_id = processor.convert_tokens_to_ids(processor.eos_token)

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            torch_dtype=torch.bfloat16
        ).to(gpu)

        # model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        # model.resize_token_embeddings(len(tokenizer))

        model = DDP(model, device_ids=[gpu])

        # Distribute messages across GPUs by simple round-robin assignment.
        inputs = []
        for idx, (key, prompt) in enumerate(item_prompt_dict.items()):
            if idx % world_size == gpu:
                input_prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt['prompt']}
                ]
                chat_format = tokenizer.apply_chat_template(
                    input_prompt, tokenize=False, add_generation_prompt=True
                )
                inputs.append(chat_format)

        item_id_pattern = r'"item_id"\s*:\s*"([^"]+)"'
        summarization_pattern = r'"summarization"\s*:\s*"([^"]+|None)"'
        reasoning_pattern = r'"reasoning"\s*:\s*"([^"]+)"'

        results = {}
        step = 0

        # Process messages in batches.
        for i in tqdm(range(0, len(inputs), args.batch_size), desc=f"GPU {gpu} processing batches"):
            if i + args.batch_size >= len(inputs):
                batch = inputs[i:]
            else:
                batch = inputs[i: i + args.batch_size]

            inputs_batch = tokenizer(batch, return_tensors="pt", padding=True)
            inputs_batch = {k: v.to(gpu) for k, v in inputs_batch.items()}

            with torch.no_grad():
                try:
                    responses = model.module.generate(
                        **inputs_batch,
                        max_new_tokens=1000,
                        do_sample=False,
                        temperature=0,
                        top_p=1.0,
                        # no_repeat_ngram_size=2,
                    )

                    torch.cuda.synchronize()  # Ensure all GPU work is complete

                except Exception as e:
                    print(f"Error while generating responses for batch: {e}")
                    continue

                for index, response in enumerate(responses):
                    generated_text = tokenizer.decode(response[len(inputs_batch['input_ids'][index]):],
                                                      skip_special_tokens=True)
                    try:
                        data_dict = parse_output_without_sentence(
                            item_id_pattern, summarization_pattern, reasoning_pattern, generated_text
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        # import pdb; pdb.set_trace()
                        continue

                    item_id = data_dict['item_id']
                    results[item_id] = {}
                    results[item_id]['summarization'] = data_dict['summarization']
                    results[item_id]['reasoning'] = data_dict['reasoning']

                # Free GPU memory by deleting the batch and responses, then clear caches.
                gc.collect()
                torch.cuda.empty_cache()

                if step % 10 == 0:
                    dist.barrier()

            step += 1

        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)

        if dist.get_rank() == 0:
            merged_file = f"{args.fashion_item_profile_path}"
            merged_results = {}

            # Now, merge the gathered results from all GPUs.
            for partial_results in gathered_results:
                merged_results.update(partial_results)

            # Save the updated merged_results back to the file.
            with open(merged_file, "wb") as f:
                pickle.dump(merged_results, f)

            print(f"Merged results saved to {merged_file}")

    finally:
        # Shutdown the distributed process group.
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8, help="Total number of GPUs")
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:23456",
                        help="Initialization method for distributed training")
    parser.add_argument("--item_system_prompt_path", type=str, default="../Datasets/item_system_prompt.txt")
    # parser.add_argument("--beauty_item_prompt_path", type=str, default="../Datasets/beauty_item_prompt.pkl")
    # parser.add_argument("--beauty_item_profile_path", type=str, default="../Datasets/beauty_item_profile.pkl")
    parser.add_argument("--fashion_item_prompt_path", type=str, default="../Datasets/fashion_item_prompt.pkl")
    parser.add_argument("--fashion_item_profile_path", type=str, default="../Datasets/fashion_item_profile.pkl")

    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Spawn one process per GPU.
    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))


if __name__ == '__main__':
    main()
