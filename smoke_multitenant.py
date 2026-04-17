from __future__ import annotations

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig

LLAMA_PATH = "/mnt/sdb1/mkx/model/Llama-3.2-1B-Instruct"
QWEN_PATH = "/mnt/sdb1/mkx/model/Qwen3-8B"


def build_config(model_path: str) -> SchedulerConfig:
    return SchedulerConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=2,
        cuda_graph_bs=[1, 2],
        offline_mode=True,
        num_page_override=20000,
    )


def main():
    print("=" * 60)
    print("Multi-tenant smoke test: Llama-3.2-1B + Qwen3-8B")
    print("=" * 60)

    llama_cfg = build_config(LLAMA_PATH)
    qwen_cfg = build_config(QWEN_PATH)

    print("[1/5] Initializing scheduler with Llama as default tenant...")
    scheduler = Scheduler(llama_cfg)

    # Collect detokenized replies instead of sending over ZMQ
    replies_buffer: list = []
    scheduler.send_result = lambda msgs: replies_buffer.extend(msgs)

    print("[2/5] Adding Qwen3-8B as tenant 'qwen'...")
    scheduler.add_tenant("qwen", qwen_cfg)

    print("[3/5] Tokenizing prompts...")
    llama_tok = AutoTokenizer.from_pretrained(LLAMA_PATH)
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_PATH)

    llama_ids = torch.tensor(
        llama_tok.encode("The capital of France is"), dtype=torch.int32
    )
    qwen_ids = torch.tensor(
        qwen_tok.encode("法国的首都是"), dtype=torch.int32
    )
    print(f"      Llama prompt ids: {llama_ids.tolist()}")
    print(f"      Qwen  prompt ids: {qwen_ids.tolist()}")

    print("[4/5] Injecting requests...")
    scheduler._process_one_msg(
        UserMsg(
            uid=1,
            input_ids=llama_ids,
            sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
            tenant_id="default",
        )
    )
    scheduler._process_one_msg(
        UserMsg(
            uid=2,
            input_ids=qwen_ids,
            sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
            tenant_id="qwen",
        )
    )

    print("[5/5] Running inference loop...")
    max_steps = 30
    finished_uids = set()
    generated: dict[int, list[int]] = {1: [], 2: []}

    for step in range(max_steps):
        forward_input = scheduler._schedule_next_batch()
        if forward_input is None:
            print(f"      Step {step}: no runnable batch (all done or idle)")
            break

        tenant_id = forward_input.tenant_id
        with scheduler.engine_stream_ctx:
            forward_output = scheduler._forward(forward_input)

        scheduler._process_last_data((forward_input, forward_output))

        for msg in replies_buffer:
            generated[msg.uid].append(msg.next_token)
            if msg.finished:
                finished_uids.add(msg.uid)
        replies_buffer.clear()

        print(
            f"      Step {step}: tenant={tenant_id:<7} "
            f"batch_size={forward_input.batch.size} "
            f"phase={forward_input.batch.phase}"
        )

    print("-" * 60)
    print("Results:")
    for uid, tokens in generated.items():
        model_name = "Llama" if uid == 1 else "Qwen"
        text = (llama_tok if uid == 1 else qwen_tok).decode(tokens, skip_special_tokens=True)
        print(f"  {model_name} (uid={uid}): tokens={tokens}")
        print(f"           decoded='{text}'")
        print(f"           finished={uid in finished_uids}")

    assert 1 in finished_uids, "Llama request did not finish"
    assert 2 in finished_uids, "Qwen request did not finish"
    print("-" * 60)
    print("SUCCESS: Both tenants produced output and finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
