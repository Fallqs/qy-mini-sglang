from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    from collections import defaultdict

    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenizer_cache: dict[str, any] = {}
    detokenize_manager_cache: dict[str, DetokenizeManager] = {}

    def _get_tokenizer(path: str):
        if path not in tokenizer_cache:
            tokenizer_cache[path] = load_tokenizer(path)
        return tokenizer_cache[path]

    def _get_detokenize_manager(path: str):
        if path not in detokenize_manager_cache:
            detokenize_manager_cache[path] = DetokenizeManager(_get_tokenizer(path))
        return detokenize_manager_cache[path]

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)

            if len(detokenize_msg) > 0:
                detokenize_groups: defaultdict[str, list[DetokenizeMsg]] = defaultdict(list)
                for msg in detokenize_msg:
                    detokenize_groups[msg.model_path or tokenizer_path].append(msg)
                all_replies: list[tuple[DetokenizeMsg, str]] = []
                for path, msgs in detokenize_groups.items():
                    dm = _get_detokenize_manager(path)
                    replies = dm.detokenize(msgs)
                    all_replies.extend(zip(msgs, replies))
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in all_replies
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                tokenize_groups: defaultdict[str, list[TokenizeMsg]] = defaultdict(list)
                for msg in tokenize_msg:
                    tokenize_groups[msg.model_path or tokenizer_path].append(msg)
                all_user_msgs: list[UserMsg] = []
                for path, msgs in tokenize_groups.items():
                    tm = TokenizeManager(_get_tokenizer(path))
                    tensors = tm.tokenize(msgs)
                    all_user_msgs.extend(
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                            tenant_id=msg.tenant_id,
                        )
                        for msg, t in zip(msgs, tensors, strict=True)
                    )
                batch_output = BatchBackendMsg(data=all_user_msgs)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[
                        AbortBackendMsg(uid=msg.uid, tenant_id=msg.tenant_id)
                        for msg in abort_msg
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
