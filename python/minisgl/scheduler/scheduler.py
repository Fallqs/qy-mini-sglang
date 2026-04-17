from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.engine import MultiTenantEngine
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    tenant_id: str
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class TenantUnit:
    """Per-tenant scheduling state."""

    def __init__(self, tenant_id: str, tenant_ctx, config: SchedulerConfig):
        self.tenant_id = tenant_id
        self.tenant_ctx = tenant_ctx
        self.table_manager = TableManager(config.max_running_req, tenant_ctx.page_table)
        self.cache_manager = CacheManager(
            tenant_id=tenant_id,
            num_pages=tenant_ctx.num_pages,
            page_size=config.page_size,
            page_table=tenant_ctx.page_table,
            type=config.cache_type,
            allocator=tenant_ctx.kv_pool.pool_mgr.allocator,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        self.engine = MultiTenantEngine(config)
        self.engine.add_tenant("default", config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize per-tenant units
        self.tenant_units: Dict[str, TenantUnit] = {
            "default": TenantUnit("default", self.engine.get_tenant("default"), config)
        }

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _get_tenant(self, tenant_id: str) -> TenantUnit:
        if tenant_id not in self.tenant_units:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        return self.tenant_units[tenant_id]

    def add_tenant(self, tenant_id: str, config: SchedulerConfig | None = None) -> None:
        """Add a new tenant to the scheduler."""
        cfg = config or self.engine.base_config
        tenant_ctx = self.engine.add_tenant(tenant_id, cfg)
        self.tenant_units[tenant_id] = TenantUnit(tenant_id, tenant_ctx, cfg)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        strict = len(self.tenant_units) == 1
        for unit in self.tenant_units.values():
            unit.cache_manager.check_integrity(strict=strict)

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or any(u.prefill_manager.runnable for u in self.tenant_units.values())
            or any(u.decode_manager.runnable for u in self.tenant_units.values())
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            any(u.prefill_manager.runnable for u in self.tenant_units.values())
            or any(u.decode_manager.runnable for u in self.tenant_units.values())
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        tenant_id = last_data[0].tenant_id
        unit = self._get_tenant(tenant_id)
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with unit.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == unit.eos_token_id
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        tenant_id=tenant_id,
                        model_path=unit.tenant_ctx.config.model_path,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    unit.decode_manager.remove_req(req)
                    self._free_req_resources(req, unit)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    unit.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tenant_id = msg.tenant_id
            if tenant_id not in self.tenant_units:
                logger.warning_rank0(f"Unknown tenant {tenant_id}, using default")
                tenant_id = "default"
            unit = self._get_tenant(tenant_id)
            input_len, max_seq_len = len(msg.input_ids), unit.tenant_ctx.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            unit.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tenant_id = msg.tenant_id
            if tenant_id not in self.tenant_units:
                tenant_id = "default"
            unit = self._get_tenant(tenant_id)
            req_to_free = unit.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or unit.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free, unit)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req, unit: TenantUnit) -> None:
        unit.table_manager.free(req.table_idx)
        unit.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, tenant_id: str, batch: Batch) -> ForwardInput:
        unit = self._get_tenant(tenant_id)
        tenant_ctx = unit.tenant_ctx
        tenant_ctx.ensure_active()
        assert tenant_ctx.graph_runner is not None
        tenant_ctx.graph_runner.pad_batch(batch)
        unit.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = tenant_ctx.page_table[input_mapping]
        tenant_ctx.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            tenant_id=tenant_id,
            batch=batch,
            sample_args=tenant_ctx.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # Decode batches can be inspected without mutating state.
        decode_candidates: List[Tuple[str, Batch]] = []
        for tenant_id, unit in self.tenant_units.items():
            batch = unit.decode_manager.schedule_next_batch()
            if batch is not None:
                decode_candidates.append((tenant_id, batch))

        if decode_candidates:
            # TODO: support other policies: e.g. round-robin, largest batch
            tenant_id, batch = max(decode_candidates, key=lambda x: x[1].size)
            return self._prepare_batch(tenant_id, batch)

        # Prefill batches mutate tenant state (table/cache allocation), so only
        # schedule the first tenant that can produce a batch.
        for tenant_id, unit in self.tenant_units.items():
            batch = unit.prefill_manager.schedule_next_batch(self.prefill_budget)
            if batch is not None:
                return self._prepare_batch(tenant_id, batch)

        return None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        tenant_id = forward_input.tenant_id
        unit = self._get_tenant(tenant_id)
        batch, sample_args, input_mapping, output_mapping = (
            forward_input.batch,
            forward_input.sample_args,
            forward_input.input_tuple,
            forward_input.write_tuple,
        )
        batch.input_ids = unit.table_manager.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args, tenant_id=tenant_id)
        unit.table_manager.token_pool[output_mapping] = forward_output.next_tokens_gpu
        unit.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
