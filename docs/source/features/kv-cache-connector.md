# KV Cache Connector

The KV Cache Connector is a flexible interface in TensorRT-LLM that enables remote or external access to the Key-Value (KV) cache. It allows developers to implement custom logic for loading, saving, and managing KV cache blocks, extending the capabilities of the standard KV cache manager.

This document explains the KV Cache Connector architecture, common use cases, and provides a detailed walkthrough of the included example.

## Use Cases

The KV Cache Connector is designed to support a variety of advanced serving scenarios:

1. **KV Cache Offloading**: Move KV cache blocks from GPU memory to cheaper/larger storage (CPU RAM, NVMe SSD, or network storage) when they are not immediately needed, and reload them when required.
2. **Custom Disaggregated Serving**: Separate the prefill (context processing) and decode (token generation) phases onto different instances or machines. The connector can be used to transmit the KV cache generated during prefill to the decode instances.
3. **KV Cache Sharing / P2P Transfer**: Share KV cache states between different model instances or across peer-to-peer connections.

## Architecture

The connector architecture is split into two main components:

* **Scheduler (Leader)**: Responsible for orchestration. It decides *what* needs to be loaded or saved and builds metadata instructions. With tensor parallelism it runs on rank 0. With attention data parallelism (ADP), each rank runs an owner-local scheduler adapter.
* **Worker**: Responsible for execution. It receives metadata from the scheduler and performs the actual data transfers (loading/saving) on the KV cache tensors. It runs on all ranks.

### API Reference

To implement a custom connector, you must subclass `KvCacheConnectorScheduler` and `KvCacheConnectorWorker`.

#### 1. Scheduler (Leader) Interface (`KvCacheConnectorScheduler`)

These methods run on the leader process for TP, or on each request-owning rank for ADP.

* **`build_connector_meta(self, scheduler_output: SchedulerOutput) -> object`**
  * **Description**: The core orchestration method. Called during the scheduling phase. It examines the current requests and decides which blocks need to be loaded from or saved to the external store.
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full, so the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. Under TP it is broadcast to all workers. Under ADP it is bound only to the local worker; `scheduler_output.attention_dp_rank` identifies the owner of its block IDs.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Called when a new request arrives. It checks to see if any KV cache can be loaded from an external KV store.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.
  * **Note**: under sliding-window attention `cache_block_ids` covers the live window, not the whole prompt. See [What a connector can persist under a sliding window](#what-a-connector-can-persist-under-a-sliding-window).

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: on `KVCacheManagerV2` with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk, because V2 allocates per chunk rather than for the whole prompt. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on later calls to `build_connector_meta`, on the entries under `scheduler_output.cached_requests`. A connector that treats this callback as its only source of block ids will under-plan. Read both lists:

    ```python
    def build_connector_meta(self, scheduler_output):
        for req in scheduler_output.new_requests:      # the first chunk
            self._plan(req.request_id, req.new_block_ids)
        for req in scheduler_output.cached_requests:   # every later chunk
            self._plan(req.request_id, req.new_block_ids)
    ```

    Both example connectors walk `new_requests` only, so neither one demonstrates this.

* **`cancel_load(self, request: LlmRequest, start: int, end: int)`**
  * **Description**: Release the ownership `get_num_new_matched_tokens` took for prompt tokens `[start, end)`, which the runtime will not consume. Offsets are absolute prompt positions, on the same scale as `num_computed_tokens`.
  * **Range handling**: the runtime can cancel either end of the offer, or the whole offer. When local reuse overtakes a leading range while the request waits, preserve the remaining suffix: the runtime still counts those tokens as externally loaded. Mooncake trims its pending block range before building the transfer metadata.
  * **When it is called**: only under `aggressive_prefix_budgeting`, where the query runs early enough that it is no longer binding. Not implementing it makes that mode fail at start-up rather than at runtime; every other configuration never reaches it. See [Contributing the prefix to the scheduler's budget](#contributing-the-prefix-to-the-schedulers-budget).

* **`update_state_after_alloc_by_layer_group(self, request: LlmRequest, block_ids_by_layer_group: list[list[int]])`**
* **`request_finished_by_layer_group(self, request: LlmRequest, cache_block_ids_by_layer_group: list[list[int]]) -> bool`**
  * **Description**: the per-layer-group forms of the two callbacks above, indexed by layer group id. Entry `[g][i]` is the page slot of block ordinal `i` in layer group `g`.
  * **When they are called**: whenever the KV cache has more than one layer group — that is, one attention window size per group, as under variable sliding-window attention (VSWA). A page index is scoped to a group, so indices from different groups cannot share one list, and the flat `block_ids` / `cache_block_ids` are empty in that case. With a single layer group — every non-VSWA, non-hybrid model — only the flat forms are called and these are never reached.
  * **Which form to implement**:
    * To run on a single layer group — every non-VSWA, non-hybrid model — implement `update_state_after_alloc` and `request_finished`.
    * To run on a VSWA or hybrid model, implement `update_state_after_alloc_by_layer_group` and `request_finished_by_layer_group`.

    A connector that implements only the flat forms is rejected on a multi-group model rather than silently reporting empty lists. See [Running under VSWA](#running-under-vswa).

##### Running under VSWA

Under variable sliding-window attention the KV cache allocates one pool per attention window size, and a page index only means something inside its own layer group. A single tensor and a single flat block list cannot describe that, so three methods have to be implemented together:

| Method | Replaces |
|---|---|
| `KvCacheConnectorWorker.register_kv_cache_layout` | `register_kv_caches` |
| `KvCacheConnectorScheduler.update_state_after_alloc_by_layer_group` | `update_state_after_alloc` |
| `KvCacheConnectorScheduler.request_finished_by_layer_group` | `request_finished` |

Implementing the per-layer-group form of a pair is enough — the flat method it replaces does not also have to be defined.

Implement them together, because a missing one surfaces at a different moment. `register_kv_cache_layout` refuses during executor bring-up, before any request is admitted, naming the group and region counts it could not describe. The two scheduler defaults are only reached once a request is scheduled, so a connector that overrides `register_kv_cache_layout` alone starts up cleanly and raises `NotImplementedError` on the first request instead.

`examples/llm-api/llm_kv_cache_connector_vswa.py` is a worked connector for this case. A VSWA connector has to do five things:

1. **Address pages per group.** `layout.groups[g].regions[r]` gives the byte ranges; `region.slot_tensor(i)` is page slot `i` *of that group*. `layout.group_of_layer(layer_id)` maps a model layer back to its group, which is what the per-layer `wait_for_layer_load` / `save_kv_layer` hooks need.
2. **Read the per-group block lists.** `RequestData.new_block_ids_by_layer_group[g]` carries the page slots; the flat `new_block_ids` is empty.
3. **Carry the layer group in the cache key, and in every transfer target.** This one is a correctness requirement, not a convenience. The same token range exists in *every* layer group holding **different** KV, so a key derived from the token sequence alone collides across groups and one group's bytes will overwrite another's — then be loaded back into the wrong group. Mix `layer_group_id` (or the window size, or the layer set) into the identifier, and carry `(layer_group_id, page_slot)` rather than `page_slot` alone as the transfer target.
4. **Filter out-of-window blocks through `valid_page_slots`**, and size the store for the window rather than the prompt. See below.
5. **Serve a block only when every group holds it.** A full-attention group keeps the whole prompt while a sliding group keeps only its window, so the prefix that can be served back is bounded by the smallest window. Stop the lookup at the first block ordinal any group misses.

##### `KvCacheLayout` reference

```python
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import KvCacheLayout
```

The type is passed to `register_kv_cache_layout`; importing it is only needed for a type annotation.

A layout describes the byte ranges that repeat once per page slot. It describes ranges rather than
implying them, which is what lets one type cover MLA (a pool simply has no `value` buffer), block
scales, sliding-window attention and hybrid models without any of them being a special case.

| Attribute | Meaning |
|---|---|
| `layout.tokens_per_block` | Tokens covered by one page. |
| `layout.dtype` | Element type of the KV data, for a typed view over a region. |
| `layout.groups` | The layer groups, as `KvCacheLayerGroupLayout`. |
| `layout.group(layer_group_id)` | One group by id. Raises `KeyError` if absent. |
| `layout.group_of_layer(layer_id)` | The group owning a model layer — what the per-layer hooks route on. |
| `layout.as_single_pool_tensor()` | The `[num_blocks, num_layers, kv_factor, block_size]` view a single-pool cache hands `register_kv_caches`, or `None` when the cache cannot be described that way. This is what the default `register_kv_cache_layout` calls. |

Each `KvCacheLayerGroupLayout`:

| Attribute | Meaning |
|---|---|
| `group.layer_group_id` | The index page slots are scoped to. Dense, starting at 0. |
| `group.window_size` | Attention window for the group, or `None` for full attention. |
| `group.layer_ids` | Global model layer indices in the group — the same index space `wait_for_layer_load` and `save_kv_layer` receive. |
| `group.regions` | The `KvCacheRegion`s making up one page of this group. |
| `group.bytes_per_page` | Total bytes the group occupies for one page slot. |

Each `KvCacheRegion` is a contiguous byte range that repeats once per page slot:

| Attribute | Meaning |
|---|---|
| `region.base` | Device address of page slot 0. |
| `region.size` | Bytes the region covers within one slot. |
| `region.stride` | Distance between consecutive slots. |
| `region.num_slots` | Number of page slots. |
| `region.buffers` | The `(layer_id, role, expansion)` tuples the region covers, in memory order. `role` is the cache manager's own name, e.g. `"key"` / `"value"`. |
| `region.address_of(slot)` | `base + stride * slot`. Raises `IndexError` outside `[0, num_slots)`. |
| `region.as_tensor(dtype=torch.uint8)` | A strided `[num_slots, size // itemsize]` view; row `i` is page slot `i`. Accepts any subscript, including `-1`. |
| `region.slot_tensor(slot_id, dtype=torch.uint8)` | The bytes of one page slot, raising `IndexError` outside `[0, num_slots)`. The guarded form of `as_tensor(dtype)[slot_id]`. |

`size` is not necessarily `stride`: a region covers one run of adjacent buffers within a slot, and a
slot may hold several runs. For a model with uniform layer shapes the buffers coalesce into a single
region spanning the whole slot, which is the whole-page transfer. A group with more than one region
must be addressed region by region.

The addresses are device addresses, and they stay valid because every cache tier below GPU is
rejected at bring-up while a connector is attached. See [KV cache tiers](#kv-cache-tiers-are-gpu-only-under-a-connector).

##### KV cache tiers are GPU-only under a connector

A connector registers device addresses and holds them across iterations. Evicting a page to another
tier reassigns its GPU slot underneath the connector, so on `KVCacheManagerV2`:

* Setting `KvCacheConfig.host_cache_size` or `KvCacheConfig.disk_cache_size` above zero **fails at
  bring-up**, with a message naming both settings.
* Leaving `host_cache_size` unset **drops the host tier** rather than failing, with a log line. That
  tier is provisioned automatically only to give the `MAX_UTILIZATION` scheduler somewhere to spill
  to via suspend/resume, which a connector run does not use.
* `enable_kv_pool_rebalance` is **ignored** — startup and inference continue, and the rebalance
  simply never runs. Rebalance suspends every active request and runs a defragmenting migration that
  reassigns the same page slots a tier eviction would.

The practical consequence is that a KV-exhausted connector deployment on V2 has no secondary tier to
fall back on. The remedies are `kv_cache_config.max_tokens`,
`kv_cache_config.free_gpu_memory_fraction`, or lowering `max_num_tokens` to hand memory back to the
KV pool; the V2 scheduler's exhaustion error says so directly when a connector is attached.

##### Block reuse alongside the connector

* On `KVCacheManagerV2`, specify `KvCacheConfig.enable_block_reuse=True` alongside a connector, or
  the combination is rejected at start-up. The check reads the value the manager resolved, not the
  one you passed: some quantization algorithms, some SM versions and hybrid linear models turn
  block reuse off on their own, so this error can appear without the flag being set anywhere in
  your configuration.
* On `KVCacheManager` (V1) the same pair is **not** rejected, but the connector's prefix is never
  honoured: the lookup, the reads and the device copies are performed and discarded, at no
  correctness cost but at full latency cost. Treat it as a configuration to avoid.

##### Fields not populated on `KVCacheManagerV2`

Two `RequestData` fields are reported empty when the connector runs on `KVCacheManagerV2`.

| Field | On V2 | Consequence |
|---|---|---|
| `block_hashes` | always `[]` | V2 has no block-hash accessor on this path. Nothing in the runtime reads the field, and neither example connector uses it — both hash the token sequence themselves. A connector that keys its external store on `block_hashes` gets no key and therefore no hits and no saves; it does not mis-address a transfer. |
| `priorities` | always `None` | `KvCacheRetentionConfig` does not reach `KVCacheManagerV2` at all, so every page carries the default priority. A warning is logged the first time a request carrying a retention config is reported. The gap is wider than the connector: a retention config set on V2 has no effect either way. |

Both are gaps to be closed rather than intended differences, and neither is a regression: on V1 both
fields behave as they always have.

##### Blocks with no page

A block that has no page in a layer group is reported as `-1` (`BAD_PAGE_INDEX`) **in place**, not dropped from the list. This keeps each entry aligned with its block ordinal, so entry `i` always describes tokens `[i * tokens_per_block, (i+1) * tokens_per_block)` and an append-delta over successive calls stays valid.

That alignment is also why the list is not safe to index with directly: `-1` is a valid Python and PyTorch subscript, so it resolves to the *last* page slot of the pool rather than raising — a transfer against another request's KV. Two API points keep a page index from reaching device memory unchecked.

| | |
|---|---|
| `valid_page_slots(page_indices)` | Yields `(block_ordinal, page_slot)` for the entries that address a page. The ordinal is preserved, so the token range a page covers is still recoverable. |
| `region.slot_tensor(slot_id)` | The bytes of one page slot, raising `IndexError` on a slot outside `[0, num_slots)`. |

Build transfer targets with `valid_page_slots` and address them with `slot_tensor`. This covers `block_ids`, `cache_block_ids`, `RequestData.new_block_ids`, and both `*_by_layer_group` forms.

```python
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import valid_page_slots

for ordinal, slot in valid_page_slots(cache_block_ids):
    tokens = all_tokens[ordinal * tokens_per_block:(ordinal + 1) * tokens_per_block]
    store.put(self._key(tokens), region.slot_tensor(slot))
```

##### What a connector can persist under a sliding window

Under sliding-window attention, a connector can persist **at most `window_size` tokens per sequence**, not `prompt_len`.

The KV cache manager reclaims a block's page once the window has moved past it, so by the time `request_finished` runs there is no readable KV for anything older than the last `window_size` tokens. Those ordinals report no page (see [Blocks with no page](#blocks-with-no-page)), and the page slots offered to save from cover the live window only. A prefix-caching connector on such a model therefore caches a tail rather than a prefix, and the prefix it can serve back on a later request is bounded the same way.

This is a property of the cache, not of the connector: the blocks are gone whether or not a connector is attached. The same bound applies to the KV cache transceiver, which drops the same range before sending.

##### Serving a prefix on `KVCacheManagerV2`

A connector written against the V1 manager runs on `KVCacheManagerV2` unchanged for any model with a single attention window size. `KVCacheManagerV2` describes its pools rather than handing over one tensor, so it calls `register_kv_cache_layout` instead of `register_kv_caches` — but that method's default reconstructs the single-pool tensor, in the same `[num_blocks, num_layers, kv_factor, block_size]` shape and KV dtype, and forwards it to `register_kv_caches`. The same applies to the two block-id callbacks: their per-layer-group forms default to the flat ones when there is a single layer group.

Variable sliding-window attention is the case where that stops working, because the cache then allocates one pool per window size and a page index is scoped to a layer group. See [Running under VSWA](#running-under-vswa).

By default both managers ask `get_num_new_matched_tokens` at the same point in the iteration: once the batch for the upcoming forward pass is final. On V1 that is inside `addSequence`, called from `KVCacheManager.prepare_resources`; on V2 it is `KVCacheManagerV2.prepare_resources` directly. A request that is asked is therefore a request that runs, and the connector can take ownership of remote blocks in the query and release it in `request_finished`. `KVCacheManagerV2` can also ask earlier, which changes that guarantee; see [Contributing the prefix to the scheduler's budget](#contributing-the-prefix-to-the-schedulers-budget).

Two differences are worth knowing when tuning a deployment.

* **The runtime may honour less than you offer.** V1 allocates KV for the whole prompt when the request's first chunk is scheduled, so an offer always fits. V2 allocates per context chunk, which is what lets chunked prefill bound its memory, so an offer reaching past the current chunk requires the runtime to grow the allocation and that can fail under pressure. The runtime then serves the part it can cover and computes the rest locally. The amount actually served is what `RequestData.computed_position` reflects; the unserved remainder needs no action from the connector beyond its usual `request_finished` cleanup.
* **The query is not part of the scheduler's budget by default.** The V2 scheduler sizes a request's chunk as if the connector will serve nothing, so a served prefix reduces the work in the forward pass but does not free budget for another request in the same iteration. `aggressive_prefix_budgeting` changes that.

Specify `enable_block_reuse=True` alongside the connector for any of this to run on `KVCacheManagerV2`; see [Block reuse alongside the connector](#block-reuse-alongside-the-connector).

`get_num_new_matched_tokens` is called **at most once per KV allocation**. This is the precise form of the "once per request" rule, and it holds on both managers: if a request's KV cache is destroyed and the request is replayed -- which `MAX_UTILIZATION` does under memory pressure -- the replay asks again, because the pages the first answer described are gone.

**Deployment note.** Under V2 with a connector, a workload that was token-bound becomes KV-bound: the connector removes forward-pass tokens but its prefix still occupies GPU pages. Lowering `max_num_tokens` to hand memory back to the KV pool is usually the right adjustment, the opposite of the guidance for a connector-free deployment.

##### Contributing the prefix to the scheduler's budget

```python
KvCacheConnectorConfig(connector="my-connector", aggressive_prefix_budgeting=True)
```

With this set, `KVCacheManagerV2` asks the connector inside the scheduling pass rather than once the batch is final. The scheduler skips the request past the served range before it checks the token budget, so a served prefix frees budget for another request in the same iteration instead of only shrinking this one's forward pass. On a workload with a high remote hit rate, that is the difference between a served prefix improving latency and it improving throughput.

The mode requires `use_kv_cache_manager_v2=True` and `scheduler_config.enable_prefix_aware_scheduling=True`. It is refused at start-up otherwise, and refused if the connector's scheduler does not implement `cancel_load`.

**What the connector has to implement.** The query is no longer binding. Every stage between scheduling and the forward pass can still drop the request, the local cache can overtake the offer while the request waits for a slot, and the pages may not cover it. `cancel_load(request, start, end)` is how the runtime hands an offer back, and it is called in four situations:

| Situation | Range handed back |
| --- | --- |
| The offer runs past `prompt_len - 1`, or past the last whole block below it | the trimmed tail |
| No pages could cover the offer this iteration | the whole offer |
| The local cache committed part of the range while the request waited | the overlapping prefix |
| The request was cancelled, timed out, or failed before the offer was delivered | the whole undelivered offer |

Release whatever ownership `get_num_new_matched_tokens` took for that range. For a synchronous offer nothing has transferred yet, so the release is exact; for one reported as asynchronous the transfer has already started and cancelling is lossy.

An offer is trimmed to a whole block before it is recorded, because the scheduler sizes the following chunk from the served end. A connector offering fewer tokens than one block therefore serves nothing, and gets the whole offer back through `cancel_load`.

**One query per allocation still holds.** A request refuted after scheduling keeps its recorded offer and delivers it in a later iteration rather than asking again. A request whose KV cache is destroyed has its offer handed back and asks again on replay, which is the same rule as the default mode.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache, shaped `[num_blocks, num_layers, kv_factor, block_size]`. Row `block_id` is that block's KV for every layer. Dimension 1 is indexed by model layer in ascending order, so the `layer_idx` passed to `wait_for_layer_load` and `save_kv_layer` indexes it directly — no mapping is supplied, and none is needed. This holds on both managers: V1 has a layer-to-pool-offset map but it is the identity for a single pool, and `KVCacheManagerV2` lays each pool out layer-major and ascending.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization *instead of* `register_kv_caches` when the cache describes itself as pools rather than one tensor, which is what `KVCacheManagerV2` does. `KvCacheLayout` gives byte ranges per layer group: `layout.groups[g].regions[r]`, where the data for page slot `i` is at `region.base + region.stride * i` for `region.size` bytes, or `region.slot_tensor(i, dtype)`. Full attribute reference: [`KvCacheLayout` reference](#kvcachelayout-reference).
  * **Default**: reconstructs the single-pool tensor and forwards it to `register_kv_caches`, so a connector that does not override this needs no changes for any single-window model. It raises when the cache cannot be described as one tensor — several layer groups (VSWA), or several regions (block scales, layers of differing size).

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization **instead of** `register_kv_caches` when the KV cache manager is `KVCacheManagerV2`, whose memory cannot be expressed as one tensor: there is one slot address space per pool and one page-index space per layer group. The default implementation raises, so a connector that does not implement it can only run on V1.
  * **Arguments**: `layout` describes the byte ranges that repeat per page slot. Each `KvCacheLayerGroupLayout` carries a tuple of `KvCacheRegion`s, and the bytes for page slot `i` of a region live at `region.base + region.stride * i` for `region.size` bytes, or equivalently at `region.as_tensor()[i]`. Page indices arriving in `RequestData.new_block_ids_by_layer_group` are scoped to a layer group and index that group's regions.
  * **Why regions rather than a tensor**: because the ranges are described rather than implied, the same structure covers MLA (a pool simply has no `value` buffer), sliding-window and hybrid models (one layer group per window size), and non-uniform slots such as MiniMax-M3's index-K buffer sitting beside K/V, without any of them being a special case.

* **`start_load_kv(self, stream: torch.cuda.Stream)`**
  * **Description**: Initiates the loading of KV blocks from the external source into the GPU memory.
  * **Arguments**: `stream` is the CUDA stream where the forward pass is executed in.

* **`wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream)`**
  * **Description**: A synchronization point. Ensures that the KV cache for a specific layer is fully loaded before the model attempts to perform the forward pass on that layer.

* **`save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream)`**
  * **Description**: Triggers the saving of a specific layer's KV cache.

* **`wait_for_save(self, stream: torch.cuda.Stream)`**
  * **Description**: A synchronization point to ensure all save operations are enqueued or completed.

* **`get_finished(self, finished_gen_req_ids, started_loading_req_ids) -> tuple[list[int], list[int]]`**
  * **Description**: Polled by the runtime to check the status of asynchronous operations.
  * **Returns**: Two lists of request IDs: those that have finished saving, and those that have finished loading.

## Attention data parallelism

Set `enable_attention_dp=True` with a connector whose **scheduler and worker
classes both declare `supports_attention_dp = True`**. Existing connectors that
have not opted in are rejected before construction. No extra scheduler adapter
configuration is required: the executor creates a scheduler and worker on each
ADP rank, including rank 0.

Each adapter owns its request lookup, allocation feedback and worker metadata.
Connector callbacks and completion polling do not perform collectives between
ADP owners. The current ADP mapping has one attention worker per owner (model TP
ranks still cooperate for the non-attention computation). TP without ADP keeps
the rank-0 scheduler and waits for all attention shards to finish a transfer.

Adapters can use **one shared logical storage pool**. They must use distinct
worker endpoints and owner-scoped request/transfer IDs, while using common,
representation-compatible content keys for reusable KV. A local block ID is an
index into the registered local tensor, not a globally addressable page.
Do not partition the content namespace by ADP rank: distinguish model revision,
KV layout/dtype, block size, complete preceding prefix and cache salt instead.
Pool capacity and placement remain the backend's responsibility; enabling ADP
does not turn unused peer HBM into directly usable local attention memory.

The capability flag commits an implementation to these requirements:

* Scheduler constructors and callbacks work on nonzero ranks and never require
  all ADP owners to issue the same requests or call sequence.
* Workers consume only local metadata. Dummy requests do not appear in storage
  lookup, allocation feedback, metadata or request-finished callbacks. Empty
  metadata is valid, including during a dummy-only forward.
* `get_finished` reports only IDs previously provided to that worker and only
  after all transfers touching the corresponding local blocks have completed.
  Cancellation is deferred until those DMA users drain; the generic API does
  not provide a transport abort operation.
* Independently arriving writes/readers share content safely, with complete
  publication, compatible representations and backend pinning during reads.

Async loading may remove every real request from an owner's scheduled batch.
The executor attempts to add a compute dummy so other owners can advance while
the transfer proceeds. If the owner has no dummy capacity or sequence slot,
the existing ADP forward gate still defers the batch.

ADP supports the V1 single-primary-pool interface and V2 layer-group layouts,
including the combined target and one-model draft layout. PP=1, CP=1 and beam
width 1 are required. V1 retains guaranteed-no-evict scheduling; V2 retains its
allocation and asynchronous-save lifetime handling. Internal host/disk tiers
and Mamba/hybrid state remain unsupported. Mooncake requires V2 and rejects
sliding-window layouts. Third-party presets must explicitly opt in.

The built-in `mooncake-store` adapter shares one unsharded attention namespace
across ADP owners. Each owner opens its own store client and contributes its
configured segment to the common master. TP uses separate keys for each
attention shard and still requires all shards for a prefix hit. A disaggregated
DEP4 prefill / TEP8 decode deployment attaches the store connector to prefill;
decode can donate host memory while keeping its native KV transceiver. The
store does not convert attention shard layouts during prefill-to-decode handoff.

## Built-in Connectors

Named presets can be selected without naming a module or class:

```python
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")
```

The available presets are `lmcache`, `lmcache-mp`, `kvbm` and `mooncake-store`. The first three are external packages; `mooncake-store` ships with TensorRT-LLM and is described below.

### Mooncake distributed store (`mooncake-store`)

Publishes KV pages into a [Mooncake](https://github.com/kvcache-ai/Mooncake) store, a shared CPU memory pool addressed by content, so a prefix computed by one engine can be replayed by another. Regular block reuse cannot do this, because it never leaves the instance that computed the prefix.

This is a **different component** from the Mooncake transfer engine that the C++ cache transceiver uses for disaggregated prefill/decode handoff. That moves KV point to point between two known peers; this publishes pages into a pool that any peer can read. The two compose: a context server can write pages into the store and still hand off to a generation server over NIXL.

#### Requirements

* `KVCacheManagerV2` (`kv_cache_config.use_kv_cache_manager_v2: true`), since that is the manager that can describe its pools through `register_kv_cache_layout`.
* The Mooncake Python bindings: `pip install mooncake-transfer-engine`. These are installed in the release container; the source build of the C++ transfer engine does not provide them.
* A reachable Mooncake master (and metadata server, unless using `P2PHANDSHAKE`). See the [Mooncake documentation](https://kvcache-ai.github.io/Mooncake/). `trtllm-serve` can start one for a single engine; see below.
* GPU-only KV cache tiers: set `kv_cache_config.host_cache_size: 0` and `disk_cache_size: 0`. A page evicted to another tier has its GPU slot reassigned, which would invalidate the addresses registered with the store.

#### Configuration

Describe the pool in `kv_connector_config.mooncake_store` and `trtllm-serve` provisions it during bringup: it resolves the master, renders the client config, and exports `MOONCAKE_CONFIG_PATH` before the ranks that open store handles are spawned.

```yaml
kv_connector_config:
  connector: mooncake-store
  mooncake_store:
    master_server_address: 10.0.0.1:50051   # a master with its own lifetime
    protocol: rdma
    device_name: mlx5_0
    global_segment_size: 32GiB
    local_buffer_size: 1GiB
```

Replacing `master_server_address` with `launch_master: true` makes the server start a `mooncake_master` itself and use it, so a single-instance deployment needs nothing prepared outside `trtllm-serve`. **That master lives and dies with the server**, which makes it wrong for anything else: several engines that should share one pool would each get their own, and a pool meant to survive a restart cannot be owned by the thing restarting.

A master started this way still publishes its address, to `master.addr` in the run directory and to `master_address_file` if one is named. That is how other processes, the donors below above all, find a pool this server owns, and how a finished run's logs say which master it used:

```yaml
mooncake_store:
  launch_master: true
  master_address_file: /shared/master.addr
```

The two cases above, several engines or surviving a restart, run the master as its own command instead:

```bash
trtllm-serve mooncake_master --rpc_port 50051 --address_file /shared/master.addr
```

The pool then lasts as long as that command, independently of any engine. `--address_file` receives `host:port` once the master accepts connections, and `master_server_address` accepts `file://<path>` as well as a literal address:

```yaml
kv_connector_config:
  connector: mooncake-store
  mooncake_store:
    master_server_address: file:///shared/master.addr
```

This is what makes a master reachable without anyone writing its address down. Under a scheduler its host is not known when the configs are written; publishing it to a file the configs already name closes that gap, and a server reading the file waits for it, so the master and the engines can be started in any order. The file is removed when the master stops, so a stale address is never dialed.

`TRTLLM_MOONCAKE_MASTER_BINARY` overrides the binary a launched master runs, and `TRTLLM_MOONCAKE_MASTER_TIMEOUT` (default 60s) sets how long startup waits for any master to accept connections or publish its address. Without that wait, a master that is not there yet fails inside every rank after the model has loaded. Set `TRTLLM_MOONCAKE_RUN_DIR` to keep the generated client config and the master's log, which are otherwise in a temporary directory removed at shutdown.

#### Servers whose ranks the launcher starts

Provisioning happens in the server process and reaches the ranks that open store handles by exporting `MOONCAKE_CONFIG_PATH` for them to inherit. That holds when the LLM constructor spawns them. It does not when the launcher starts one task per rank, as `trtllm-llmapi-launch` under a scheduler does, because those ranks were already running.

Naming a shared run directory covers that case: the rendered config is read back from `$TRTLLM_MOONCAKE_RUN_DIR/mooncake.json` by any rank that inherited no path, so every rank of a multi-GPU server joins the pool its own leader provisioned. The directory has to be one they all see, which under a scheduler means the job's own, and it is where the master's log and published address already go:

```bash
export TRTLLM_MOONCAKE_RUN_DIR=/shared/run/$SLURM_JOB_ID
srun trtllm-llmapi-launch trtllm-serve "$model" --config ctx.yaml
```

Without it, a rank that inherited nothing fails during bringup naming `MOONCAKE_CONFIG_PATH`, rather than serving without a store.

#### Reading bringup in the log

Everything the pool is assembled from is logged under the `mooncake-store:` prefix before the model loads, because a pool that came up wrong is otherwise visible only as a low hit rate hours later. In order: the run directory, the master's command line and pid, the address it published and where, the rendered client config in full, and the capacity each rank will contribute. A server lending memory logs the master it resolved, the segment in both GiB and bytes, and the transport, so that a size string parsed wrong is caught before the pool starts evicting far too eagerly.

Both waits report progress every five seconds, since waiting for a master in another job is normal and indistinguishable from a hang if it is silent. A master that dies during startup has the tail of its own log quoted in the failure, which is where the reason, a port in use or a bad flag, actually is.

#### Pool capacity

Capacity comes only from processes that open a store handle, and `global_segment_size` is what each contributes, so the pool is that value times the number of such processes. In a disaggregated deployment the connector belongs on the context servers only, which makes every byte of the pool prefill-node memory: prefill's DRAM caching prefill's GPUs, largely duplicating what `kv_cache_config.host_cache_size` already does.

To give the pool memory from nodes whose engines run no connector, ask those servers to lend it:

```yaml
# generation server: no connector, memory only
mooncake_donation:
  master_server_address: file:///shared/master.addr
  segment_size: 320GiB
  protocol: rdma
  device_name: mlx5_1
```

`trtllm-serve` then holds that segment for as long as the server runs, so a generation node holds pages prefill wrote while its own engine stays connector-free and keeps its cache transceiver for the prefill-to-decode handoff. The server is ready only once the segment is mounted, which makes its readiness the signal that the pool has this capacity.

Lending memory is deliberately outside `kv_connector_config`, and not a `TRTLLM_MOONCAKE_STORE_ROLE` either. Both of those attach a connector, and a connector reads or writes: `producer`, `consumer` and `both` all describe traffic, and none of them means "contribute memory only". Expressing capacity there would therefore start this server using the store. Capacity and traffic are separate, and configured separately.

Size is charged **per server process, not per rank**, unlike `global_segment_size`. Two servers on one node lend twice this. The memory is charged to the process and competes with everything else on the node, `kv_cache_config.host_cache_size` above all, so size the two together.

A node that runs no server at all can still lend, as its own command:

```bash
trtllm-serve mooncake_donor --master_server_address file:///shared/master.addr \
    --segment_size 160GiB --protocol rdma --device_name mlx5_0
```

Topology can equally come from a JSON file named by `MOONCAKE_CONFIG_PATH`, using the same schema as the vLLM Mooncake store connector so one deployment can point both engines at the same pool:

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "global_segment_size": "32GiB",
  "local_buffer_size": "1GiB"
}
```

Only `master_server_address` is required. `metadata_server` may be left out, in which case it is `P2PHANDSHAKE`, Mooncake's peer-to-peer handshake, which is also what `mooncake_store` and `mooncake_donation` default to; the example above names a metadata service instead.

An inherited `MOONCAKE_CONFIG_PATH` wins over `mooncake_store` and is logged as doing so, so an orchestrator that already provisions the pool, as the SLURM benchmark harness does, keeps working unchanged.

Three further settings are TensorRT-LLM's rather than Mooncake's, and stay in the environment because they are per process rather than per pool:

| Variable | Default | Meaning |
|---|---|---|
| `TRTLLM_MOONCAKE_STORE_ROLE` | `both` | `producer` writes only, `consumer` reads only, `both` does both. |
| `TRTLLM_MOONCAKE_STORE_PREFIX` | `trtllm` | Leading component of every key, for isolating deployments that share a pool. |
| `TRTLLM_MOONCAKE_STORE_MODEL_KEY` | model directory basename | Identity keys are namespaced by. Two engines share cache only when they agree on it, so the default is the basename rather than the full path, since the same checkpoint is routinely mounted elsewhere on another host, which is exactly what sharing is for. |

In a disaggregated deployment, run context servers as `both` and leave generation servers unconfigured. Generated tokens are rarely a reused prefix, so writing them costs bandwidth for no hit rate.

#### Partial block reuse is forced off

`kv_cache_config.enable_partial_reuse` is set to `false` when this connector is configured, with a warning, whether or not it was requested explicitly. It defaults to `true`, so most deployments will see that warning.

The store is addressed by whole blocks. The connector is handed the device match as `num_computed_tokens` and offers only blocks beyond it, but it can resume only from a block boundary, so when the device match ends mid-block it declines the lookup and the store is not consulted at all. Partial reuse is precisely what puts the match off a boundary, so it trades part of one block of device reuse for every stored block of the remaining prefix. Measured on MiniMax-M3, leaving it enabled declined 97.2% of lookups and left actual prompt cache read at 35% against a 96% ceiling; forcing it off raised that to 94% and roughly doubled throughput.

For one-model speculative decoding such as MTP, V2 also aligns the final local reuse claim after applying the required prompt lookahead backoff. Disabling partial radix matching alone does not keep this claim aligned: a one-token backoff can turn a complete block into a partial local prefix and suppress the whole store lookup. Alignment discards at most the remaining partial block while retaining the lookahead safety requirement, so Mooncake can load the stored suffix.

#### How it keys pages

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so the connector derives block identity itself: a blake2b chain where each block's hash covers its own tokens *and* every token before it, seeded by the request's `cache_salt`. A key is `<prefix>/<model>/w<attention shard count>r<attention shard rank>/lg<layer group>/t<tokens per block>b<bytes per page>/<block hash>`. Under ADP all owners use `w1r0`, since each holds complete attention KV; the MPI owner rank remains local transfer state. The namespace pins down everything that would make the stored bytes mean something different, so a mismatched shard count, layer group or page geometry reads as a cache miss rather than as garbage.

The value for one key is the concatenation of that layer group's regions for one page slot, handed to Mooncake's multi-buffer batch APIs as a list of `(address, size)` pairs.

For a one-model draft with a known prompt lookahead, each block hash also covers the following lookahead tokens. The hash seed identifies the lookahead length, so these entries cannot alias older token-only entries. A stored target-plus-draft page must match the tokens that produced both caches, including draft inputs beyond the block boundary.

The scheduler publishes only complete blocks covered by the upcoming forward pass; the worker waits for that pass before reading them. Prompt hashes, allocated capacity, and speculative scratch can exist before their KV values are valid, so their presence alone does not make a page ready to save. The versioned hash seed makes entries written before this completed-block validation appear as misses after an upgrade.

#### Transfer behavior

* **Loads are synchronous**, performed in `start_load_kv` before the forward pass. A failed load raises: the runtime has already counted those tokens as computed, so a partial load is a wrong answer rather than a slow one.
* **Saves are asynchronous**, handed to a background thread behind a CUDA event recorded on the forward stream. The pages are only complete once the pass that wrote them retires, and blocking the executor loop on an RDMA write is the cost the store exists to avoid. The leader reports such requests as saving asynchronously, so their pages stay pinned until `get_finished` confirms the writes landed. A dropped save is logged rather than raised, since it only costs a future cache miss.
* Pages the store already holds are skipped, so several ranks or instances converging on the same prefix write it once.

#### Tuning TTFT and retained capacity

Measure end-to-end time to first token (TTFT) and connector load time separately. TTFT also includes request queues, context computation, the prefill-to-decode handoff, and decode scheduling. A high prefix hit rate alone does not establish low TTFT.

For disaggregated serving, tune the generation server's `max_batch_size` and `cuda_graph_config.batch_sizes` together for the intended concurrency. A generation limit chosen for low concurrency can leave requests waiting after a fast prefill or restore. Verify that `max_num_tokens` covers the resulting generation and draft-token budget, and that the resolved GPU KV capacity accommodates the active sequences.

For full-attention models using V2 without an eviction tier, disaggregated decode admission reserves logical GPU-page headroom through each request's output limit, including speculative padding. Incoming transfers may overlap decoding, but cannot consume the pages reserved for admitted requests to finish. The reservation is released when the request finishes or is cancelled; it does not eagerly allocate output pages. Windowed and recurrent caches retain their existing admission policy. This GPU admission budget is separate from the shared Mooncake store capacity.

Under the same full-attention, disaggregated, no-eviction-tier conditions, the default connector prefill path reserves headroom through the end of each admitted prompt. Chunk allocation remains incremental. A new request waits for GPU headroom before querying the connector, which prevents other requests' small chunks from consuming the space needed to restore its matching prefix. Returning an unconsumed first chunk to the queue releases its reservation; request completion and cancellation release it too. This policy applies when `aggressive_prefix_budgeting=False`. Store residency alone does not establish a cache hit: the prefix must match and the receiving GPU must have room to restore it.

Size the shared pool from retained payload measured over the complete workload, including warmup and final drain. Concurrency is not proportional to storage demand when conversation trees have different prefix lengths and branching. Check segment capacity, eviction counters, and allocation failures throughout the run, and leave headroom for allocator overhead and newly replayed prefixes.

#### Unsupported configurations

These are rejected at startup, before any request is admitted:

| Configuration | Reason |
|---|---|
| Context parallelism | A rank holds a slice of the sequence rather than whole blocks of it, so one key would name different bytes on different ranks. |
| Sliding-window attention / VSWA | A page's validity depends on where the window sits, which is a property of the request that read it rather than of the tokens it holds. |
| MiniMax-M3 with `sparse_disable_index_value: false` | The index-V cache is a plain tensor outside the paged pools, so a replayed prefix would pair stored index-K with stale index-V. Disaggregated serving applies the same restriction. |
| Pipeline parallelism | Untested rather than unsound. Use tensor parallelism. |
| `KVCacheManagerV1` | Identity here is a per-layer-group hash chain; V1 supplies real block hashes over a single flat block space. |

Beam search, non-GPU cache tiers and Mamba caches are rejected for all connectors by the executor. Attention DP is supported through a local scheduler adapter on every owner.

#### Example

`examples/llm-api/configs/trtllm_mooncake_store_connector_extra.yaml` is a starting point for `trtllm-serve`.

## Example Implementation

The file `examples/llm-api/llm_kv_cache_connector.py` provides a reference implementation of a **Persistent KV Cache**.

### Overview

This example implements a file-system based KV cache.
1. **Save**: When a request finishes or needs to be swapped out, its KV blocks are saved to disk as `.pt` files.
2. **Load**: When a new request arrives with the same prompt prefix, the connector identifies the cached files and loads them back into GPU memory, skipping re-computation.

### Implementation Details

* **Metadata**: The example defines a `PersistentKvCacheConnectorMetadata` dataclass containing lists of `(file_path, block_id)` tuples for both loading and saving. This simple structure allows the Scheduler to tell the Worker exactly which file corresponds to which GPU block index.

* **Hashing Strategy**: The `PersistentKvCacheConnectorLeader` uses SHA-256 over the complete prefix through each block and its cache salt. Keys are stable across independently started ADP processes.

* **Worker Logic**:
  * `start_load_kv`: Iterates through the load list provided in the metadata, loads the `.pt` file to CPU, and copies it to the specific `block_id` in the GPU tensor.
  * `wait_for_save`: Performs the reverse. It copies data from the GPU `block_id` to CPU and saves it to disk using `torch.save`.
    It writes a temporary file in the same directory and atomically publishes
    the completed file so concurrent owners cannot read a partial write.

For an ADP demonstration, point `CONNECTOR_CACHE_FOLDER` at the same shared
filesystem directory on every rank. Use a dedicated directory for each model
revision and KV representation. The example supports unsharded attention KV
(single rank or ADP); it does not support sharded attention TP or chunked prefill.
This filesystem example demonstrates sharing, not elastic HBM placement or
production performance.

### Limitations & Patterns

This example illustrates the API mechanics but has several limitations that make it unsuitable for high-performance production use without modification:

1. **Blocking I/O**: The example uses `torch.load` and `torch.save` synchronously. In a real implementation, these should be offloaded to a background thread or asynchronous I/O handler to avoid stalling the GPU.
2. **Simplified Block Matching**: The `get_num_new_matched_tokens` implementation in the example only matches full blocks. It does not handle partial cache hits.
3. **FileSystem Latency**: Storing one file per block can create high filesystem overhead.

### Usage

To run the example:

```bash
python examples/llm-api/llm_kv_cache_connector.py <model_path>
```

The script demonstrates:

1. Generating text for a prompt (First run).
2. Destroying the LLM instance.
3. Creating a new LLM instance with the same connector config.
4. Generating text for the same prompt (Second run).
5. Asserting that the outputs match, proving the state was correctly restored from the disk cache.
