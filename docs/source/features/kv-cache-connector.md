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

* **Scheduler (Leader)**: Responsible for orchestration. It decides *what* needs to be loaded or saved and builds metadata instructions. It runs only on the leader rank (rank 0).
* **Worker**: Responsible for execution. It receives metadata from the scheduler and performs the actual data transfers (loading/saving) on the KV cache tensors. It runs on all ranks.

### API Reference

To implement a custom connector, you must subclass `KvCacheConnectorScheduler` and `KvCacheConnectorWorker`.

#### 1. Scheduler (Leader) Interface (`KvCacheConnectorScheduler`)

These methods run on the leader process and drive the connector's behavior.

* **`build_connector_meta(self, scheduler_output: SchedulerOutput) -> object`**
  * **Description**: The core orchestration method. Called during the scheduling phase. It examines the current requests and decides which blocks need to be loaded from or saved to the external store.
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full -- the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. This object is broadcasted to all workers.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Called when a new request arrives. It checks to see if any KV cache can be loaded from an external KV store.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.
  * **Note**: under sliding-window attention `cache_block_ids` covers the live window, not the whole prompt. See [What a connector can persist under a sliding window](#what-a-connector-can-persist-under-a-sliding-window).

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on later calls to `build_connector_meta`, on the entries under `scheduler_output.cached_requests`. A connector that treats this callback as its only source of block ids will under-plan. Read both lists:

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
  * **When it is called**: only under `aggressive_prefix_budgeting`, where the query runs early enough that it is no longer binding. Not implementing it makes that mode fail at start-up rather than at runtime; every other configuration never reaches it. See [Contributing the prefix to the scheduler's budget](#contributing-the-prefix-to-the-schedulers-budget).

* **`update_state_after_alloc_by_layer_group(self, request: LlmRequest, block_ids_by_layer_group: list[list[int]])`**
* **`request_finished_by_layer_group(self, request: LlmRequest, cache_block_ids_by_layer_group: list[list[int]]) -> bool`**
  * **Description**: the per-layer-group forms of the two callbacks above, indexed by layer group id. Entry `[g][i]` is the page slot of block ordinal `i` in layer group `g`.
  * **When they are called**: whenever the KV cache reports page indices per layer group. A page index is scoped to its group — one group per attention window size — so a cache with more than one group can be described no other way, and the flat `block_ids` / `cache_block_ids` are empty there. With a single layer group the flat lists carry that group's indices as well, so a connector that implements only the flat forms keeps working on those models.
  * **Which form to implement**: implement exactly one complete set.

    | Set | Models it covers |
    |---|---|
    | per-layer-group | every model, VSWA included |
    | flat | non-VSWA, non-hybrid only |

    A set is complete when both of its methods are defined. Mixing the two — one method from each — is rejected during executor bring-up, naming the method that is missing. An existing flat connector needs no change on the models it already covers: with a single layer group the base per-layer-group implementation folds back to the flat call. Hybrid / linear-attention models are not enabled for the connector yet; the per-layer-group set is the shape their cache will need. See [Running under VSWA](#running-under-vswa).

##### Running under VSWA

Under variable sliding-window attention the KV cache allocates one pool per attention window size, and a page index only means something inside its own layer group. A single tensor and a single flat block list cannot describe that, so three methods have to be implemented together:

| Method | Replaces |
|---|---|
| `KvCacheConnectorWorker.register_kv_cache_layout` | `register_kv_caches` |
| `KvCacheConnectorScheduler.update_state_after_alloc_by_layer_group` | `update_state_after_alloc` |
| `KvCacheConnectorScheduler.request_finished_by_layer_group` | `request_finished` |

Implementing the per-layer-group form of a pair is enough — the flat method it replaces does not also have to be defined.

All three are checked during executor bring-up, before any request is admitted. `register_kv_cache_layout` refuses there, naming the group and region counts it could not describe, and the two scheduler methods are checked alongside it. Nothing is deferred to the first request, so a partial implementation costs a start-up failure rather than one after the model is loaded.

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
tier reassigns its GPU slot underneath the connector, so with a connector attached:

* Setting `KvCacheConfig.host_cache_size` or `KvCacheConfig.disk_cache_size` above zero **fails at
  bring-up**, with a message naming both settings.
* Leaving `host_cache_size` unset **drops the host tier** rather than failing, with a log line. That
  tier is provisioned automatically only to give the `MAX_UTILIZATION` scheduler somewhere to spill
  to via suspend/resume, which a connector run does not use.
* `enable_kv_pool_rebalance` is **ignored** — startup and inference continue, and the rebalance
  simply never runs. Rebalance suspends every active request and runs a defragmenting migration that
  reassigns the same page slots a tier eviction would.

The practical consequence is that a KV-exhausted connector deployment has no secondary tier to
fall back on. The remedies are `kv_cache_config.max_tokens`,
`kv_cache_config.free_gpu_memory_fraction`, or lowering `max_num_tokens` to hand memory back to the
KV pool; the scheduler's exhaustion error says so directly when a connector is attached.

##### Block reuse alongside the connector

* Specify `KvCacheConfig.enable_block_reuse=True` alongside a connector. Without it the connector's
  prefix is never honoured: either the combination is rejected at start-up, or the lookup, the reads
  and the device copies are performed and discarded, at no correctness cost but at full latency
  cost.
* The start-up check reads the value the cache resolved, not the one you passed: some quantization
  algorithms, some SM versions and hybrid linear models turn block reuse off on their own, so this
  error can appear without the flag being set anywhere in your configuration.

##### A page slot must not be reassigned underneath the connector

The connector holds page indices across iterations, and `RequestData` reports only the pages appended
since the last call. Anything that hands a slot the connector already knows about to a different
request therefore goes unreported and corrupts the next transfer against it. Three configurations do
that, and each is rejected at bring-up.

| Configuration | Mechanism |
|---|---|
| Speculative decoding | Rejected draft tokens shrink a request's page list, and the freed slot goes to whichever request allocates next. The connector is never told the tail block moved. |
| A capacity scheduler policy other than `GUARANTEED_NO_EVICT` | A destroyed-and-replayed request comes back on different pages, and the connector's per-request block delta is then measured against pages that were freed with it. |
| A host or disk cache tier | Tier eviction reassigns the GPU slot. See [KV cache tiers are GPU-only under a connector](#kv-cache-tiers-are-gpu-only-under-a-connector). |

The exact set the runtime refuses depends on your cache configuration; the bring-up error is
authoritative. Speculative decoding is unsupported with a connector wherever it is not refused —
the same page-list shrink happens there.

##### `RequestData` fields that may not be populated

Two `RequestData` fields can be reported empty depending on the cache configuration. A connector must tolerate both.

| Field | When empty | Consequence |
|---|---|---|
| `block_hashes` | `[]` | No block-hash accessor exists on this path. Nothing in the runtime reads the field, and neither example connector uses it — both hash the token sequence themselves. A connector that keys its external store on `block_hashes` gets no key and therefore no hits and no saves; it does not mis-address a transfer. |
| `priorities` | `None` | `KvCacheRetentionConfig` is not honoured, so every page carries the default priority. A warning is logged the first time a request carrying a retention config is reported. The gap is wider than the connector: the retention config has no effect either way in that configuration. |

Both are gaps to be closed rather than intended behaviour.

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

Alignment is not the same as stability. Under a sliding window an entry that was reported with a page reads back as `-1` once the window passes its block, and the delta — which carries only the ordinals appended since the last call — does not restate it. Save a block when its tokens complete rather than deferring: a completed block is far inside the window for any usable window size, whereas a deferred save can reach a slot the cache has already reclaimed.

##### What a connector can persist under a sliding window

Under sliding-window attention, a connector can persist **at most `window_size` tokens per sequence**, not `prompt_len`.

The KV cache manager reclaims a block's page once the window has moved past it, so by the time `request_finished` runs there is no readable KV for anything older than the last `window_size` tokens. Those ordinals report no page (see [Blocks with no page](#blocks-with-no-page)), and the page slots offered to save from cover the live window only. A prefix-caching connector on such a model therefore caches a tail rather than a prefix, and the prefix it can serve back on a later request is bounded the same way.

This is a property of the cache, not of the connector: the blocks are gone whether or not a connector is attached. The same bound applies to the KV cache transceiver, which drops the same range before sending.

##### Serving a prefix

A connector that implements only the flat callbacks and `register_kv_caches` runs unchanged on any model with a single *non-sliding* attention window. Where the cache describes itself as pools rather than one tensor it calls `register_kv_cache_layout` instead — but that method's default reconstructs the single-pool tensor, in the same `[num_blocks, num_layers, kv_factor, block_size]` shape and KV dtype, and forwards it to `register_kv_caches`. The same applies to the two block-id callbacks: their per-layer-group forms default to the flat ones when there is a single layer group.

Variable sliding-window attention is the case where that stops working, because the cache then allocates one pool per window size and a page index is scoped to a layer group. See [Running under VSWA](#running-under-vswa).

A model whose layers all share one sliding window stays a single layer group, so the flat callbacks still apply and such a connector is not refused. What differs is that the callbacks cover the **live window only**. Blocks the window has passed report `-1` (`BAD_PAGE_INDEX`) in place — the list stays aligned to block ordinals, so entry `i` still describes tokens `[i * tokens_per_block, (i+1) * tokens_per_block)`, but the up-front blocks carry no page and are not available to load into or save from. Filter with `valid_page_slots`, described in [Blocks with no page](#blocks-with-no-page); without it a `-1` resolves to the last page slot of the pool. A warning naming the window size is logged at start-up when a flat-only connector is attached to such a model.

By default both managers ask `get_num_new_matched_tokens` at the same point in the iteration: once the batch for the upcoming forward pass is final. On V1 that is inside `addSequence`, called from `KVCacheManager.prepare_resources`; on V2 it is `KVCacheManagerV2.prepare_resources` directly. A request that is asked is therefore a request that runs, and the connector can take ownership of remote blocks in the query and release it in `request_finished`. `KVCacheManagerV2` can also ask earlier, which changes that guarantee; see [Contributing the prefix to the scheduler's budget](#contributing-the-prefix-to-the-schedulers-budget).

Two differences are worth knowing when tuning a deployment.

* **The runtime may honour less than you offer.** V1 allocates KV for the whole prompt when the request's first chunk is scheduled, so an offer always fits. V2 allocates per context chunk, which is what lets chunked prefill bound its memory, so an offer reaching past the current chunk requires the runtime to grow the allocation and that can fail under pressure. The runtime then serves the part it can cover and computes the rest locally. The amount actually served is what `RequestData.computed_position` reflects; the unserved remainder needs no action from the connector beyond its usual `request_finished` cleanup.
* **The query is not part of the scheduler's budget by default.** The V2 scheduler sizes a request's chunk as if the connector will serve nothing, so a served prefix reduces the work in the forward pass but does not free budget for another request in the same iteration. `aggressive_prefix_budgeting` changes that.

Specify `enable_block_reuse=True` alongside the connector for any of this to run; see [Block reuse alongside the connector](#block-reuse-alongside-the-connector).

`get_num_new_matched_tokens` is called **at most once per KV allocation**. This is the precise form of the "once per request" rule: if a request's KV cache is destroyed and the request is replayed -- which `MAX_UTILIZATION` does under memory pressure -- the replay asks again, because the pages the first answer described are gone.

**Deployment note.** Under a connector, a workload that was token-bound becomes KV-bound: the connector removes forward-pass tokens but its prefix still occupies GPU pages. Lowering `max_num_tokens` to hand memory back to the KV pool is usually the right adjustment, the opposite of the guidance for a connector-free deployment.

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
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache, shaped `[num_blocks, num_layers, kv_factor, block_size]`. Row `block_id` is that block's KV for every layer. Dimension 1 is indexed by model layer in ascending order, so the `layer_idx` passed to `wait_for_layer_load` and `save_kv_layer` indexes it directly — no mapping is supplied, and none is needed.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization *instead of* `register_kv_caches` when the cache describes itself as pools rather than one tensor. `KvCacheLayout` gives byte ranges per layer group: `layout.groups[g].regions[r]`, where the data for page slot `i` is at `region.base + region.stride * i` for `region.size` bytes, or `region.slot_tensor(i, dtype)`. Full attribute reference: [`KvCacheLayout` reference](#kvcachelayout-reference).
  * **Default**: reconstructs the single-pool tensor and forwards it to `register_kv_caches`, so a connector that does not override this needs no changes for any single-window model. It raises when the cache cannot be described as one tensor — several layer groups (VSWA), or several regions (block scales, layers of differing size).

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization **instead of** `register_kv_caches` when the KV cache manager is `KVCacheManagerV2`, whose memory cannot be expressed as one tensor: there is one slot address space per pool and one page-index space per layer group. The default implementation raises, so a connector that does not implement it can only run on V1.
  * **Arguments**: `layout` describes the byte ranges that repeat per page slot. Each `KvCacheLayerGroupLayout` carries a tuple of `KvCacheRegion`s, and the bytes for page slot `i` of a region live at `region.base + region.stride * i` for `region.size` bytes -- or equivalently at `region.as_tensor()[i]`. Page indices arriving in `RequestData.new_block_ids_by_layer_group` are scoped to a layer group and index that group's regions.
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

## Built-in Connectors

Named presets can be selected without naming a module or class:

```python
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")
```

The available presets are `lmcache`, `lmcache-mp`, `kvbm` and `mooncake-store`. The first three are external packages; `mooncake-store` ships with TensorRT-LLM and is described below.

### Mooncake distributed store (`mooncake-store`)

Publishes KV pages into a [Mooncake](https://github.com/kvcache-ai/Mooncake) store -- a shared CPU memory pool addressed by content -- so a prefix computed by one engine can be replayed by another. Regular block reuse cannot do this, because it never leaves the instance that computed the prefix.

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

Those deployments run the master as its own command instead:

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

`TRTLLM_MOONCAKE_MASTER_BINARY` overrides the binary a launched master runs, and `TRTLLM_MOONCAKE_MASTER_TIMEOUT` (default 60s) how long startup waits for any master to accept connections or publish its address -- reaching a master that is not there otherwise fails inside every rank after the model has loaded. Set `TRTLLM_MOONCAKE_RUN_DIR` to keep the generated client config and the master's log, which are otherwise in a temporary directory removed at shutdown.

#### Pool capacity

Capacity comes only from processes that open a store handle, and `global_segment_size` is what each contributes -- so the pool is that value times the number of such processes. In a disaggregated deployment the connector belongs on the context servers only, which makes every byte of the pool prefill-node memory: prefill's DRAM caching prefill's GPUs, largely duplicating what `kv_cache_config.host_cache_size` already does.

To give the pool memory from nodes that run no connector, run a donor on them:

```bash
trtllm-serve mooncake_donor --master_server_address file:///shared/master.addr \
    --segment_size 160GiB --protocol rdma --device_name mlx5_0
```

A donor holds a segment and issues no reads or writes, so a generation node can hold pages that prefill wrote while its engine stays connector-free and keeps its cache transceiver for the prefill-to-decode handoff. Contributing memory is deliberately not a `TRTLLM_MOONCAKE_STORE_ROLE`: the roles describe an engine's traffic, and every one of them reads or writes, so expressing capacity as a role would start that engine using the store. The donated memory is charged to the donor process and competes with anything else on the node, `kv_cache_config.host_cache_size` above all, so size the two together.

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

An inherited `MOONCAKE_CONFIG_PATH` wins over `mooncake_store` and is logged as doing so, so an orchestrator that already provisions the pool -- as the SLURM benchmark harness does -- keeps working unchanged.

Three further settings are TensorRT-LLM's rather than Mooncake's, and stay in the environment because they are per process rather than per pool:

| Variable | Default | Meaning |
|---|---|---|
| `TRTLLM_MOONCAKE_STORE_ROLE` | `both` | `producer` writes only, `consumer` reads only, `both` does both. |
| `TRTLLM_MOONCAKE_STORE_PREFIX` | `trtllm` | Leading component of every key, for isolating deployments that share a pool. |
| `TRTLLM_MOONCAKE_STORE_MODEL_KEY` | model directory basename | Identity keys are namespaced by. Two engines share cache only when they agree on it, so the default is the basename rather than the full path -- the same checkpoint is routinely mounted elsewhere on another host, which is exactly what sharing is for. |

In a disaggregated deployment, run context servers as `both` and leave generation servers unconfigured. Generated tokens are rarely a reused prefix, so writing them costs bandwidth for no hit rate.

#### Partial block reuse is forced off

`kv_cache_config.enable_partial_reuse` is set to `false` when this connector is configured, with a warning, whether or not it was requested explicitly. It defaults to `true`, so most deployments will see that warning.

The store is addressed by whole blocks. The connector is handed the device match as `num_computed_tokens` and offers only blocks beyond it, but it can only resume from a block boundary -- so when the device match ends mid-block, it declines the lookup and the store is not consulted at all. Partial reuse is precisely what puts the match off a boundary, which means it trades part of one block of device reuse for every stored block of the remaining prefix. Measured on MiniMax-M3, leaving it enabled declined 97.2% of lookups and left actual prompt cache read at 35% against a 96% ceiling; forcing it off raised that to 94% and roughly doubled throughput.

#### How it keys pages

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so the connector derives block identity itself: a blake2b chain where each block's hash covers its own tokens *and* every token before it, seeded by the request's `cache_salt`. A key is `<prefix>/<model>/w<world size>r<rank>/lg<layer group>/t<tokens per block>b<bytes per page>/<block hash>`. The namespace pins down everything that would make the stored bytes mean something different, so a mismatched shard count, layer group or page geometry reads as a cache miss rather than as garbage.

The value for one key is the concatenation of that layer group's regions for one page slot, handed to Mooncake's multi-buffer batch APIs as a list of `(address, size)` pairs.

#### Transfer behavior

* **Loads are synchronous**, performed in `start_load_kv` before the forward pass. A failed load raises: the runtime has already counted those tokens as computed, so a partial load is a wrong answer rather than a slow one.
* **Saves are asynchronous**, handed to a background thread behind a CUDA event recorded on the forward stream. The pages are only complete once the pass that wrote them retires, and blocking the executor loop on an RDMA write is the cost the store exists to avoid. The leader reports such requests as saving asynchronously, so their pages stay pinned until `get_finished` confirms the writes landed. A dropped save is logged rather than raised -- it only costs a future cache miss.
* Pages the store already holds are skipped, so several ranks or instances converging on the same prefix write it once.

#### Unsupported configurations

These are rejected at startup, before any request is admitted:

| Configuration | Reason |
|---|---|
| Context parallelism | A rank holds a slice of the sequence rather than whole blocks of it, so one key would name different bytes on different ranks. |
| Sliding-window attention / VSWA | A page's validity depends on where the window sits, which is a property of the request that read it rather than of the tokens it holds. |
| MiniMax-M3 with `sparse_disable_index_value: false` | The index-V cache is a plain tensor outside the paged pools, so a replayed prefix would pair stored index-K with stale index-V. Disaggregated serving applies the same restriction. |
| Pipeline parallelism | Untested rather than unsound. Use tensor parallelism. |
| `KVCacheManagerV1` | Identity here is a per-layer-group hash chain; V1 supplies real block hashes over a single flat block space. |

Beam search, attention data parallelism, non-GPU cache tiers and Mamba caches are rejected for all connectors by the executor.

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

* **Hashing Strategy**: The `PersistentKvCacheConnectorLeader` hashes the token sequence of a block to generate a unique filename (e.g., `hash_value.pt`). This acts as the lookup key.

* **Worker Logic**:
  * `start_load_kv`: Iterates through the load list provided in the metadata, loads the `.pt` file to CPU, and copies it to the specific `block_id` in the GPU tensor.
  * `wait_for_save`: Performs the reverse. It copies data from the GPU `block_id` to CPU and saves it to disk using `torch.save`.

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
