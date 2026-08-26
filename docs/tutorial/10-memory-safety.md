# Chapter 10: Memory Safety

**Prerequisites:** [Chapter 0: Getting Started](00-getting-started.md) (helpful for codebase context, not required)

**Time:** ~11 min

> After this chapter you can explain `defer`/`errdefer`, the allocator pattern, and how to detect memory leaks in tests.

Zig's approach to memory management: **explicit allocation, guaranteed cleanup**. No garbage collector, no hidden allocations, no surprises. When you call `allocator.alloc()`, you must call `allocator.free()`, and Zig provides tools to make this **automatic and bulletproof**.

## Code Flow

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d

    Acquire["alloc / open / init\na resource"]:::setup
    Own{"does the caller\nend up owning it?"}
    Defer["defer cleanup()\nruns on every exit"]:::sync
    Fallible{"more fallible\nsteps below?"}
    ErrDefer["errdefer cleanup()\nruns only on error"]:::migration
    None["no defer needed\nownership transfers as-is"]:::success
    Success["function returns normally"]:::success
    Failure["function returns an error"]:::danger

    Acquire --> Own
    Own -- "no, this function owns it" --> Defer
    Own -- "yes, caller owns it" --> Fallible
    Fallible -- "yes" --> ErrDefer
    Fallible -- "no" --> None
    Defer --> Success
    Defer --> Failure
    ErrDefer --> Success
    ErrDefer --> Failure
    None --> Success
```

`defer` and `errdefer` differ only in when they run: `defer` fires on every scope exit, `errdefer` fires only when the scope exits through an error. The rest of this chapter applies that one distinction to real allocation and initialization code.

## defer: Guaranteed Cleanup

`defer` executes a statement when the current scope exits, **always**, whether by normal return, error return, or early return:

```text
processFile(path):
    file = open(path)
    defer close(file)               # runs on every exit, no matter how

    data = readAll(file)
    defer free(data)                # runs before close(file): declared last, runs first

    ... process data ...

    if someCondition:
        return error(Invalid)       # both defers still run

    return                          # normal return: both defers still run
```

**Implementation:** [`src/pull.zig`](../../src/pull.zig) (open/read/close pattern around model downloads)

**Execution order:** Defers run in **reverse order** of declaration (stack unwinding, last declared, first executed):

```mermaid
sequenceDiagram
    participant Code as Function Body
    participant Stack as Defer Stack
    participant Cleanup as Cleanup Actions

    Code->>Stack: defer file.close()
    Code->>Stack: defer allocator.free(data)
    Note over Code: ... work happens ...
    Code->>Code: return (normal or error)
    Stack->>Cleanup: allocator.free(data)  [last declared, first run]
    Stack->>Cleanup: file.close()          [first declared, last run]
```

```text
defer print("Third")
defer print("Second")
defer print("First")
# prints: First, Second, Third (reverse of declaration order)
```

**Why reverse order?** Resources should be released in the opposite order they were acquired (last acquired, first released, like closing nested function calls).

## errdefer: Cleanup Only on Error

`errdefer` runs **only if the function returns an error** after the `errdefer` was declared. It's for cleaning up partial initialization:

```text
initModel(config):
    model.weights = alloc(f32, config.n_params)
    errdefer free(model.weights)              # only if we error out later

    model.cache = KVCache.init(config.max_seq_len)
    errdefer model.cache.deinit()             # only if we error out after this point

    model.backend = Backend.init()
    errdefer model.backend.deinit()

    return model                              # success: no errdefers run, caller owns model
```

**Implementation:** [`src/kvcache/manager.zig`](../../src/kvcache/manager.zig) (`allocKvCache`), [`src/main.zig`](../../src/main.zig) (multi-component init chain)

**What happens on error?**

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    A["alloc weights\nerrdefer free(weights)"]:::setup
    B["KVCache.init()\nerrdefer cache.deinit()"]:::setup
    C["Backend.init()\nerrdefer backend.deinit()"]:::setup
    D["return model  (success)"]:::migration
    E["free(weights)"]:::danger
    F["cache.deinit()\nfree(weights)"]:::danger
    G["caller owns model\nno errdefers run"]:::success

    A --> B
    B --> C
    C --> D
    B -- "KVCache.init() fails" --> E
    C -- "Backend.init() fails" --> F
    D --> G
```

- If `KVCache.init()` fails → only `model.weights` is freed
- If `Backend.init()` fails → `model.cache.deinit()` AND `allocator.free(model.weights)` run
- If all succeed → nothing runs, model is returned to caller

**What happens on success?**

- No `errdefer` runs
- Caller is responsible for cleanup (usually via `model.deinit()`)

## The Pattern: defer + errdefer

**Rule:** Use `defer` immediately after acquiring a resource that must **always** be cleaned up. Use `errdefer` for partial initialization where cleanup depends on success.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Acquire["Resource Acquisition"]
        R1["open file"]:::setup --> D1["defer file.close()"]:::sync
        R2["alloc buffer"]:::setup --> D2["errdefer free(buffer)"]:::sync
        R3["init struct"]:::setup --> D3["errdefer struct.deinit()"]:::sync
    end

    subgraph Exit["Scope Exit"]
        direction TB
        OK["success path\nreturn value"]:::migration --> NE["errdefers skipped\ncaller owns resources"]:::success
        ERR["error path\nreturn error"]:::danger --> ED["errdefers run\npartial state cleaned up"]:::danger
        BOTH["always"]:::migration --> DD["defers run\n(in reverse order)"]:::sync
    end

    Acquire --> Exit
```

**Quick decision guide**, which keyword to reach for:

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    START["You just acquired a resource\n(alloc, open, init)"]:::setup
    USE_DEFER["use defer\ncleanup runs on ALL exits\n(success and error)"]:::sync
    USE_ERRDEFER["use errdefer\ncleanup runs ONLY on error\ncaller gets ownership on success"]:::migration
    NO_DEFER["no defer needed\nreturn and transfer ownership\nno error path to guard"]:::success

    START --> Q1{"Are you going to return\nthis resource to the caller?"}

    Q1 -- "yes\ne.g. return buf, return struct" --> Q2{"Could the function\nerror out after this point?"}
    Q1 -- "no\nthis function owns it fully" --> USE_DEFER

    Q2 -- "yes\nmore fallible steps below" --> USE_ERRDEFER
    Q2 -- "no\nthis is the last step" --> NO_DEFER
```

### Example 1: Simple Allocation

```text
processTokens(tokens):                        # BUG
    embeddings = alloc(f32, tokens.len * 768)
    defer free(embeddings)                    # always runs, even on the return below

    for token, i in tokens:
        ... compute embedding ...
        if token >= vocab_size:
            return error(InvalidToken)        # defer still runs, fine here

    return embeddings                         # WRONG: defer frees it before the caller sees it
```

**Bug:** `defer` runs before the return, so we're returning a pointer to freed memory.

**Fix:** Only use `defer` when you **don't** return the resource:

```text
processTokens(tokens):                        # FIX
    embeddings = alloc(f32, tokens.len * 768)
    errdefer free(embeddings)                  # only cleanup on error

    for token, i in tokens:
        if token >= vocab_size:
            return error(InvalidToken)         # errdefer runs, embeddings freed

    return embeddings                          # success: errdefer skipped, caller owns embeddings
```

**Implementation:** [`src/kvcache/manager.zig`](../../src/kvcache/manager.zig) (allocKvCache errdefer pattern)

### Example 2: Struct with Multiple Resources

**Pattern:** Each struct with allocated resources provides a `deinit()` method:

```text
KVCache:
    fields: keys, values

    init(max_seq_len, kv_dim):
        keys = alloc(u8, max_seq_len * kv_dim)
        errdefer free(keys)

        values = alloc(u8, max_seq_len * kv_dim)
        errdefer free(values)

        return KVCache{ keys, values }

    free(cache):
        free(values)
        free(keys)
```

**Implementation:** [`src/kvcache/manager.zig`](../../src/kvcache/manager.zig) (`allocKvCache`, `freeKvCache`)

**Usage:**

```text
cache = KVCache.init(4096, 640)
defer cache.deinit()                          # always cleanup

... use cache ...
```

**Why this works:**

- If `init()` fails partway through → `errdefer` cleans up what was allocated
- If `init()` succeeds → caller uses `defer cache.deinit()` to clean up later
- No memory leaks on any code path

### Example 3: Nested Initialization

Simplified illustrative example showing how Agave-style multi-component initialization chains defer and errdefer:

```text
initAndRun(args):
    fmt = Format.init(args.model_path)        # loads model weights from disk
    defer fmt.deinit()

    be = Backend.init(args.backend_type)      # GPU/CPU compute
    defer be.deinit()

    tok = Tokenizer.init(fmt)                 # text <-> token IDs
    defer tok.deinit()

    model = Model.init(fmt, be)               # weights + forward pass
    defer model.deinit()

    # if ANY init fails, all prior defers run automatically
    # if all succeed, all defers run at function exit

    runGeneration(model, tok, args)
```

**Implementation:** [`src/main.zig`](../../src/main.zig) (format, backend, tokenizer, and model init chain)

**Clean and safe:** No manual error handling, no forgotten cleanup, no leaks.

## Common Pitfalls

### Pitfall 1: defer in a Loop

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph BAD["BAD, defer in loop body (defers pile up)"]
        direction TB
        B1["iteration 1\nopen file_a\ndefer file_a.close()"]:::migration --> B2["iteration 2\nopen file_b\ndefer file_b.close()"]:::migration
        B2 --> B3["iteration 3\nopen file_c\ndefer file_c.close()"]:::migration
        B3 --> B4["function exits\n(or errors out)"]:::migration
        B4 --> B5["file_c.close()\nfile_b.close()\nfile_a.close()\nALL 3 files were open simultaneously"]:::danger
    end

    subgraph GOOD["GOOD, explicit inner scope forces early run"]
        direction TB
        G1["iteration 1\n{ open file_a\n  defer file_a.close()\n  ... process ... }"]:::setup --> G1C["file_a.close() runs here"]:::success
        G1C --> G2["iteration 2\n{ open file_b\n  defer file_b.close()\n  ... process ... }"]:::setup
        G2 --> G2C["file_b.close() runs here"]:::success
        G2C --> G3["iteration 3\n{ open file_c\n  defer file_c.close()\n  ... process ... }"]:::setup
        G3 --> G3C["file_c.close() runs here"]:::success
    end
```

```text
# BAD: defer accumulates, all run at function exit
for path in files:
    file = open(path)
    defer close(file)                         # wrong: all files stay open until function exits

    ... process file ...
```

**Fix:** Use an explicit scope or call cleanup directly:

```text
# GOOD: explicit inner scope
for path in files:
    {                                          # new scope
        file = open(path)
        defer close(file)                      # runs at end of this block

        ... process file ...
    }                                           # close(file) runs here

# Or: manual cleanup when defer isn't appropriate
for path in files:
    file = open(path)
    errdefer close(file)                       # only cleans up on the error path
    ... process ...
    close(file)                                # explicit cleanup on the success path
```

**Better pattern:** Extract to a helper function:

```text
processFile(path):
    file = open(path)
    defer close(file)                          # runs at end of this function
    ... process ...

for path in files:
    processFile(path)                          # clean and correct: one file open at a time
```

**Implementation:** [`src/pull.zig`](../../src/pull.zig) (per-file processing extracted into a helper so each file's handle closes before the next opens)

### Pitfall 2: Conditional defer

```text
# BAD: defer is unconditional, can't be scoped to just the if
if use_cache:
    cache = alloc(u8, size)
    defer free(cache)                          # runs when the function exits, not at end of if
# cache is out of scope here, but the defer still tries to free it -> use-after-free
```

**Fix:** Don't do this. Use `errdefer` with explicit cleanup, or refactor:

```text
# Option 1: always allocate, conditionally use
cache = use_cache ? alloc(u8, size) : empty_slice
defer if use_cache: free(cache)

# Option 2: refactor into a separate function
if use_cache:
    withCache(size)

withCache(size):
    cache = alloc(u8, size)
    defer free(cache)
    ... use cache ...
```

### Pitfall 3: Forgetting errdefer in Multi-Step Init

```text
# BAD: leaks if the second allocation fails
init():
    buf1 = alloc(u8, 1024)
    buf2 = alloc(u8, 2048)                     # if this fails, buf1 leaks

    return MyStruct{ buf1, buf2 }
```

**Fix:** Use `errdefer` after each allocation:

```text
# GOOD: no leaks on any error path
init():
    buf1 = alloc(u8, 1024)
    errdefer free(buf1)

    buf2 = alloc(u8, 2048)
    errdefer free(buf2)

    return MyStruct{ buf1, buf2 }
```

**Implementation:** [`src/kvcache/manager.zig`](../../src/kvcache/manager.zig) (`allocKvCache`, per-iteration and outer `errdefer` for the per-layer allocation loop)

## Testing for Leaks

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph TestAllocator["std.testing.allocator, Leak Detection Lifecycle"]
        direction TB
        T0["test begins\nallocator created\nalloc table = {}"]:::setup

        T0 --> A1["alloc(u8, 100)\nregisters ptr_A → 100 bytes\ntable = {ptr_A}"]:::sync
        A1 --> A2["alloc(u32, 64)\nregisters ptr_B → 256 bytes\ntable = {ptr_A, ptr_B}"]:::sync
        A2 --> F1["free(ptr_A)\nremoves ptr_A from table\ntable = {ptr_B}"]:::migration
        F1 --> TE["test body exits"]:::migration

        TE --> CHECK{"alloc table\nempty?"}
        CHECK -- "yes\ntable = {}" --> PASS["TEST PASS\nno leaks"]:::success
        CHECK -- "no\ntable = {ptr_B}" --> FAIL["TEST FAIL\nmemory leak detected\nptr_B (256 bytes) was never freed"]:::danger
    end
```

Zig's test allocator **automatically detects leaks**:

```text
test "no leaks":
    allocator = testing.allocator             # tracks every alloc/free

    {
        cache = KVCache.init(1024, 128)
        defer cache.deinit()

        ... test logic ...
    }

    # if any allocation wasn't freed, the test fails with "memory leak detected"
```

**Implementation:** [`src/kvcache/manager.zig`](../../src/kvcache/manager.zig) (tests use `std.testing.allocator` around `allocKvCache`/`deinit`)

**Example failure:**

```text
test "leak example":
    allocator = testing.allocator

    buf = alloc(u8, 100)
    # oops, forgot `defer free(buf)`

# output:
#   Test [leak example] leaked memory.
#   All test allocations must be freed before test completion.
```

This is **your safety net**, write tests, use `std.testing.allocator`, catch leaks before production.

## Advanced Pattern: Arena Allocator

For temporary allocations that all get freed together, use `std.heap.ArenaAllocator`:

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    PA["Parent Allocator\n(e.g. GPA)"]:::setup
    Arena["ArenaAllocator\ndefer arena.deinit()"]:::setup
    A1["tokenize()\narena_alloc"]:::sync
    A2["embed()\narena_alloc"]:::sync
    A3["generate()\narena_alloc"]:::sync
    Final["decode()\nallocator  (parent)"]:::sync
    Return["return text\n(owned by caller)"]:::success
    Free["ALL temp buffers freed\nin one operation"]:::success

    PA --> Arena
    Arena --> A1
    Arena --> A2
    Arena --> A3
    PA --> Final
    A3 --> Return
    Final --> Return
    Arena -- "arena.deinit()" --> Free
```

```text
generateText(prompt):
    arena = ArenaAllocator.init(allocator)
    defer arena.deinit()                       # frees ALL arena allocations in one go

    # every allocation below is freed by arena.deinit()
    tokens          = tokenize(arena, prompt)
    embeddings      = embed(arena, tokens)
    output_tokens   = generate(arena, embeddings)

    # final result: allocate from the parent allocator, not the arena
    text = decode(allocator, output_tokens)

    return text                                 # arena.deinit() runs, cleans up temps
```

**Implementation:** [`src/pull.zig`](../../src/pull.zig) (`std.heap.ArenaAllocator` for request-scoped temporary allocations during model downloads and repository listing)

**When to use:**

- HTTP request handlers (all request-scoped allocations freed together)
- Compiler passes (free all AST nodes after pass completes)
- **Not** for long-lived allocations (model weights, KV cache)

Agave uses arena allocators in `src/pull.zig` for temporary allocations during model downloads and repository listing.

## Memory Safety Checklist

Before merging code, verify:

- [ ] Every `allocator.alloc()` has a matching `defer allocator.free()` or `errdefer allocator.free()`
- [ ] Every `init()` has a matching `defer obj.deinit()` or `errdefer obj.deinit()`
- [ ] Multi-step initialization uses `errdefer` to clean up partial state
- [ ] Resources returned to caller use `errdefer`, not `defer`
- [ ] No `defer` inside loops (unless in an explicit scope)
- [ ] All tests use `std.testing.allocator` (leak detection enabled)

**Tool:** Run tests with leak checking:

```bash
zig build test
# All tests automatically use std.testing.allocator
# Leaks → test failure
```

## Gotchas

**A per-iteration `errdefer` only guards that one iteration, not the ones that already succeeded.** In a loop that allocates one resource per item, `errdefer allocator.free(item[i])` declared inside the loop body cleans up `item[i]` if the current iteration fails, but it already ran out of scope for every earlier iteration that completed without error. `allocKvCache()` in [src/kvcache/manager.zig](../../src/kvcache/manager.zig) shows the fix: alongside the per-iteration `errdefer allocator.free(keys[i])`, it tracks a running `init_count` and installs one more `errdefer` *before* the loop starts that walks `0..init_count`, freeing every layer that already succeeded. Drop that outer errdefer and a failure on layer 5 of 24 leaks layers 0 through 4.

---

**In the code:** Every file with allocations ([src/main.zig](../../src/main.zig), [src/models/](../../src/models/), [src/backend/](../../src/backend/), [src/kvcache/](../../src/kvcache/))

**Related:** [Zig Language Reference, defer](https://ziglang.org/documentation/master/#defer), [Zig Language Reference, errdefer](https://ziglang.org/documentation/master/#errdefer)

**Next:** [Chapter 11: Metal Backend Internals →](11-metal-backend-internals.md) | **Back:** [Chapter 9: CPU SIMD Optimization ←](09-cpu-simd-optimization.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)

---

## Glossary

**arena allocator**, A bulk allocator that frees all allocations at once via `deinit()`, useful for short-lived temporary data.

**defer**, Zig keyword scheduling a statement to execute when the current scope exits, regardless of normal or error exit.

**deinit() pattern**, Convention where structs with owned resources provide a `deinit()` method that releases all internal allocations.

**errdefer**, Zig keyword scheduling cleanup only if the scope exits via an error return.

**explicit allocation**, Zig's memory model where every allocation must be paired with a manual free; no garbage collector.

**std.testing.allocator**, Zig's test allocator that tracks all allocations and automatically detects memory leaks when a test completes.
