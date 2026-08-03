# Chapter 0: Getting Started

**Time:** ~15 min

> After this chapter you can explain the path from a model file on disk to sampled text tokens.

<!-- TODO: Task 2 — full prose -->

### Code Flow

```mermaid
flowchart TD
  classDef setup fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
  classDef sync fill:#dcfce7,stroke:#22c55e,color:#14532d
  classDef success fill:#bbf7d0,stroke:#16a34a,color:#14532d

  Load["load model artifact"]:::setup --> Tok["tokenize prompt"]:::sync
  Tok --> Prefill["prefill"]:::sync
  Prefill --> Decode["decode loop"]:::sync
  Decode --> Sample["sample token"]:::sync
  Sample --> Text["detokenize to text"]:::success
```

## Gotchas

- Placeholder — filled in Task 2.

## How This Relates to the Code

**In the code:** [`main` generation path](../../src/main.zig)

```text
load → tokenize → prefill → (forward → sample → append)* → text
```

**Next:** [Chapter 1: Tokens and Text →](01-tokens-and-text.md) | **Product docs:** [README](../../README.md)

---

## Glossary

(Terms added in Task 2.)
