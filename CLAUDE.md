# CLAUDE.md

## ⚠️ ALWAYS-FIRST STEP — READ THE SPEC

**Before answering, planning, or implementing ANYTHING, first read
[documents/project_specification.md](documents/project_specification.md) and confirm your
proposal is aligned with it.** This is a hard gate, not a suggestion: no design, refactor, or
code change is valid unless it traces back to the task, evaluation protocol, and constraints in
that file. If a request conflicts with the spec, say so and stop before acting.

## Engineering standards

Act as a senior engineer with deep expertise in software architecture, scalability, and robustness. Every change should leave the codebase cleaner than you found it — write code that a careful reviewer would approve without comment.

### Principles

- **SOLID, always.** Single responsibility per module/class/function; depend on abstractions, not concretions; keep things open for extension but closed for modification. If a file imports a *sibling* peer just to reuse a helper, that helper belongs in a shared module both depend on — never reach sideways across peers.
- **DRY, but not WET.** Duplicated logic is a defect waiting to diverge. Extract the single owner of a responsibility (e.g. a dedicated I/O / persistence / config module) rather than copy-pasting. Resist premature abstraction too — extract on the *second* real occurrence, not the speculative one.
- **Design patterns when they earn their keep.** Reach for the established pattern (factory, strategy, adapter, dependency injection, etc.) when it removes real coupling or duplication — not as decoration. Name the pattern in a comment only when it aids the reader.
- **Robustness.** Validate inputs at boundaries, fail loudly and early with clear messages, handle the error and edge cases (empty, None, NaN, shape mismatch), and never swallow exceptions silently. Make illegal states unrepresentable where you can.
- **Scalability & clarity.** Prefer clear, well-named seams over clever one-liners. Keep functions small and cohesive; keep public surfaces minimal. Choose data structures and algorithms that hold up as inputs grow.
- **MANDATORY REQUIREMENT** Everything you propose must follow what's in documents\project_specification.md. Before implementing or planning something you should check if your propose is linear with what's wrote in the project specification.

### Zero tolerance for code smell

Before finishing any change, self-review for and eliminate: dead code, long parameter lists, deep nesting, god functions, magic numbers/strings, leaky abstractions, circular or sideways imports, mutable shared state, and copy-paste. If you spot a smell adjacent to your change that you can safely fix, fix it; if it's out of scope, flag it.

When a design decision has a real trade-off, state it briefly and recommend the option a senior engineer would pick — don't enumerate every alternative.

## Commenting conventions

Comments exist to convey signal a careful reader can't get for free from the code itself. Do not narrate what the code already says.

### Core rules

- Keep every comment **short but reasonable** — one to a few dense lines, never a paragraph of filler.
- **No long, useless, or redundant comments.** If a comment just restates the method name or the line below it, delete it.
- Don't comment the obvious (`// increment i`, `// loop over items`, `// return the result`).
- Prefer one high-signal comment over several low-signal ones.
- Comment the **why** and the **what-for**, not the **how** — the how is the code.

### Per-method comment style

Every method gets exactly **one** leading comment block in this house style. It should read like a compressed abstract: a label, the core operation (formula where one exists), the approach, its purpose/role, and an optional parenthetical note capturing a non-obvious insight or motivation.

**Canonical example:**

```
// Tier 0 — Vanilla zero-shot baseline (lower bound).
// q = normalize( v_ref + Σ_i α·t⁺_i − Σ_j β·t⁻_j ), score DB by cosine.
// Naïve latent arithmetic, no SVD, no learning. Establishes the floor
// and exercises the eval pipeline.
// (Note: because of CLIP's modality gap, adding raw text vectors to image
// vectors is geometrically crude — this motivates everything that follows.)
```

**Structure to follow:**

1. **Label** — short name/identifier for what the method is, with its role in parentheses where useful (e.g. *lower bound*, *baseline*, *fast path*).
2. **Core operation** — the defining formula in inline math/notation, or a one-line description of the transform. Use real symbols (`Σ`, `α`, `·`, `⁺`, `⁻`) when they're clearer than prose.
3. **Approach** — what kind of thing this is and what it deliberately does *not* do (`no SVD, no learning`).
4. **Purpose** — why it exists / what it establishes / what it feeds into.
5. **Note (optional)** — a parenthetical insight, caveat, or motivation that a maintainer would otherwise have to rediscover.

Not every method needs all five lines. Methods that are genuinely trivial get a single label line and nothing more. The richer blocks are reserved for methods that carry a real idea.

### Good vs bad

**Bad** — verbose, redundant, zero signal:

```
// This method takes the reference vector and then it adds the positive
// text vectors to it one by one, and after that it subtracts the negative
// text vectors, and finally it normalizes the result and returns it so
// that we can use it to score the database later on in the pipeline.
```

**Good** — dense, house style:

```
// Tier 0 — zero-shot baseline. q = normalize(v_ref + Σα·t⁺ − Σβ·t⁻), cosine-scored.
// Raw latent arithmetic, no learning — the floor every later tier must beat.
```

Keep it tight. If you can't say something worth a maintainer's attention, write the single label line and stop.
