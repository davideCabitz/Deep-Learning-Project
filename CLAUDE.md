# CLAUDE.md

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
