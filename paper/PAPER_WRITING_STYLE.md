---
name: Paper-writing style for NeurIPS-class submissions
description: How Utkarsh prefers to draft, revise, and trim ML/SciML papers — section discipline, claim discipline, edit discipline.
type: feedback
originSessionId: 0eff7f22-38ca-4bf8-8b43-276d8a05c2da
---
# Paper-writing style for NeurIPS-class submissions

A consolidated set of preferences observed while drafting a NeurIPS submission. Apply these whenever helping with the paper, related drafts, or review responses.

## Section discipline: method ≠ findings

**Rule:** The methodology section describes *what we do*, not *what we found*. Move all of the following out of methods, in order of priority:

- "We observe X" / "We find Y" → results
- Comparisons of components (e.g., "the role of SFT vs RL") → results, framed as observations
- Ablation rationale and verbal ablation results → results
- Implementation details (allowed primitives, sandbox flags, hyperparameters) → experimental setup or appendix
- Hidden-evaluation protocol → experimental setup
- Verbose math or PDE-family enumerations → appendix; main text gets a one-paragraph taxonomy and a forward-ref

**Why:** Reviewers read the method to understand what was built, then the results to learn what was discovered. Mixing the two reads as defensive (method as discussion) and bloats the page count. This is also how senior reviewers expect papers to be structured.

**How to apply:** When writing or editing a methodology subsection, ask: "Is this a *design choice* or an *empirical claim*?" Empirical claims belong in results. The method can have one-sentence forward-references ("we examine this empirically in §X") but never fold the empirical answer back in.

## Move, don't delete

**Rule:** Before deleting content from the paper, decide where else it could live (intro, method, results, conclusion, appendix). Removed content from one section should land in another, not vanish.

**Why:** Useful content that was cut for placement reasons is almost always still load-bearing somewhere else. Outright deletion loses information that would have to be recreated; the reflex of relocating preserves the substance while improving the local flow.

**How to apply:** When a subsection or paragraph is being cut, propose a new home along with the cut. Examples from this project: "two-phase dynamics" cut from method → moved to results §two-phase; "roles of SFT and RL" cut from method → results §sft-rl-roles; novelty paragraph cut from method → contributions list in intro.

## Kill overclaim, even when it would be flattering

**Rule:** Drop claims that the data does not strictly support, drop "novelty callouts" that read as defensive, drop dedicated paragraphs for things that are not the paper's contribution.

**Why:** Reviewers reward narrative but punish overclaim. Kill the hill before R2 attacks it. A modest, defensible claim is worth more than an aspirational one that an ablation undercuts.

**How to apply:** If a paragraph says "to our knowledge, no prior work…" or "we are the first to…" inside the methodology, move it to a contributions list in the intro instead — never embed it inside a method subsection. If a finding is interesting but not the paper's contribution (e.g., an entropy-mechanism connection), give it at most one inline citation; do not give it a dedicated paragraph in related work or method. If the data shows a graceful saturation rather than a sharp transition, do not call it a "phase transition" even if the storytelling would benefit.

## Trim aggressively, then trim more

**Rule:** When a paragraph is described as "verbose," cut it to one or two sentences. Aim for the methodology to read clean and abstract, with hooks/anchors that hint at novelty rather than explain it.

**Why:** Method bloat dilutes the technical contribution and makes reviewers skim. A clean method with strategic forward-refs reads as confident and lets the results section deliver the payoff.

**How to apply:** Replace multi-paragraph subsections with one-sentence anchors when the substance lives elsewhere (e.g., a forward-ref to results). Move standard formulations (vanilla GRPO objective, standard SFT loss) toward appendix or compress to one inline equation. The default state of an unhelpful paragraph is "delete and place the substance elsewhere."

## Pre-crystallized one-liners

**Rule:** Utkarsh frequently provides ready-to-paste one-sentence framings ("Code is the action; physics is the verifier"; "SFT teaches the syntax of solvers; RL teaches when solvers are physically reliable"). Thread these into the appropriate sections verbatim or near-verbatim.

**Why:** These are pre-engineered narrative beats that do a lot of work — they unify multiple paragraphs and stick in reviewers' heads. Paraphrasing dilutes them.

**How to apply:** When the user offers a one-liner, place it at the natural rhetorical anchor (opening of methodology subsection, opening of results subsection, contributions bullet) and italicize if it works as a framing aphorism. Do not water down or expand into a fuller explanation; the punch is the point.

## Sequential reveal — treat the paper as a whole

**Rule:** The paper is read sequentially; each section should set up the next without giving away the answer. Forward-references should land somewhere concrete in a later section, not in another forward-reference.

**Why:** Sequential reveal is how readers actually consume papers. A method that telegraphs results loses the payoff; results without method anchors feel ad hoc.

**How to apply:** When adding a forward-ref ("we examine this empirically in §X"), confirm §X actually answers the question. When refining one section, re-read the surrounding sections to keep the arc coherent. When content moves between sections, update the forward and backward references on both ends.

## Targeted edits, not rewrites

**Rule:** When the user is going through the draft line-by-line with specific requests, make only the requested change. Do not rewrite surrounding paragraphs unless explicitly asked.

**Why:** A line-by-line review is a sequential mental process; an unsolicited rewrite of an adjacent paragraph forces the reviewer to re-read everything and breaks their concentration. The user has explicitly objected to this in the past.

**How to apply:** When a request is "remove paragraph X" or "trim sentence Y," do exactly that. Do not pre-emptively reorganize the section or harmonize neighbors. If you notice a related issue, mention it as a separate suggestion at the end and wait for confirmation.

## Hooks for novelty inside method, claims in intro

**Rule:** The methodology section should "speak novelty" through abstractions and hooks (e.g., introducing a "verifiability gap" that motivates the design) rather than through explicit "this is novel" sentences. Explicit novelty claims live in the intro contributions list.

**Why:** Methodology novelty embedded in the prose feels confident; explicit "Novelty:" callouts feel defensive and invite challenge. Contributions in the intro are the canonical place for explicit claim-staking and reviewers expect to find them there.

**How to apply:** When inserting a new methodology subsection, give it an evocative name (e.g., "The Verifiability Gap in PDE Code Generation") and let the framing carry the novelty. If you want to stake an explicit claim, write it as a bullet in the intro contributions paragraph instead.

## Honest about negative or partial findings

**Rule:** When data does not support a claim, soften it. State limitations explicitly in the results or conclusion. Do not paper over null effects.

**Why:** Reviewers respect honesty about limits and punish hidden weaknesses. The narrative discipline is "we found exactly this much" — not more, not less.

**How to apply:** When the data shows a smaller or different effect than the methodology suggested, edit the methodology language to match. Limitations belong in the results' last subsection or in conclusion's "limitations and future work" paragraph. If a "first to do X" claim becomes weak after data, narrow the scope (e.g., "first within this PDE family setting") rather than dropping it entirely.

## TODO marking convention

**Rule:** Inline notes-to-self use `\utkarsh{TODO: ...}` macro (the user has this defined). Citation placeholders that the user will fill in later use `\citep{TODO_xxx}` with descriptive snake-case keys. Do not add citations the user has not asked for; default to placeholder.

**Why:** The user adds real citations themselves and prefers to grep `TODO_` to find them. The `\utkarsh{}` macro renders inline notes visibly during draft compilation.

**How to apply:** When a real citation is needed but the .bib entry is uncertain or speculative, use `\citep{TODO_descriptive_key}` (e.g., `TODO_pinns`, `TODO_deepseekr1`, `TODO_pdebench`). For inline notes, use `\utkarsh{TODO: <thing to do>}`. Never invent a fake bib entry; never add a real bib entry without being asked.

## Right-sizing the appendix

**Rule:** Anything that is implementation detail, parameter table, derivation, full enumeration, or auxiliary protocol goes to the appendix with a forward-ref from the main text. Keep main-text equations limited to those that carry the conceptual content.

**Why:** Eight-page main-text limits at NeurIPS are real and the appendix has effectively no length cap; reviewers will follow refs for details and skim main text for argument.

**How to apply:** Default placements: PDE governing equations and parameter ranges → appendix table; full GRPO objective with clip and KL terms → main text only if it carries substance, otherwise appendix; sandbox primitives and timeouts → appendix; verifier quadrature weights → appendix; example prompts (concrete strings) → appendix; ablation tables that aren't headline numbers → appendix.

## Tone

**Rule:** Direct, terse, no marketing language. State results crisply, avoid "interestingly," avoid "remarkably," avoid hedging beyond what the data requires.

**Why:** Senior writing in this area has a flat declarative tone. Marketing language and aspirational hedges cue reviewers to look harder for problems.

**How to apply:** Strip "interestingly," "notably," "remarkably," "we believe," "we hope" from drafts unless preserving them is load-bearing. Replace "the model achieves substantial gains" with the actual number. Replace "we propose a novel framework" with "we introduce X." Use italics (`\emph{}`) sparingly for first-mention of a coined term.
