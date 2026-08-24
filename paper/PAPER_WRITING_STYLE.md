# Paper-writing style for NeurIPS-class submissions

This is the durable paper-writing preference supplied by Utkarsh. Apply it to
the manuscript, related drafts, appendices, and review responses.

## Section discipline: method is not findings

The methodology describes what we do, not what we found. Move observations,
component comparisons, ablation findings, and verbal ablation results to the
results. Put implementation details, allowed primitives, flags,
hyperparameters, and hidden-evaluation protocols in experimental setup or the
appendix. Put verbose derivations and full enumerations in the appendix, with a
short taxonomy and forward reference in the main text.

When editing a methodology subsection, ask whether a sentence is a design
choice or an empirical claim. Empirical claims belong in results. Method may
use a one-sentence forward reference, but should not give away the finding.

## Move, do not delete

Before deleting content, decide whether it belongs in the introduction,
method, results, conclusion, or appendix. Content removed for placement should
move rather than disappear. Preserve the substance while improving local flow.

## Kill overclaim

Drop claims the data does not strictly support. Drop defensive novelty
callouts and dedicated paragraphs for facts that are not contributions. A
modest, defensible claim is preferable to an aspirational one that an ablation
undercuts.

Put explicit novelty claims in the introduction's contribution list, not in
method. Narrow weak first-to-do-X claims rather than masking conflicting data.
Do not call graceful saturation a phase transition.

## Trim aggressively

When a paragraph is verbose, cut it to one or two sentences. Method should be
clean and abstract, with hooks that establish the design without overexplaining
it. Compress standard formulations and move implementation detail or secondary
equations to the appendix.

## Preserve pre-crystallized one-liners

When Utkarsh supplies a ready-to-paste one-sentence framing, use it verbatim or
near-verbatim at the natural rhetorical anchor. Do not dilute it by expanding
it. Italicize only when it genuinely works as a framing aphorism.

## Sequential reveal

Treat the paper as a document read in order. Each section should set up the
next without revealing the empirical answer early. Every forward reference
must land in a concrete later section. When content moves, update references in
both directions.

## Targeted edits, not rewrites

During line-by-line review, make only the requested change. Do not rewrite
neighboring paragraphs unless explicitly asked. If a nearby issue matters,
raise it separately and wait for confirmation.

## Hooks for novelty inside method; claims in the introduction

Method should communicate novelty through useful abstractions and section
framing, not explicit novelty labels. Put explicit claim-staking in the
introduction's contribution list.

## Report negative and partial findings honestly

When data does not support a claim, soften it. State limitations in results or
the conclusion rather than hiding null effects. The narrative rule is: report
exactly what was found, no more and no less.

## TODO convention

Use `\utkarsh{TODO: ...}` for inline notes. Use descriptive snake-case citation
placeholders such as `\citep{TODO_pinns}` when Utkarsh will supply the citation.
Do not invent bibliography entries or add real citations unless asked.

## Right-size the appendix

Move implementation details, parameter tables, derivations, full
enumerations, auxiliary protocols, and secondary ablations to the appendix
with a forward reference. Keep only concept-carrying equations and headline
evidence in the main text.

## Tone

Use direct, terse, declarative prose. Avoid marketing language and words such
as “interestingly,” “notably,” “remarkably,” “we believe,” and “we hope” unless
they are indispensable. Replace qualitative praise with the measured number.
Prefer “we introduce X” to “we propose a novel framework.” Use `\emph{}`
sparingly for the first mention of a coined term.
