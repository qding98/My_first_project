# LLM Safety / Alignment Literature Review Export

Date: 2026-04-07  
Scope: Public Internet search, focused on 2025-2026 papers and official conference pages  
Research question: Under fixed early-token input/output formats (e.g., fixed CoT prefix, fixed schema, fixed multiple-choice / structured-answer template, fixed prompt wrapper), do safety-related SFT or test-time adaptation / learning procedures induce shortcut learning that increases over-refusal, selective-refusal collapse, or abnormal safety-utility tradeoffs?

## Executive Takeaway

No exact-match paper found; closest cluster is:

- task-specific benign fine-tuning -> safety drift
- prompt/token-form dependence -> shallow refusal shortcut
- reasoning-format / CoT wrapper -> refusal failure or over-refusal snowball

My overall judgment is that this exact framing has not been fully done end-to-end. The nearest collision zone is the combination of:

- task-specific fine-tuning risks
- answer-structure / prompt-template dependence
- early-token or shallow safety alignment
- CoT-specific safety failure

## Bucket 1: Papers Most Likely to Collide with the Core Idea

### 1. Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models

1. Title: Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models
2. Year: 2025
3. Venue / status: ICLR 2025 Poster
4. URL: https://openreview.net/forum?id=lXE5lB6ppV
5. Relevance level: Near-match
6. Which sub-angle it belongs to: A
7. One-sentence summary: Shows that task-specific fine-tuning on datasets with clear ground-truth answers can undermine safety alignment, and proposes task-format-mimicking safety data as mitigation.
8. Why it matters for my question: This is the closest official conference paper to the hypothesis that fixed task format, prompting style, or answer schema can encourage shortcut-like safety behavior instead of robust boundary learning.
9. Collision risk: High
10. What I should borrow from it: Treat task format, prompt wrapper, and answer schema as explicit experimental variables.
11. What I must avoid copying: The main novelty cannot just be "task-specific FT hurts safety" plus a format-mimicking repair recipe.

### 2. Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning

1. Title: Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning
2. Year: 2025
3. Venue / status: arXiv preprint; later submitted to ICLR 2026
4. URL: https://arxiv.org/abs/2502.01116
5. Relevance level: Near-match
6. Which sub-angle it belongs to: A
7. One-sentence summary: Finds that benign instruction tuning can still degrade safety alignment, with answer structure, identity calibration, and role-play emerging as key factors.
8. Why it matters for my question: It directly supports the possibility that models latch onto structural cues in answers or wrappers rather than learning deeper safety criteria.
9. Collision risk: High
10. What I should borrow from it: Use answer structure and wrapper design as first-class ablation axes.
11. What I must avoid copying: Do not stop at a generic "benign fine-tuning hurts safety" story without isolating format-conditioned shortcutting.

### 3. Safety Depth in Large Language Models: A Markov Chain Perspective

1. Title: Safety Depth in Large Language Models: A Markov Chain Perspective
2. Year: 2025
3. Venue / status: NeurIPS 2025 Poster
4. URL: https://openreview.net/forum?id=tu3P6KSHGN
5. Relevance level: Near-match
6. Which sub-angle it belongs to: C
7. One-sentence summary: Formalizes safety depth as the output position at which the model reliably refuses harmful content, and studies how deeper alignment changes robustness.
8. Why it matters for my question: This is the cleanest official venue evidence that early-token alignment is a real mechanism, not just an intuition.
9. Collision risk: High
10. What I should borrow from it: Early-token position metrics, depth-oriented evaluation, and test-time analysis.
11. What I must avoid copying: Do not reuse the Markov-chain framing unless there is a genuine new theoretical contribution.

### 4. Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment

1. Title: Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment
2. Year: 2025
3. Venue / status: Submitted to ICLR 2026
4. URL: https://openreview.net/forum?id=7cecAmjinr
5. Relevance level: Near-match
6. Which sub-angle it belongs to: B
7. One-sentence summary: Shows that refusal behavior is highly sensitive to token-form drift, even when semantics are preserved.
8. Why it matters for my question: If safety depends on fixed token surface forms, then fixed prompt wrappers and schemas are plausible shortcut carriers.
9. Collision risk: High
10. What I should borrow from it: Form-preserving drift tests and cross-form robustness evaluation.
11. What I must avoid copying: Do not reduce the problem to simple perturbation robustness only.

### 5. Chain-of-Thought Hijacking

1. Title: Chain-of-Thought Hijacking
2. Year: 2025
3. Venue / status: ICLR 2026 Conference Desk Rejected Submission
4. URL: https://openreview.net/forum?id=kQRjcBmFAD
5. Relevance level: Near-match
6. Which sub-angle it belongs to: D
7. One-sentence summary: Demonstrates that benign-looking CoT padding plus final-answer cues can weaken safety checking and turn explicit reasoning into a jailbreak vector.
8. Why it matters for my question: It directly matches the concern that fixed CoT prefixes and reasoning wrappers can distort safety behavior.
9. Collision risk: Medium-High
10. What I should borrow from it: CoT-prefix length, benign reasoning padding, and final-answer cue ablations.
11. What I must avoid copying: Do not frame the entire project as a new jailbreak attack only.

## Bucket 2: Official Conference Papers / Official Pages That Can Anchor the Positioning

### 1. Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning

1. Title: Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning
2. Year: 2026
3. Venue / status: ICLR 2026 ConditionalOral
4. URL: https://openreview.net/forum?id=yfM2e8Icsw
5. Relevance level: Supporting-mechanism
6. Which sub-angle it belongs to: A
7. One-sentence summary: Shows that latent adversarial behaviors can remain dormant pre-finetuning and become activated after downstream fine-tuning.
8. Why it matters for my question: It strengthens the case that fine-tuning can reliably trigger non-obvious safety failures, including over-refusal.
9. Collision risk: Medium
10. What I should borrow from it: The idea of pre-existing latent behavior modes activated by adaptation.
11. What I must avoid copying: Do not pivot the project into a supply-chain or trojan paper.

### 2. A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space

1. Title: A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space
2. Year: 2026
3. Venue / status: ICLR 2026 Poster
4. URL: https://openreview.net/forum?id=887vde4ZAW
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: C
7. One-sentence summary: Proposes GuardSpace, which preserves safety during fine-tuning by isolating a safety-sensitive subspace and constraining harmful updates.
8. Why it matters for my question: It is a strong official mitigation baseline for layer protection and subspace-preserving adaptation.
9. Collision risk: Low
10. What I should borrow from it: Subspace freezing, null-space projection, and safety-preserving PEFT design.
11. What I must avoid copying: Do not directly replicate the same subspace decomposition story.

### 3. ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs

1. Title: ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs
2. Year: 2026
3. Venue / status: ICLR 2026 Poster
4. URL: https://openreview.net/forum?id=QkHKaPfRAB
5. Relevance level: Near-match
6. Which sub-angle it belongs to: C
7. One-sentence summary: Frames over-refusal as a representational overlap between harmful and pseudo-harmful inputs and mitigates it with projected low-rank pruning.
8. Why it matters for my question: This is one of the strongest official venue papers for the pseudo-harmful / over-refusal representation-space view.
9. Collision risk: Medium
10. What I should borrow from it: Pseudo-harmful vs harmful separation, layer selection, and over-refusal-focused metrics.
11. What I must avoid copying: Do not just build another low-rank pruning paper.

### 4. AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models

1. Title: AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models
2. Year: 2026
3. Venue / status: ICLR 2026 Poster
4. URL: https://openreview.net/forum?id=mIe17L3kWn
5. Relevance level: Near-match
6. Which sub-angle it belongs to: D
7. One-sentence summary: Identifies a snowball effect in safety CoT tuning and teaches models to self-correct harmful or over-cautious reasoning drifts.
8. Why it matters for my question: It is direct official evidence that reasoning-format tuning can create both harmful compliance and excessive refusal.
9. Collision risk: Medium
10. What I should borrow from it: Temptation-correction and hesitation-correction style supervision.
11. What I must avoid copying: Do not reuse the same adversarial CoT tuning pipeline as the central method.

### 5. PAFT: Prompt-Agnostic Fine-Tuning

1. Title: PAFT: Prompt-Agnostic Fine-Tuning
2. Year: 2025
3. Venue / status: EMNLP 2025 Main
4. URL: https://aclanthology.org/2025.emnlp-main.37/
5. Relevance level: Supporting-mechanism
6. Which sub-angle it belongs to: B
7. One-sentence summary: Shows that standard fine-tuning overfits to specific prompt wording and improves generalization via prompt diversification.
8. Why it matters for my question: It is a clean main-conference signal that fixed prompt wording and templates are genuine overfitting risks.
9. Collision risk: Medium
10. What I should borrow from it: Prompt diversification during safety fine-tuning.
11. What I must avoid copying: Do not relabel generic prompt robustness as safety novelty.

### 6. Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models

1. Title: Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models
2. Year: 2025
3. Venue / status: ICLR 2025 Poster
4. URL: https://openreview.net/forum?id=lXE5lB6ppV
5. Relevance level: Near-match
6. Which sub-angle it belongs to: A
7. One-sentence summary: Formalizes task-specific fine-tuning as a distinct safety risk setting and shows format-aware mitigation.
8. Why it matters for my question: It is the anchor paper for the task-specific / format-constrained fine-tuning risk angle.
9. Collision risk: High
10. What I should borrow from it: Task-format-aware safety mixture design.
11. What I must avoid copying: Do not copy the same threat model and mitigation framing.

## Bucket 3: Papers Explicitly Relevant to Early Tokens / Prompt Templates / Output Schema

### 1. PAFT: Prompt-Agnostic Fine-Tuning

1. Title: PAFT: Prompt-Agnostic Fine-Tuning
2. Year: 2025
3. Venue / status: EMNLP 2025 Main
4. URL: https://aclanthology.org/2025.emnlp-main.37/
5. Relevance level: Supporting-mechanism
6. Which sub-angle it belongs to: B
7. One-sentence summary: Dynamic prompt variation during fine-tuning reduces overfitting to specific prompt wording.
8. Why it matters for my question: Directly supports the claim that fixed wrappers and prompt phrasings can become spurious anchors.
9. Collision risk: Medium
10. What I should borrow from it: Prompt paraphrase pools and wrapper diversification.
11. What I must avoid copying: A pure prompt robustness benchmark without safety content.

### 2. Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning

1. Title: Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning
2. Year: 2025
3. Venue / status: arXiv preprint; later submitted to ICLR 2026
4. URL: https://arxiv.org/abs/2502.01116
5. Relevance level: Near-match
6. Which sub-angle it belongs to: A
7. One-sentence summary: Safety degradation after benign tuning is strongly affected by answer structure and response style.
8. Why it matters for my question: This is one of the strongest pieces of evidence that output schema itself can matter.
9. Collision risk: High
10. What I should borrow from it: Structured-answer and role-framing ablations.
11. What I must avoid copying: A broad survey-style result without a sharper format-specific mechanism.

### 3. Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment

1. Title: Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment
2. Year: 2025
3. Venue / status: Submitted to ICLR 2026
4. URL: https://openreview.net/forum?id=7cecAmjinr
5. Relevance level: Near-match
6. Which sub-angle it belongs to: B
7. One-sentence summary: Refusal collapses under semantics-preserving token-form changes, implying alignment remains surface-form sensitive.
8. Why it matters for my question: It is almost the direct "fixed token form shortcut" hypothesis in paper form.
9. Collision risk: High
10. What I should borrow from it: Token-form drift stress tests, cross-form robustness, and patch-then-break evaluation.
11. What I must avoid copying: Positioning the entire problem as adversarial perturbation only.

### 4. Safety Depth in Large Language Models: A Markov Chain Perspective

1. Title: Safety Depth in Large Language Models: A Markov Chain Perspective
2. Year: 2025
3. Venue / status: NeurIPS 2025 Poster
4. URL: https://openreview.net/forum?id=tu3P6KSHGN
5. Relevance level: Near-match
6. Which sub-angle it belongs to: C
7. One-sentence summary: Treats safety as a depth property over generated tokens rather than a single refusal token decision.
8. Why it matters for my question: It directly motivates your early-token safety-conflict framing.
9. Collision risk: High
10. What I should borrow from it: Multi-token refusal-depth metrics and test-time interventions.
11. What I must avoid copying: A purely theoretical repackaging of "deep safety alignment."

### 5. Deep Safety Alignment Requires Thinking Beyond the Top Token

1. Title: Deep Safety Alignment Requires Thinking Beyond the Top Token
2. Year: 2025
3. Venue / status: ICLR 2026 Conference Withdrawn Submission
4. URL: https://openreview.net/forum?id=GpBo97z6GH
5. Relevance level: Exact-match
6. Which sub-angle it belongs to: C
7. One-sentence summary: Argues that supposedly deep safety alignment can still shortcut onto a single refusal top token while harmful tokens remain high-probability in the distribution.
8. Why it matters for my question: This is the strongest direct critique of early-token refusal shortcuts and fixed-prefix-style safety.
9. Collision risk: High
10. What I should borrow from it: Top-k token mass analysis, prefilling stress tests, and distribution-level diagnostics.
11. What I must avoid copying: PRESTO or RAP-style method reuse as the core technical contribution.

### 6. Chain-of-Thought Hijacking

1. Title: Chain-of-Thought Hijacking
2. Year: 2025
3. Venue / status: ICLR 2026 Conference Desk Rejected Submission
4. URL: https://openreview.net/forum?id=kQRjcBmFAD
5. Relevance level: Near-match
6. Which sub-angle it belongs to: D
7. One-sentence summary: Explicit reasoning traces can themselves become the mechanism by which safety checks are diluted.
8. Why it matters for my question: This is directly relevant to fixed CoT prefixes and reasoning-format-specific failures.
9. Collision risk: Medium-High
10. What I should borrow from it: CoT-length, final-answer-cue, and attention-dilution analyses.
11. What I must avoid copying: Treating a jailbreak attack as equivalent to a format-conditioned alignment paper.

## Bucket 4: Mitigation / Method Papers Worth Borrowing From

### 1. A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space

1. Title: A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space
2. Year: 2026
3. Venue / status: ICLR 2026 Poster
4. URL: https://openreview.net/forum?id=887vde4ZAW
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: C
7. One-sentence summary: Preserves safety during adaptation by guarding safety-relevant parameter subspaces.
8. Why it matters for my question: Strong fit for layer protection and subspace-based mitigation if format-specific adaptation is the cause of failure.
9. Collision risk: Low
10. What I should borrow from it: Safety-preserving PEFT, guarded subspace updates, null-space constraints.
11. What I must avoid copying: The same decomposition + projector recipe.

### 2. Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment

1. Title: Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment
2. Year: 2025
3. Venue / status: EMNLP 2025 Main
4. URL: https://aclanthology.org/2025.emnlp-main.406/
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: A
7. One-sentence summary: Detects safety-degrading features in seemingly benign fine-tuning data and filters them using safety-sensitive layers.
8. Why it matters for my question: If fixed schemas or wrappers carry shortcut-rich cues, this provides a data-centric mitigation path.
9. Collision risk: Low
10. What I should borrow from it: Sample filtering, sample reweighting, and safety-sensitive representation screening.
11. What I must avoid copying: A simple data-cleaning paper with no shortcut mechanism analysis.

### 3. Representation Bending for Large Language Model Safety

1. Title: Representation Bending for Large Language Model Safety
2. Year: 2025
3. Venue / status: ACL 2025 Main
4. URL: https://aclanthology.org/2025.acl-long.1173/
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: C
7. One-sentence summary: Brings activation-steering ideas into loss-based fine-tuning to reshape harmful representations directly.
8. Why it matters for my question: Strong method inspiration for representation steering and subspace manipulation.
9. Collision risk: Low
10. What I should borrow from it: Representation-space intervention instead of only changing data mixtures.
11. What I must avoid copying: Presenting a generic jailbreak defense as if it solved the format-conditioning mechanism.

### 4. ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs

1. Title: ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs
2. Year: 2026
3. Venue / status: ICLR 2026 Poster
4. URL: https://openreview.net/forum?id=QkHKaPfRAB
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: C
7. One-sentence summary: Uses projected low-rank pruning to separate harmful and pseudo-harmful features and reduce false refusals.
8. Why it matters for my question: Strongest official over-refusal mitigation paper in the representation-space family.
9. Collision risk: Medium
10. What I should borrow from it: Pseudo-harmful/harmful disentanglement and over-refusal-focused internal diagnostics.
11. What I must avoid copying: The same pruning-centered solution.

### 5. Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models

1. Title: Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models
2. Year: 2025
3. Venue / status: Submitted to ICLR 2026
4. URL: https://openreview.net/forum?id=enpCeRYBhe
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: C
7. One-sentence summary: Separates boilerplate refusal statements from rationale and shows that rationale-only supervision reduces false refusals.
8. Why it matters for my question: This is extremely relevant if fixed refusal prefixes or templates are the shortcut carriers.
9. Collision risk: Medium
10. What I should borrow from it: Rationale-only or reduced-boilerplate supervision.
11. What I must avoid copying: A pure supervision-format paper without generalization or format stress tests.

### 6. Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models

1. Title: Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models
2. Year: 2025
3. Venue / status: ICLR 2025 Poster
4. URL: https://openreview.net/forum?id=lXE5lB6ppV
5. Relevance level: Mitigation-inspiration
6. Which sub-angle it belongs to: A
7. One-sentence summary: Uses safety data that mimics downstream task format and prompting style to restore safety.
8. Why it matters for my question: It is the clearest task-format-aware mitigation baseline.
9. Collision risk: High
10. What I should borrow from it: Downstream-format-aware safety data mixing.
11. What I must avoid copying: Making format-mimic safety mixing the only contribution.

## Final Judgment

### 1. Has the exact thesis already been fully done?

No. I did not find a paper that cleanly unifies:

- fixed early tokens
- fixed output schema / wrapper
- safety SFT or test-time adaptation
- shortcut learning
- over-refusal / selective-refusal collapse / abnormal safety-utility tradeoff

What exists instead is a close cluster spread across A, B, C, and D.

### 2. Top 5 papers to build around

1. Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models
2. Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning
3. Safety Depth in Large Language Models: A Markov Chain Perspective
4. Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment
5. Chain-of-Thought Hijacking

Strong alternate fifth anchor:

- Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning

### 3. Suggested problem names

- Format-Selective Safety Shortcutting
- Early-Token Safety Myopia under Format-Constrained Adaptation
- Schema-Conditioned Refusal Collapse
- Reasoning-Wrapper Safety Drift

## Recommended Positioning

The most defensible positioning is:

"Current safety alignment for LLM adaptation is partially format-bound: fixed early-token wrappers, answer schemas, and reasoning templates can induce shortcut safety behaviors that appear aligned on in-distribution formats but break as over-refusal, selective refusal collapse, or utility loss under format shift."

That framing is close to prior work, but not obviously fully occupied.
