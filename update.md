Yes — your instinct is basically right, but I would separate **two different problems**:

1. **Forcing the base model to emit exactly one reasoning step** is likely an unnatural generation objective and can hurt the quality of the candidates.
2. **Injecting the step classifier only at `<step>` boundaries** is a much cleaner formulation than adding the classifier to every token.

Your current code already has the beginnings of this idea in `MODE_TOKEN`, but it currently applies the policy to **every token**:

```python
base_log_probs = torch.log_softmax(base_logits, dim=-1)[candidate_ids]
policy_scores = self.policy.score_tokens(state, candidate_ids.tolist())
return combine_scores(base_log_probs, policy_scores, self.config.alpha)
```



What I would change is to make the **base model generate an unconstrained continuation**, and make the policy intervene only when the continuation reaches a step boundary. In other words, the policy should operate over **completed steps**, not over individual language-model tokens.

## The key idea

Suppose the base model defines

$$
p_\theta(y_{1:T}\mid x)
=
\prod_t p_\theta(y_t\mid x,y_{<t})
$$

and your classifier defines a score/probability for an entire candidate step

$$
q_\phi(s_i\mid x,h).
$$

Then what you really want is something like

$$
P(s_i\mid x,h)
\propto
P_\theta(s_i\mid x,h)
\;q_\phi(s_i\mid x,h)^\alpha.
$$

Taking logs:

$$
\log P(s_i)
=
\log P_\theta(s_i)
+
\alpha\log q_\phi(s_i)
+C.
$$

That's a perfectly sensible product-of-experts interpretation.

The important point is that **you should calculate this after the entire step has been generated**, rather than trying to distribute `log q_phi` across the individual tokens of the step.

---

# Why I wouldn't literally use assisted decoding

Hugging Face's `assistant_model` / assisted decoding is primarily designed for a **draft model proposing tokens that the target model verifies**, i.e. speculative decoding.

That's not quite your setup.

Your classifier isn't a smaller LM proposing tokens. It is a **step-level reward/policy model**. So conceptually you're closer to:

> constrained / guided beam search with a segment-level scorer

than to speculative decoding.

You can implement this using the Hugging Face generation machinery, but I wouldn't try to shoehorn the classifier into `assistant_model`.

The natural architecture is:

```text
                         ┌──────────────┐
                         │   Base LM    │
                         └──────┬───────┘
                                │
                         generates tokens
                                │
                ┌───────────────┴───────────────┐
                │                               │
             ordinary                        <step>
              token                             │
                │                               ▼
                │                      ┌────────────────┐
                │                      │ Step classifier│
                │                      └───────┬────────┘
                │                              │
                │                     score completed step
                │                              │
                └───────────────┬──────────────┘
                                ▼
                       beam / candidate selection
```

This is substantially closer to what I think you actually want.

---

# The length-normalization question is the important part

You specifically asked:

> how do I redistribute the classifier probability to the given step in such a way that it is independent of length and other factors that might skew the whole process?

I would **not distribute the classifier probability uniformly across tokens**.

For example, don't do:

$$
\log p_{\phi}(s)/T
$$

and add that to every token.

There are two reasons.

First, it changes the effective temperature depending on step length.

Second, it makes the classifier act like a token-level LM despite the fact that it isn't one.

Instead, keep the two quantities at their natural granularities.

### Base-model score

For candidate step \(s = y_1,\ldots,y_T\):

$$
L_\theta(s)
=
\sum_{t=1}^T
\log p_\theta(y_t\mid x,y_{<t}).
$$

Then normalize **the LM score**, not the classifier score:

$$
\bar L_\theta(s)
=
\frac{1}{T^\beta}
\sum_t\log p_\theta(y_t\mid x,y_{<t}).
$$

Here β is a tunable length-normalization parameter.

Common choices are:

* β = 0 → ordinary sequence log probability
* β = 1 → average token log probability
* 0 < β < 1 → compromise

Then your step score becomes

$$
S(s)
=
\bar L_\theta(s)
+
\alpha \log q_\phi(s).
$$

That's the cleanest formulation.

The classifier contributes **one scalar per step**.

---

# Even better: don't make the classifier compete with LM probability at all

There's another formulation I think is particularly attractive for your setup.

Suppose your classifier estimates

$$
q_\phi(\text{correct step}\mid x,h,s).
$$

Then use it as a **reranker among beams/candidates**, rather than trying to interpret it as another probability distribution over tokens.

For each completed step:

```text
candidate             base LM score       classifier
-----------------------------------------------------
7 * 6 = 42                -4.2              .97
7 + 6 = 13                -3.1              .01
6 * 6 = 36                -3.8              .02
```

Then rank using something like

$$
S(s)
=
\frac{\log P_\theta(s)}{T^\beta}
+
\alpha \operatorname{logit}(q_\phi(s)).
$$

I actually prefer **logit** over `log(probability)` here:

$$
S(s)
=
\bar L_\theta(s)
+
\alpha\,
\operatorname{logit}(q_\phi(s)).
$$

Why?

If your classifier says:

```text
0.50
0.90
0.99
```

the differences in `log probability` are very asymmetric and probabilities near 1 get compressed.

The logit gives:

```text
0
2.20
4.60
```

which behaves much more like a learned energy/reward.

---

# But there's an even bigger issue with your current architecture

Your current `_next_step_candidates()` does:

```python
candidates = self._sample_candidates(...)
scores = self._score_candidates(...)
best = candidates[order[0]]
```



So you're sampling **K complete steps**, then throwing away K−1 of them.

That means the classifier has no influence over *which prefixes the base model explores*.

Imagine:

```text
Base LM
│
├── "We multiply 7 by 6..."
│       └── 7*6=42
│
├── "First add..."
│       └── 7+6=13
│
├── "Since..."
│       └── ...
│
└── ...
```

If the good trajectory isn't among the K samples, your classifier can never recover it.

This is exactly where **beam search with boundary-level scoring** becomes interesting.

---

# What I would implement

I'd use a **segment-level beam search**.

Instead of:

```text
generate step
→ classify
→ throw away everything except best
→ generate next step
```

maintain B partial trajectories:

```text
beam 1: question → step 1
beam 2: question → step 1'
beam 3: question → step 1''
...
```

Each beam generates normally until it produces `<step>`.

At that point:

1. Extract the completed step.
2. Calculate its base-model log probability.
3. Run the classifier.
4. Combine the scores.
5. Keep the top B trajectories.
6. Continue generation from those B trajectories.

Conceptually:

```text
                    base LM
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          step A     step B     step C
             │         │         │
          policy     policy     policy
             │         │         │
             └─────────┼─────────┘
                       │
                  top B beams
                       │
                 continue LM
```

This gives your classifier exactly the role you're describing.

---

# Crucially, let the LM generate naturally inside a step

I would also **stop telling the model "write ONE step, then stop."**

Instead, give it a natural format such as:

```text
Solve the problem by writing a sequence of reasoning steps.
Separate reasoning steps with <step>.
After the final reasoning step, write:
Final answer: ...
```

Then the LM is solving normally:

```text
We have 7 groups of 6.
<step>
7 * 6 = 42.
<step>
Final answer: 42
```

The decoder, rather than the system prompt, decides when the segment ends.

This is a subtle but important distinction.

Your current prompt explicitly asks the LM to behave as a one-step generator:

> "write ONE step, then the marker `<step>`, then stop"

That's likely creating exactly the OOD behavior you're concerned about.

The decoder can enforce the segmentation without requiring the model to be trained to do that behavior.

---

# Hugging Face implementation

You don't actually need to modify the model.

You can use the normal HF generation primitives to obtain token-level log probabilities, while maintaining your own beam state.

The conceptual loop is:

```python
beams = [initial_state]

while not finished:

    next_beams = []

    for beam in beams:

        # Generate ordinary LM continuation.
        continuations = generate_until_step_boundary(
            model,
            beam.input_ids,
        )

        for continuation in continuations:

            step = extract_step(continuation)

            lm_score = sequence_logprob(
                continuation
            )

            policy_score = classifier(
                question,
                beam.history,
                step,
            )

            score = (
                lm_score / len(continuation) ** beta
                + alpha * policy_score
            )

            next_beams.append(
                Beam(
                    tokens=...,
                    history=...,
                    score=score,
                )
            )

    beams = top_k(next_beams, beam_width)
```

This is essentially **beam search over segments rather than tokens**.

---

# A useful variation: token beam search with delayed policy evaluation

You can go one step further.

Instead of making each beam generate a whole step independently, maintain ordinary token-level beams.

Then:

```text
token token token token <step>
                         ↑
                         │
                    policy fires
```

At `<step>`:

```python
step_score = classifier(step)
```

and then modify the accumulated beam score:

```python
beam.score += alpha * classifier_score
```

Then prune beams.

That means the classifier is literally interacting with beam search, but only at boundaries.

This sounds very close to what you're asking for.

The resulting objective is approximately:

$$
S(y_{1:T})
=
\sum_t \log p_\theta(y_t|y_{<t})
+
\sum_{k}
\alpha f_\phi(s_k).
$$

where each \(s_k\) is a completed reasoning step.

This has a very nice interpretation:

> the LM determines linguistic/token-level plausibility; the classifier supplies an additional reward whenever a semantic unit (a reasoning step) has been completed.

---

# One caveat: beam search and length normalization

If you're doing this, I'd be careful about **two different lengths**.

There is:

1. token length within the current step
2. number of completed reasoning steps

You don't necessarily want to normalize the classifier by either.

For example:

```text
Step A:  "7 * 6 = 42"
Step B:  "Therefore the answer is 42 because..."
```

The classifier's output should mean:

```text
How good is this step?
```

regardless of how many tokens it contains.

So I'd use:

$$
S_{\text{beam}}
=
L_\theta
+
\alpha\sum_i f_\phi(s_i)
$$

and handle LM length bias separately:

$$
L_\theta
=
\frac{\sum_t \log p_\theta(y_t)}
{T^\beta}.
$$

I would **not** divide \(f_\phi(s_i)\) by the number of tokens in \(s_i\).

---

# An important alternative: calibrate the classifier

There's one issue with

$$
\log P_\theta + \alpha \operatorname{logit}(q_\phi)
$$

that is easy to overlook.

The classifier's raw probability isn't necessarily calibrated against LM log probabilities.

For example:

```text
LM sequence score:      -12
classifier logit:        +4
```

doesn't mean the classifier should contribute exactly `+4`.

You can learn the scale.

A simple approach is:

$$
S =
L_\theta +
\alpha f_\phi
$$

and tune α on a held-out validation set.

Even better, fit a temperature/scale for the classifier:

$$
f_\phi(s)=
\frac{\operatorname{logit}(q_\phi(s))}{\tau_\phi}.
$$

Then α controls the overall strength.

Your existing `alpha` already provides exactly this kind of knob; currently you're applying it to candidate scores at the reranking level. 

---

# I would also change your classifier's training target

If your classifier is currently trained simply as:

```text
step → correct / incorrect
```

you might get a better signal by training it as a **relative preference model**:

```text
Given:
    question
    previous steps
    candidate A
    candidate B

which step is preferable?
```

Then it learns:

$$
f_\phi(s^+) > f_\phi(s^-)
$$

rather than having to produce an absolutely calibrated probability.

That's especially well suited to beam search because beam search itself only needs to know:

> which of these partial trajectories should survive?

You can then use:

$$
S(s)=
\bar L_\theta(s)+\alpha f_\phi(s)
$$

without pretending that `f` is literally a probability.

---

# There's a particularly clean formulation for your problem

I would model your decoder as a **hierarchical search**:

$$
\boxed{
\text{token LM}
\quad\rightarrow\quad
\text{step segmentation}
\quad\rightarrow\quad
\text{step policy}
}
$$

rather than trying to turn the classifier into a token LM.

At the token level:

$$
p_\theta(token\mid prefix)
$$

At the step level:

$$
q_\phi(step\mid question,history)
$$

Then:

$$
\boxed{
S(\text{trajectory})
=
\underbrace{
\frac{1}{T^\beta}
\sum_t \log p_\theta(y_t|y_{<t})
}_{\text{language-model likelihood}}
+
\alpha
\underbrace{
\sum_i f_\phi(s_i)
}_{\text{step-level guidance}}
}
$$

That is IMO much closer to the mathematical object you're actually trying to build.

---

## And your existing code is already very close

Your `MODE_TOKEN` implementation currently does:

```text
every token
    ↓
base logits
    +
policy token scores
    ↓
next token
```



I'd instead change the architecture to:

```text
every token
    ↓
base LM only
    ↓
<step>
    ↓
freeze segment
    ↓
classifier(step)
    ↓
modify beam score
    ↓
beam pruning
    ↓
resume LM
```

So the classifier does **not** need to provide probabilities for vocabulary tokens at all.

That is probably the biggest conceptual simplification.

### My recommended experiment

I'd test these four conditions:

| Decoder | Base LM           | Step classifier                        |
| ------- | ----------------- | -------------------------------------- |
| A       | normal generation | none                                   |
| B       | normal generation | sample K steps + rerank                |
| C       | normal generation | segment-level beam search              |
| D       | normal generation | segment-level beam + Lean verification |

The critical comparison is **B vs C**. If C substantially outperforms B at the same token budget, that would support your hypothesis that the current problem is not primarily the classifier but the fact that you're asking the LM to repeatedly restart generation at artificial step boundaries.

And importantly, you don't need to make the model itself output only one step. **Make `<step>` a decoder-level segmentation event, not a behavioral instruction to the model.**

Your existing implementation already explicitly separates the token-guidance stage from the candidate/reranking stages, so this would be a fairly natural evolution of the architecture rather than a complete rewrite. 

