# Article Evaluation Benchmark Protocol

## Purpose

The final article evaluation uses a 100-case controlled benchmark designed to evaluate the proposed Tool Interaction Manager across the tool-interaction behaviors claimed by the architecture.

The benchmark is not presented as a large-scale educational dataset. It is a controlled prototype-level benchmark designed to test whether the system can analyze a learner request, collect context, plan tool use, select appropriate tools, execute them, validate outputs, recover from failure, save traces, and generate a final tutoring answer.

## Relationship to the 30-case pilot

The original 30-case dataset was used as a pilot and regression set during thesis development. It is retained internally to verify continuity after implementation alignment. The official article evaluation is based on the 100-case benchmark.

## Benchmark structure

The benchmark contains 100 cases organized into five equally represented categories:

1. Conceptual grounded tutoring — 20 cases
2. Computation and verification — 20 cases
3. Visual and step-by-step tutoring — 20 cases
4. Code-related tutoring — 20 cases
5. Recovery, personalization, and trace-aware scenarios — 20 cases

This structure was selected to align the evaluation with the implemented tool capabilities and the claimed contribution of the architecture.

## Category rationale

### Conceptual grounded tutoring

This category evaluates whether the system can produce concept explanations supported by retrieval-based grounding. It primarily tests the use of the RAGTool for tutoring concepts such as convolution, kernels, stride, padding, feature maps, and matrix multiplication.

### Computation and verification

This category evaluates deterministic tool use. It tests whether the system can select computation tools for arithmetic, matrix multiplication, dimension checking, convolution output-size calculations, and simple convolution operations.

### Visual and step-by-step tutoring

This category evaluates whether the system can select visual support when the learner request requires illustration, step-by-step explanation, or comparison between visual procedures such as kernel sliding and row-by-column multiplication.

### Code-related tutoring

This category evaluates whether the system can generate, explain, or debug code with tool support. It focuses on matrix multiplication, convolution, output-size calculation, and code-level misconceptions.

### Recovery, personalization, and trace-aware scenarios

This category evaluates the main contribution of the proposed architecture: validation, failure recovery, trace use, and memory-aware tutoring. It includes both natural trace-aware prompts and controlled recovery prompts using forced failure conditions.

## Justification of 100 cases

The benchmark size was selected for coverage and feasibility. A total of 100 cases allows a balanced distribution of 20 cases per category and makes percentage-based metrics directly interpretable, since each case corresponds to one percentage point. The benchmark extends the initial 30-case pilot and provides broader coverage of the main tool-interaction scenarios targeted by the system.

## Evaluated systems

All evaluated systems must run on the same 100 cases:

1. Direct LLM baseline
2. Basic Tool Use baseline
3. Manager Without Recovery baseline
4. Full Tool Interaction Manager

## Backend metrics

The backend evaluation reports:

- Workflow completion rate
- Tool selection precision
- Tool selection recall
- Tool selection F1
- Tool execution success rate
- Output validation success rate
- Recovery use rate
- Recovery success rate
- Trace save rate
- Trace completeness
- Memory update rate
- Final answer generation rate
- Topic consistency rate where applicable

## LLM-as-a-Judge protocol

After backend outputs are generated for all systems, answer quality is evaluated using an independent GPT-4-class judge.

The judge uses a pointwise rubric-based protocol. Each answer is evaluated independently without revealing the identity of the system that produced it. The judge receives the learner question, task category, expected behavior, generated answer, and the rubric.

The judge assigns scores from 1 to 5 for:

- Correctness
- Grounding quality
- Tool evidence use
- Pedagogical clarity
- Personalization
- Applicability
- Overall quality

Scores are reported globally and per category.

## Limitations

The 100-case benchmark improves coverage compared with the initial 30-case pilot, but it remains a controlled prototype-level evaluation. It does not replace large-scale classroom deployment or human learner evaluation. LLM-as-a-Judge evaluation is also treated as an automated quality estimate, not as a substitute for human expert judgment.
