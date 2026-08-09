# Sample RAG benchmark

This benchmark is a small, versioned evidence-level acceptance set for the complete document-qa pipeline. It contains six synthetic company documents and 30 gold questions:

- 24 answerable questions
- 6 unanswerable questions
- 2 multi-source questions
- numeric hard negatives such as 30-day, 14-day, and 7-day refund policies

The corpus is synthetic so it can be committed and evaluated without exposing private documents.

## Retrieval baseline

Run the production Docling, Hugging Face, and LanceDB retrieval path against a fresh temporary index:

```bash
uv run document-qa evaluate benchmarks/sample/dataset.json --fresh-index
```

Baseline measured with the locked project dependencies and default retrieval configuration:

| Metric | Result |
| --- | ---: |
| Hit@1 | 23/24 (95.8%) |
| Hit@3 | 24/24 (100.0%) |
| MRR@3 | 97.9% |
| Evidence recall@3 | 100.0% |

The only Hit@1 miss is `support-channels`: the `Response Targets` section ranks first and the correct `Service Hours` section ranks second. This is retained as a hard negative rather than tuned away.

## End-to-end evaluation

The following command additionally makes 30 configured chat-model calls and scores facts, citations, abstentions, and strict overall accuracy:

```bash
uv run document-qa evaluate benchmarks/sample/dataset.json \
  --fresh-index \
  --with-answers \
  --output rag-evaluation-report.json
```

The published OpenRouter baseline uses `openai/gpt-oss-120b` at temperature `0.0`:

| Metric | Result |
| --- | ---: |
| Strict RAG overall accuracy | 29/30 (96.7%) |
| All-facts accuracy | 23/24 (95.8%) |
| Fact recall | 30/31 (96.8%) |
| Citation precision | 100.0% |
| Citation recall | 100.0% |
| No-answer accuracy | 6/6 (100.0%) |
| Citation contract validity | 100.0% |

The one failed case, `return-shipping-charge`, contained the correct refund fact and citation but introduced an unsupported company name in another language. It remains failed instead of being hidden by a case-specific alias.

`dataset.json` pins the embedding model, prompt names, chunk size, retrieval depth, weights, and generation temperature so local `.env` overrides cannot silently change the protocol. `baseline.json` records the dataset, corpus, and `uv.lock` hashes alongside the measured metrics.

Published runs must use `--fresh-index`; an existing LanceDB can contain chunks from an older corpus revision even when filenames and section paths still match.

Generation scores use normalized lexical facts, explicit aliases, and subject-bound patterns from `dataset.json`; they are transparent and reproducible, but they are not a semantic or LLM-as-judge score. They do not prove that every additional claim is grounded. Provider and model drift can still change a new live run.
