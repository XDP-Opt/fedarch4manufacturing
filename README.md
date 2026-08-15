# Federated learning in manufacturing — replication material

Data and code behind two studies on federated learning (FL) in manufacturing:

1. a **systematic literature review** of academic papers and company sources, screened
   in stages and classified along four dimensions;
2. a **factorial survey**, conducted as vignette-based interviews.

```
literature_study/
    academic sources/
        abstract_screening.py        the LLM title/abstract screening stage
        corpus.csv                   bibliographic metadata for the screened papers
        screening_results/
            abstract/                stage 1 results, per model and combined
            fullpaper/               stage 2 results
        classification/              coding of the included papers
    company sources/
        Company_Review.csv           review of commercial FL offerings
factorial_survey/
    vignettes/                       the survey instrument
```

The interview responses are not included here. They are pseudonymised personal data,
and are held back pending clarification of what the participant consent covers. The
survey instrument below is complete, so the design is fully documented either way.

## Literature review — academic sources

### Funnel

| Stage | Papers | Method |
|---|---|---|
| Retrieved and de-duplicated | 1167 | database search, managed in Zotero |
| Passed title/abstract screening | 226 | LLM ensemble, three models, majority vote |
| Passed full-text screening | 38 | manual assessment |
| Classified | 38 | manual coding against a four-dimension schema |

Every record is identified by its **Zotero item key**, an 8-character code such as
`3KAIZA5P`. The same key is used in every file here, so any decision can be traced from
the corpus through both screening stages to the final coding. The three sets nest
strictly: the 38 classified papers are a subset of the 226, which are a subset of
the 1167.

### Corpus

`academic sources/corpus.csv` holds one row per screened paper — `Key` plus title,
authors, year, venue, DOI, URL and identifiers.

**Abstracts are deliberately not included.** They are copyright of their publishers and
are not the authors' to redistribute. 1165 of the 1167 records carry a DOI or URL, so
the abstracts can be retrieved from the publishers through your own access.

### Stage 1 — title/abstract screening

[`abstract_screening.py`](literature_study/academic%20sources/abstract_screening.py) reads the corpus from
a Zotero collection and asks a locally hosted LLM, one abstract per call, whether a
paper should be carried forward. It is deliberately lenient: a paper wrongly kept costs
one extra read at the next stage, whereas a paper wrongly dropped is never looked at
again. Only the two criteria an abstract can support a judgement on are applied (E2 and
E5); the rest require the full text.

The stage was run once per model for three models:

| File | Include | Exclude |
|---|---|---|
| `screening_results_total.qwen2.5-14b.csv` | 295 | 872 |
| `screening_results_total.ministral-3-14b.csv` | 244 | 923 |
| `screening_results_total.llama3.1-8b.csv` | 117 | 1050 |

Columns: `id` (Zotero item key), `decision`, `exclusion_criterion`, `confidence`, `reasoning`.

The three were combined by majority vote into `screening_results_total.csv`, which keeps
every model's individual answer next to the agreed one, so any decision can be traced
back to the votes behind it:

| Column | Meaning |
|---|---|
| `decision`, `confidence`, `exclusion_criterion` | the agreed values |
| `*__agree_n` | how many models supported the agreed value |
| `n_models`, `unanimous` | votes cast, and whether they were unanimous |
| `decision__<model>` and friends | each model's own answer |

935 of 1167 decisions were unanimous; 226 papers were carried forward.

### Stage 2 — full-text screening

`screening_results/fullpaper/screening_results_total.csv` records the manual assessment
of those 226 papers: `item_key`, `decision`, `exclusion_criteria_met`. 38 were included.

### Exclusion criteria

The numbering has a gap at I1–I3; those were dropped during protocol development and the
surviving ids keep their original numbers.

| Id | Criterion | Applied at |
|---|---|---|
| E1 | Pure algorithm development without application context or implementation or discussion | full text |
| E2 | FL application outside manufacturing (healthcare, finance, etc.) unless transferable models to manufacturing are explicitly discussed | abstract, full text |
| E3 | Low quality content | full text |
| E4 | Purely legal/regulatory analysis without business model implications | full text |
| E5 | Literature review/survey without own contribution | abstract, full text |
| E6 | Source not accessible, orphaned, or not clearly verifiable | full text |
| E7 | Presents no practical use-case or application of its own; only cites or reviews other use-cases | full text |

### Classification

The 38 included papers were coded against `classification/class_schema.json` along four
dimensions, each with an explicit "Not specified" value so an absent characteristic is
recorded rather than guessed:

| Dimension | Values |
|---|---|
| Aggregator type | Lead Organization Network, Self-Governed Network, Network Administrative Organization |
| Business relationship | Collaborative cause, Neutral, Competitor |
| Incentive mechanism | Compensation, No incentive, Fee |
| Deployment context | Academic Lab Experiment, Productive Industrial Environment |

`classification/paper_classification.csv` holds one row per paper, keyed by Zotero item
key, plus the raw coding tags.

## Literature review — company sources

`company sources/Company_Review.csv` reviews 75 commercial FL offerings against an
adapted form of the same exclusion criteria: `Unternehmen`, `Link`, `Ausschlussgrund`.
Six were carried forward; 69 were excluded, most often as FL outside manufacturing
(E2, 38) or as presenting no use-case of their own (E7, 17). Exclusion reasons are
recorded in German.

## Factorial survey

Nine vignettes were presented to nine participants from manufacturing companies, each
rated against the same seven questions. The instrument is given in full below; the
responses are not part of this repository.

### Instrument — `factorial_survey/vignettes/`

| File | Contents |
|---|---|
| `factors.csv` | the nine factor combinations, one per vignette |
| `vignettes.csv` | the nine vignette texts as presented (German) |
| `survey.csv` | the seven questions asked about each vignette (German) |

Each vignette varies six factors, three of which mirror the dimensions used to classify
the literature:

| Factor | Levels |
|---|---|
| Data sensitivity | Low, Medium, High |
| Aggregator | Lead Organization Network, Network Administrative Organization, Self-Governed Network |
| Number of participants | 5, 10, 20+ |
| Business relationship | collaborative cause, competitors |
| Expected increase in model performance | small, medium, large |
| Incentive model | No Payment, compensation per contribution, subscription fee |

## Running the screening script

Requires Python 3.13+ and a running [Ollama](https://ollama.com) server holding the model
you want to screen with.

```bash
uv sync
cp .env.example .env    # then fill in your Zotero credentials
uv run python "literature_study/academic sources/abstract_screening.py" --model qwen2.5:14b
```

| Flag | Effect |
|---|---|
| `--model` | which Ollama model to screen with |
| `--limit N` | screen at most N abstracts, for a short trial run |
| `--collection` | Zotero collection key, overriding `ZOTERO_COLLECTION` |
| `--output` | base output path; the model name is inserted into it |

Results are appended one row at a time and a re-run skips ids already present in the
output file, so an interrupted run can simply be started again.

Reproducing the published numbers exactly requires access to the Zotero library the
corpus lives in. The results in this repository are the output of that run and can be
read on their own.

## Licensing

[`abstract_screening.py`](literature_study/academic%20sources/abstract_screening.py) is
licensed under the [MIT License](LICENSE).
