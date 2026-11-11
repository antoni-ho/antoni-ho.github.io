# Antoni — Data Analyst Portfolio

The source of [antoni-ho.github.io](https://antoni-ho.github.io/): a static site with no
build step, no frameworks and no client-side dependencies. Each project lives in its own
folder and is self-contained — code, data, figures, README and a written case study.

## Projects

| # | Project | Question it answers | Stack |
| --- | --- | --- | --- |
| 01 | [AI app review sentiment](projects/01-ai-app-reviews/) | What are ChatGPT, Gemini and Claude users actually complaining about? | Python · pandas · VADER · matplotlib |
| 02 | [Used-car value retention](projects/02-used-car-market/) | Which used cars hold their value, and what does the market really charge for? | Python · pandas · scikit-learn · matplotlib |

## Structure

```
.
├── index.html                    landing page: hero, about, project cards, contact
├── style.css                     shared styles for every page
├── projects/
│   ├── 01-ai-app-reviews/
│   │   ├── index.html            full case study
│   │   ├── README.md             method, results, limitations
│   │   ├── *.py                  scrape → clean → score → visualise
│   │   ├── *.csv                 raw, cleaned and scored review data
│   │   └── images/               300 dpi figures + web-optimised copies
│   └── 02-used-car-market/
│       ├── index.html            full case study
│       ├── README.md             method, results, limitations
│       ├── METHODOLOGY.md        sampling design and the two judgement calls
│       ├── *.py                  enumerate → scrape → parse → clean → audit → eda → verify → model
│       ├── data/                 cleaned dataset, predictions, generated reports
│       └── images/               the eight figures used on the page
├── README.md
└── .gitignore
```

## Running it locally

There is nothing to install for the site itself — open `index.html` in a browser, or serve
the folder:

```bash
python -m http.server 8000
```

Each project folder has its own README with the commands to reproduce its analysis from
scratch. Raw crawl output is deliberately excluded from git: project 02 carries its cleaned
dataset (`data/clean.csv.gz`) and can regenerate the raw data with two commands.

## Editing conventions

- One shared `style.css`; project pages link to it with `../../style.css`.
- Figures are referenced relatively (`images/…`) so a project folder stays portable.
- Code comments and console output are in English; the published write-ups are in English.
