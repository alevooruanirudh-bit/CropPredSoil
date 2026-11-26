# Smart Agriculture AI — The Complete Beginner's Guide

> **Who this document is for:** someone who has never written a line of code, never opened a
> terminal, and maybe has never even used a laptop before. Every idea is explained from zero —
> what a file is, what a program is, what "installing a dependency" means, how a website talks
> to a server, what a machine-learning model actually computes, and why this specific project is
> built the way it is. If you read this whole document carefully, you should be able to rebuild
> this project from scratch yourself, and understand *every single line* of code in it.

This project is a **crop recommendation system**. You type in some numbers that describe a
farm's soil and weather (how much Nitrogen is in the soil, how hot it is, how much it rains,
etc.), and the program tells you which crop is most likely to grow well there, why, and what to
fix if it won't. It can also look at a **photo of soil** and guess what type of soil it is
(Alluvial, Black, Clay, or Red — the four major Indian soil types).

Under the hood it is two separate programs that talk to each other over the network, exactly
like your web browser talks to a website:

1. **The backend** — a Python program that holds the "brain" (the trained AI models) and answers
   questions sent to it. Think of it as a very fast, very specialized calculator sitting on a
   server.
2. **The frontend** — a webpage (written in JavaScript/React) that you actually see and click on
   in your browser. It doesn't do any "thinking" itself — it just draws boxes and buttons on
   screen and forwards your input to the backend, then draws the backend's answer nicely.

Everything below explains, in order, what a computer program even is, how to get this exact
project running on your machine, and then walks through **every file** in the repository
explaining not just *what* it does but *why* it's written that way — including the underlying
math and machine-learning theory.

---

## Table of contents

1. [Part 0 — What is a computer program, really?](#part-0--what-is-a-computer-program-really)
2. [Part 1 — The big picture: how this project is shaped](#part-1--the-big-picture-how-this-project-is-shaped)
3. [Part 2 — Installing everything from a blank computer](#part-2--installing-everything-from-a-blank-computer)
4. [Part 3 — Running the project](#part-3--running-the-project)
5. [Part 4 — Project map: every file, one sentence each](#part-4--project-map-every-file-one-sentence-each)
6. [Part 5 — Python crash course (using real code from this project)](#part-5--python-crash-course-using-real-code-from-this-project)
7. [Part 6 — JavaScript, React and HTML crash course](#part-6--javascript-react-and-html-crash-course)
8. [Part 7 — The backend, file by file](#part-7--the-backend-file-by-file)
9. [Part 8 — Machine learning theory, from zero](#part-8--machine-learning-theory-from-zero)
10. [Part 9 — The crop model, line by line (`ml/crop_model.py`)](#part-9--the-crop-model-line-by-line-mlcrop_modelpy)
11. [Part 10 — Explaining the AI's decision (`ml/explainability.py`)](#part-10--explaining-the-ais-decision-mlexplainabilitypy)
12. [Part 11 — The hybrid recommender for 182 crops (`ml/hybrid_recommender.py`)](#part-11--the-hybrid-recommender-for-182-crops-mlhybrid_recommenderpy)
13. [Part 12 — Suitability checks and remedies (`ml/suitability.py`)](#part-12--suitability-checks-and-remedies-mlsuitabilitypy)
14. [Part 13 — The soil photo classifier: a neural network (`ml/soil_model.py`)](#part-13--the-soil-photo-classifier-a-neural-network-mlsoil_modelpy)
15. [Part 14 — The 182-crop reference catalog](#part-14--the-182-crop-reference-catalog)
16. [Part 15 — The frontend, file by file](#part-15--the-frontend-file-by-file)
17. [Part 16 — The full API reference](#part-16--the-full-api-reference)
18. [Part 17 — Training your own models from scratch](#part-17--training-your-own-models-from-scratch)
19. [Part 18 — Automated tests](#part-18--automated-tests)
20. [Part 19 — Troubleshooting](#part-19--troubleshooting)
21. [Part 20 — Glossary of every technical word used above](#part-20--glossary-of-every-technical-word-used-above)

---

## Part 0 — What is a computer program, really?

Skip this part if you've coded before. If you have genuinely never touched a computer, read
every word.

**A computer only understands numbers.** Everything you see — letters, pictures, sounds — is
secretly stored as numbers. A *program* (also called *code* or *software*) is a very precise,
very literal list of instructions, written in a language the computer can translate into those
numbers, telling the computer exactly what to do, in exactly what order. Computers do not guess
or use "common sense" — they do precisely what the instructions say, nothing more, nothing less.
If the instructions are wrong, the result is wrong, even if it's wrong in a way a human would
never be confused by.

**A file** is just a named box of information saved on your disk (your hard drive/SSD — the
permanent storage inside your computer, as opposed to RAM, which is temporary and empties when
you turn the power off). A code file is a plain text file — you could open it in any text
editor — that happens to follow the strict grammar ("syntax") of a particular *programming
language*.

**A folder** (also called a *directory*) is a container that groups files together, and can
contain other folders inside it, forming a tree. This whole project lives inside one folder
called `crop prediction AI`, and everything discussed below is a file or folder inside it.

**A programming language** is one specific set of grammar rules for writing instructions. This
project uses two languages:

- **Python** — used for the backend (the "brain": the web server and the AI models). Python is
  popular for AI/data work because it reads almost like plain English and has enormous
  pre-built libraries for math and machine learning.
- **JavaScript** (in a dialect called **JSX**, via the **React** library) — used for the
  frontend (the webpage you look at). JavaScript is the only language web browsers can run
  natively, so essentially all interactive websites use it.

**Running a program** means telling the computer "read this file's instructions and carry them
out, right now, from top to bottom." You do this either by double-clicking something, or — far
more commonly for developers — by typing a command into a **terminal**.

**The terminal** (also called the *command line*, *shell*, or on Mac specifically the app named
"Terminal") is a plain text window where, instead of clicking icons, you *type* the name of a
program and press Enter to run it. It feels intimidating at first purely because it's unfamiliar
— it is not more difficult than clicking icons, it's just a different way of giving the same
instructions. Every "run this command" instruction in this README means: open the Terminal app,
type the exact text shown, and press the Enter/Return key.

**Dependencies / libraries / packages** — nobody writing a new program starts from absolute zero.
If ten thousand programmers before you already wrote and tested code that does "read a CSV
spreadsheet file" or "train a decision tree," you download and reuse their code instead of
rewriting it yourself. Each such reusable bundle of pre-written code is called a **library**,
**package**, or **dependency** (it's called a "dependency" because your program *depends on* it
existing to work). "Installing a dependency" simply means downloading that bundle of code onto
your computer so your own program is allowed to use it. This project depends on roughly 15
different Python libraries and 4 JavaScript libraries — [Part 2](#part-2--installing-everything-from-a-blank-computer)
walks through installing every one of them.

---

## Part 1 — The big picture: how this project is shaped

Modern interactive software (almost every app or website you use) is split into two
cooperating halves:

```
┌─────────────────────────┐        HTTP requests        ┌──────────────────────────┐
│        FRONTEND         │  ───────────────────────►   │         BACKEND          │
│  (runs in your browser) │                              │  (runs on a server /     │
│  React + JavaScript     │  ◄───────────────────────    │   your own machine here) │
│  "What you see"         │        JSON responses         │  Python + Flask + AI     │
└─────────────────────────┘                              └──────────────────────────┘
```

- The **backend** is the part that does the actual thinking. It holds the trained AI models in
  memory (loaded from files on disk when the server starts), and exposes a small set of "doors"
  called **API endpoints** — specific web addresses that accept a question and return an answer.
  In this project the backend is a Python **Flask** web server.
- The **frontend** is the part that draws the visual page: input boxes, buttons, colored cards.
  It has *zero* intelligence of its own — when you click "Analyze," it packages your numbers into
  a small text message, sends that message to the backend, waits, and then draws whatever answer
  comes back. In this project the frontend is a **React** app served during development by a
  tool called **Vite**.
- They talk to each other using **HTTP** (HyperText Transfer Protocol — the same protocol your
  browser uses to load every website) and they exchange data formatted as **JSON**
  (JavaScript Object Notation — a simple, universal text format for structured data that looks
  like `{"name": "rice", "confidence": 0.91}`; almost every API on the internet speaks JSON).

Because they are two separate programs, you run **two servers at once**, in two separate
terminal windows, while you develop:

- One terminal runs the Python backend on `http://127.0.0.1:5001`
  (`127.0.0.1` — pronounced "localhost" — is a special address that always means "this same
  computer"; `5001` is the **port number**, essentially an apartment number that tells your
  computer which of many programs listening on the network should receive the message).
- One terminal runs the frontend dev server on `http://localhost:5173`.

You then open `http://localhost:5173` in your web browser. The frontend dev server (Vite) is
configured to automatically forward (**proxy**) any web address starting with `/api` to the
backend at port 5001, so the browser only ever has to know about one address. That forwarding
rule lives in [`frontend/vite.config.js`](frontend/vite.config.js) — you'll see it again in
[Part 15](#part-15--the-frontend-file-by-file).

**Why not just one program?** Separating them means the visual design can be changed completely
without touching the AI code, and the AI/backend can be tested, reused, or swapped independently
of the webpage. It is the standard shape of essentially all modern web applications.

---

## Part 2 — Installing everything from a blank computer

You need two separate pieces of underlying software installed on your computer before *any* of
this project's own code can run: **Python** (to run the backend) and **Node.js** (to run the
frontend). Think of Python and Node.js as translators — your computer's processor doesn't
understand Python or JavaScript directly; these programs read your `.py` / `.js` files and
translate/execute them on the fly.

### 2.1 Install Python

1. Go to `python.org/downloads` in your browser and download the installer for your operating
   system (this project needs **Python 3.11**; some dependencies here — `pandas`, `numpy` — do
   not yet support the very newest Python release, and Python 3.10 is too old for them, so 3.11
   is the sweet spot for this specific project).
2. Run the installer. On Windows, make sure to tick "Add Python to PATH" during setup — this
   makes the word `python` typeable from any terminal window.
3. Confirm it worked: open a terminal and run:
   ```bash
   python3 --version
   ```
   You should see something like `Python 3.11.x` printed back.

### 2.2 Install Node.js (which includes `npm`)

1. Go to `nodejs.org` and download the "LTS" (Long Term Support — the stable, recommended)
   version.
2. Run the installer, accepting the defaults.
3. Confirm it worked:
   ```bash
   node --version
   npm --version
   ```
   `npm` (Node Package Manager) is the tool that downloads JavaScript dependencies — it is
   installed automatically alongside Node.js.

### 2.3 Get the project files onto your computer

If you already have the `crop prediction AI` folder (for example, you downloaded or copied it),
open a terminal and move into it:

```bash
cd "/Users/vap-purnvasu/Downloads/crop prediction AI"
```

(`cd` means "change directory" — move the terminal's focus into that folder. Every command
below assumes your terminal is sitting inside this folder.)

### 2.4 Create a Python "virtual environment"

Different Python projects on the same computer often need *different versions* of the same
library, which would conflict if everything were installed globally on your machine. A
**virtual environment** is an isolated, private copy of Python just for this one project, so its
dependencies never clash with any other project's. You create it once:

```bash
python3.11 -m venv .venv
```

This creates a new folder named `.venv` containing a private copy of Python. You then
**activate** it — telling your current terminal window "use the Python inside `.venv`, not the
one installed system-wide":

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

You'll know it worked because your terminal prompt now starts with `(.venv)`. You need to run
this `source` command again every time you open a *new* terminal window to work on this project
(it does not stay active permanently — it's specific to the terminal window it was run in).

### 2.5 Install the Python dependencies

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```

`pip` is Python's own package manager (the Python equivalent of `npm`). The file
[`requirements.txt`](requirements.txt) is a plain list of every Python library this project
needs, one per line, with an allowed version range. `pip` reads that list and downloads every
one of them (and *their* dependencies, recursively) into your `.venv` folder. Here is what each
line actually does for this project, and why it's needed:

| Package | What it is | Why this project needs it |
|---|---|---|
| `Flask` | A lightweight web server framework | Runs the backend, defines the API endpoints |
| `numpy` | Fast numerical array math | The mathematical foundation almost every other library here is built on |
| `pandas` | Spreadsheet-like data tables in Python | Loads and cleans the crop training CSV data |
| `scikit-learn` | Classical machine-learning algorithms | Provides Random Forest, Logistic Regression, Gradient Boosting, and the tools to train/evaluate them |
| `joblib` | Saves/loads Python objects to disk efficiently | Saves the trained crop model to a file so it doesn't need retraining every time the server starts |
| `shap` | Explainable AI toolkit | Computes *why* the model predicted what it predicted, feature by feature |
| `Pillow` | Image loading/processing | Needed by TensorFlow to read soil photos as pixel arrays |
| `tensorflow` | Deep learning framework | Trains and runs the soil-photo neural network (MobileNetV2) |
| `kagglehub` | Downloads datasets from Kaggle | One-line downloader for the public 2200-row crop dataset |
| `pytest` | Testing framework | Runs the automated correctness checks in `tests/` |

This step downloads roughly 250 MB (mostly TensorFlow) and can take a few minutes the first
time.

### 2.6 Install the frontend dependencies

Move into the `frontend` folder and let `npm` install everything listed in
[`frontend/package.json`](frontend/package.json):

```bash
cd frontend
npm install
```

What gets installed, and why:

| Package | What it is | Why this project needs it |
|---|---|---|
| `react` / `react-dom` | The UI library | Lets us describe the page as reusable "components" instead of raw HTML, and efficiently updates the screen when data changes |
| `vite` | A dev server + build tool | Instantly serves the frontend while you edit it (with live-reload), and bundles it into optimized static files for production |
| `@vitejs/plugin-react` | Glue between Vite and React | Teaches Vite how to translate `.jsx` files (JavaScript with embedded HTML-like tags) into plain JavaScript the browser understands |
| `lucide-react` | An icon library | Supplies the little SVG icons used throughout the interface (leaf, thermometer, upload cloud, etc.) |

`npm install` reads `package.json`, downloads every package into a new folder called
`node_modules` (this folder is intentionally excluded from version control — see
[`.gitignore`](.gitignore) — because it can be regenerated any time by re-running `npm install`),
and records the *exact* versions it picked in `package-lock.json`, so re-running install later
reproduces an identical set of files.

Move back to the project root when you're done:

```bash
cd ..
```

---

## Part 3 — Running the project

You need **two terminal windows open at the same time** — one for the backend, one for the
frontend. They keep running continuously while you use the app (closing the window, or pressing
`Ctrl+C` inside it, stops that server).

**Terminal 1 — backend:**

```bash
cd "/Users/vap-purnvasu/Downloads/crop prediction AI"
source .venv/bin/activate
python run.py
```

You should eventually see Flask print something like `Running on http://127.0.0.1:5001`. The
first run is slower because it has to load TensorFlow and the trained model files into memory.

**Terminal 2 — frontend:**

```bash
cd "/Users/vap-purnvasu/Downloads/crop prediction AI/frontend"
npm run dev
```

Vite will print a local address, normally `http://localhost:5173/`.

**Then:** open your web browser and visit `http://localhost:5173`. That's the entire app —
type in soil/weather numbers, click "Analyze conditions," and read the result.

> **A macOS-specific gotcha:** Apple's built-in AirPlay Receiver feature silently occupies port
> `5000` on every Mac by default. Flask's traditional default port is also `5000`, so on a Mac
> the backend would appear to "run" but never actually be reachable. This project's `run.py` and
> `frontend/vite.config.js` are already configured to use port **5001** instead, specifically to
> dodge that conflict — you don't need to do anything extra, this is just an explanation of why
> `5001` (an otherwise arbitrary number) was chosen instead of the more common `5000`. See
> [Part 19](#part-19--troubleshooting) if you ever see the backend fail to respond.

---

## Part 4 — Project map: every file, one sentence each

```
crop prediction AI/
├── run.py                        Tiny script that starts the Flask backend server.
├── requirements.txt              List of Python dependencies (see Part 2.5).
├── README.md                     This file.
├── .gitignore                    Tells Git which files/folders NOT to track (see below).
│
├── app/                          The web server layer — turns HTTP requests into answers.
│   ├── __init__.py               Empty file that marks `app` as a Python package (see Part 5.6).
│   ├── api.py                    Defines every URL ("route") the backend responds to.
│   ├── services.py               The "orchestrator": combines the ML model, the profile
│   │                             scoring, and the soil image model into one coherent answer.
│   └── crop_catalog.py           Loads the 182-crop reference list and tags which of those
│                                 182 crops the trained ML model actually knows about.
│
├── ml/                           The machine-learning / AI logic. Framework-agnostic —
│   │                             nothing in here knows about Flask or HTTP at all.
│   ├── __init__.py               Marks `ml` as a Python package.
│   ├── crop_model.py             Trains, saves, loads, and runs the 22-class crop classifier
│   │                             (Random Forest + Logistic Regression + Gradient Boosting).
│   ├── explainability.py         Uses SHAP to explain *why* the model predicted what it did.
│   ├── hybrid_recommender.py     Scores ALL 182 catalog crops — the 22 the ML model knows,
│   │                             plus 160 more via a transparent rule-based profile method.
│   ├── suitability.py            Compares your numbers against a crop's ideal ranges and
│   │                             generates human-readable remedies ("add more Nitrogen").
│   └── soil_model.py             Trains/loads/runs the MobileNetV2 image classifier that
│                                 guesses soil type from a photograph.
│
├── data/
│   └── reference/
│       ├── india_crop_catalog.json   The 182-crop reference database (name, category, common
│       │                             varieties, growing notes) used by the catalog UI.
│       └── india_crop_catalog.csv    Same data, spreadsheet format.
│
├── models/                       Saved, already-trained model files (binary/JSON) — the
│   │                             output of running the training scripts once.
│   ├── crop_voting.joblib        The trained scikit-learn ensemble model itself.
│   ├── crop_metadata.joblib      Per-crop quartile "profiles" used for gap analysis.
│   ├── crop_metrics.json         Accuracy/precision/recall numbers from training.
│   ├── soil_mobilenetv2.keras    The trained TensorFlow neural network for soil photos.
│   ├── soil_metadata.json        Class names + dataset counts for the soil model.
│   └── soil_metrics.json         Accuracy numbers for the soil model.
│
├── scripts/                      One-off command-line utilities you run manually, not part
│   │                             of the live web server.
│   ├── download_crop_dataset.py      Downloads the public 2200-row crop CSV from Kaggle.
│   ├── download_mendeley_agro_data.py Downloads a larger public agro-meteorological dataset.
│   ├── train_crop_model.py           Runs `ml/crop_model.py`'s `train()` on the CSV.
│   ├── train_soil_model.py           Runs `ml/soil_model.py`'s `train()` on soil photos.
│   └── validate_crop_catalog.py      Sanity-checks the 182-entry catalog JSON is intact.
│
├── tests/                        Automated correctness checks (see Part 18).
│   ├── test_api_contract.py      Confirms the API still answers correctly end-to-end.
│   └── test_suitability.py       Confirms the remedy-generation logic behaves as expected.
│
└── frontend/                     The React website.
    ├── index.html                 The one real HTML file — the empty shell React fills in.
    ├── vite.config.js             Configures the dev server, including the API proxy.
    ├── package.json               List of JavaScript dependencies (see Part 2.6).
    └── src/
        ├── main.jsx                Entry point: mounts the React app into index.html.
        ├── App.jsx                 The entire application UI and logic, as React components.
        └── styles.css              All visual styling (colors, spacing, layout).
```

**About `.gitignore` and `models/`:** [`.gitignore`](.gitignore) tells the version-control
system Git to *not* track certain files — `.venv/` (regenerable, huge), `__pycache__/`
(temporary Python bytecode caches), and, importantly, the trained model files under `models/`.
Wait — those model files are actually present in this copy of the project (they total roughly
50 MB and were committed anyway so the app works immediately without retraining); the
`.gitignore` rule simply means *future* retraining won't force you to re-commit them by accident.
You do **not** need to run any training script to use the app — the backend loads these
pre-trained files directly. Training scripts exist only if you want to retrain on new data
([Part 17](#part-17--training-your-own-models-from-scratch)).

---

## Part 5 — Python crash course (using real code from this project)

Every concept below is demonstrated with an actual line taken from this repository, so you can
see the theory and the real application side by side.

### 5.1 Variables

A variable is a named box that holds a value. Python has no separate "declare a box" step — you
create one just by assigning something to a name:

```python
IMG_SIZE = (224, 224)          # ml/soil_model.py, line 8
```
This creates a variable named `IMG_SIZE` holding a *tuple* (an ordered, fixed-size pair of
values) — the width and height, in pixels, that every soil photo gets resized to before the
neural network looks at it.

### 5.2 Data types used throughout this project

- **`int`** — a whole number, e.g. `42`.
- **`float`** — a number with a decimal point, e.g. `20.8` (used for temperature, pH, etc. —
  see `FEATURES` below).
- **`str`** (string) — text, e.g. `"rice"`. Always wrapped in quotes in code.
- **`bool`** — exactly `True` or `False`. Used for yes/no facts, e.g. `is_model_supported`.
- **`list`** — an ordered collection you can loop over, written with square brackets:
  ```python
  FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]   # ml/crop_model.py, line 17
  ```
  This is the master list of the seven numeric inputs the whole system is built around — every
  soil-nutrient value and every weather value the user types in.
- **`dict`** (dictionary) — a collection of `key: value` pairs, for looking things up by name
  instead of by position, written with curly braces:
  ```python
  DISPLAY = {
      "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
      "temperature": "Temperature", "humidity": "Humidity", "ph": "Soil pH", "rainfall": "Rainfall"
  }   # ml/suitability.py, lines 9-12
  ```
  Here, `DISPLAY["N"]` instantly gives you back `"Nitrogen (N)"` — used to turn the short
  internal code `"N"` into a friendly label for display.
- **`tuple`** — like a list, but its size and contents are meant to never change after creation,
  e.g. `(224, 224)` above, or a `(low, high)` numeric range.

### 5.3 `for` loops — doing something to every item in a collection

A `for` loop repeats a block of instructions once for every item in a list (or any other
"iterable" collection), without you having to write the same code out by hand N times.

```python
for f in FEATURES:                       # ml/crop_model.py, inside crop_profiles()
    q1, med, q3 = g[f].quantile([0.25, 0.5, 0.75])
    item[f] = {"q1": float(q1), "median": float(med), "q3": float(q3)}
```
Read this as: "for each feature name `f` in the list `FEATURES` (so first `f = "N"`, then
`f = "P"`, and so on through all seven) — run the two indented lines below using that `f`."
This is how the project computes the ideal Nitrogen range, then the ideal Phosphorus range, then
Potassium, etc., without writing seven nearly-identical blocks of code by hand. The indentation
(consistent whitespace at the start of a line) is not just a style choice in Python — it is how
Python knows which lines belong *inside* the loop versus which lines come after it. This is
different from many other languages, which use curly braces `{ }` for that instead.

A nested version — a loop inside a loop — appears in
[`ml/hybrid_recommender.py`](ml/hybrid_recommender.py):
```python
for row in catalog:                      # outer loop: for every one of the 182 crops...
    ...
    for feature in FEATURES:             # inner loop: ...check all 7 of its features
        low, high = _range_from_profile(entry["profile"], feature)
        ...
```
For each of the 182 crops, the inner loop runs all the way through the 7 features before the
outer loop moves on to the next crop — so this block runs `182 × 7 = 1,274` times per request.

### 5.4 `if` / `elif` / `else` — making decisions

A conditional lets the program take a different path depending on whether something is `True`
or `False`.

```python
if v < p["q1"]:                          # ml/suitability.py, inside feature_gap()
    direction = "low"
    severity = (p["q1"] - v) / max(abs(p["median"] - p["q1"]), 1e-9)
elif v > p["q3"]:
    direction = "high"
    severity = (v - p["q3"]) / max(abs(p["q3"] - p["median"]), 1e-9)
else:
    direction = "within"
    severity = 0.0
```
In plain English: "if the user's value `v` is below the crop's typical lower quartile `q1`, mark
it `low` and compute how far below; otherwise, if it's above the upper quartile `q3`, mark it
`high`; otherwise, it must be inside the normal range, so mark it `within` with zero severity."
Only **one** of the three branches ever runs, in order — Python checks `if` first, and only
checks `elif` (short for "else if") if the first was `False`, and only runs `else` if *neither*
above was `True`.

A one-line condition-check used constantly in this project is the **chained comparison**:
```python
if q1 <= v <= q3:                        # ml/suitability.py, inside agricultural_suitability()
```
This is Python shorthand for "is `v` between `q1` and `q3`, inclusive" — equivalent to writing
`if q1 <= v and v <= q3:` but more readable.

### 5.5 Functions — reusable, named blocks of instructions

A function is a named, reusable recipe: you define it once, then "call" it (run it) as many
times as you like, optionally feeding it different input values (**arguments**/**parameters**)
each time, and optionally getting a value back out (**return value**).

```python
def _score_dimension(value: float, low: float, high: float) -> float:   # ml/hybrid_recommender.py
    if low <= value <= high:
        mid = (low + high) / 2.0
        half = max((high - low) / 2.0, 1e-9)
        return 1.0 - 0.25 * min(abs(value - mid) / half, 1.0)
    distance = (low - value) if value < low else (value - high)
    span = max(high - low, 1e-9)
    return max(0.0, 1.0 - 0.75 * (distance / span))
```
- `def` starts a function definition; `_score_dimension` is its name (leading underscore is just
  a Python convention meaning "internal helper, not meant to be used from outside this file").
- `(value: float, low: float, high: float)` lists its three inputs. The `: float` parts are
  **type hints** — notes for humans (and some tools) about what kind of value is expected; Python
  does not strictly enforce them at runtime, they are documentation, not a hard rule.
- `-> float` says the function promises to hand back a `float` when it's done.
- `return` is the keyword that sends a value back to whoever called the function and immediately
  stops running the rest of the function.
- Calling it looks like `_score_dimension(65.0, 40.0, 140.0)` — plug in real numbers for
  `value`, `low`, `high` and you get one number back describing "how well does 65 fit inside the
  ideal range 40–140". The exact formula and its meaning is explained fully in
  [Part 11.2](#112-the-suitability-score-formula-explained).

### 5.6 Modules, packages, and `import`

Every `.py` file is automatically a reusable **module** — a bundle of functions/variables other
files can borrow. A folder becomes a **package** (an importable group of modules) the moment it
contains a (often empty) file literally named `__init__.py` — that's the entire purpose of the
empty [`app/__init__.py`](app/__init__.py) and [`ml/__init__.py`](ml/__init__.py) files you saw
in the project map: they exist purely to tell Python "treat this folder as an importable unit,"
even though they contain zero code themselves.

```python
from ml.crop_model import FEATURES, CropBundle, load_bundle   # app/services.py, line 10
```
This line reaches into the `ml` package, opens the `crop_model.py` module inside it, and pulls
three specific names (`FEATURES`, the `CropBundle` class, and the `load_bundle` function) into
the current file so they can be used directly by name. This is how the project stays organized
into small, focused files instead of one giant file — `app/services.py` doesn't need to know
*how* `load_bundle` works internally, only that importing it gives it a working function to
call.

### 5.7 Classes and objects

A **class** is a blueprint for creating structured bundles of related data and behavior. An
**object** (or **instance**) is one concrete thing built from that blueprint.

```python
@dataclass
class CropBundle:                 # ml/crop_model.py, lines 20-23
    model: VotingClassifier
    metadata: dict[str, Any]
```
`@dataclass` is a **decorator** (a function that wraps another piece of code to add behavior to
it, marked with `@` directly above what it modifies) that auto-generates the boring boilerplate
code for a simple data container — here, `CropBundle` is a labeled pair: one trained
`VotingClassifier` model object, plus one `dict` of metadata (crop names, quartile profiles).
Elsewhere in the code you'll see `bundle.model` and `bundle.metadata` — the dot (`.`) accesses a
named field that lives *inside* that particular `CropBundle` object, exactly like `DISPLAY["N"]`
looked something up inside a dictionary, but here the fields are fixed by the class's blueprint
rather than being arbitrary lookup keys.

A second, larger class in this project is a **service class**:
```python
class AgricultureService:                       # app/services.py, line 17
    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.crop: CropBundle = load_bundle(self.models_dir)
        ...
```
`__init__` (short for "initialize") is a special function that runs automatically the moment an
object of this class is created — it's the "constructor." `self` is how a method (a function
defined inside a class) refers back to *this particular object's own* data — `self.crop`
becomes a piece of data permanently attached to that one `AgricultureService` object, loaded
once, and reused for every request afterward rather than being re-loaded from disk on every
single question the server answers. This matters a lot for a web server: model files can be
tens of megabytes, and loading them from disk is slow — you want that to happen exactly once, at
startup, not on every click.

### 5.8 Error handling: `try` / `except` and `raise`

```python
try:
    value = float(values[feature])
except (TypeError, ValueError) as exc:
    raise ValueError(f"Invalid numeric value for {feature}") from exc
```
(from `AgricultureService.validate_inputs` in `app/services.py`)

`try:` marks a block of code that *might* fail. If anything inside it throws an **exception**
(Python's word for "something went wrong, stop normal execution immediately"), control jumps
straight to the matching `except` block instead of crashing the whole program. Here: if the text
the user typed can't actually be converted to a number (e.g. they typed letters into a numeric
field), Python's built-in `float()` conversion raises a `TypeError` or `ValueError`; this code
catches that and re-raises a clearer, custom `ValueError` with a helpful message instead, which
[`app/api.py`](app/api.py) later catches again and turns into a proper HTTP error response for
the user (see [Part 7](#part-7--the-backend-file-by-file)). `raise` is the keyword used to
deliberately trigger an exception yourself.

### 5.9 `print()` and why this project barely uses it

`print(...)` writes text to the terminal — the most basic way to see what a running program is
doing, invaluable for learning and debugging. You'll find it used sparingly in the one-off
[`scripts/`](scripts/) (e.g. `print(f"Copied {src} -> {out}")` in
`download_crop_dataset.py`, or the accuracy summary at the end of `train_crop_model.py`) because
those scripts are run manually by a human who wants to watch progress. The web server code
(`app/`, `ml/`) intentionally avoids `print()` — a running server has no one sitting there
reading terminal output in real time; instead it uses Flask's own `app.logger.exception(...)`
(see `app/api.py`) which writes to a proper log with a timestamp and full error detail, and
returns a clean JSON error to whoever asked the question. `f"..."` strings, seen in several
examples above, are **f-strings** — text with `{variable}` placeholders inside them that Python
automatically fills in with that variable's actual value when the string is built.

### 5.10 List/dict comprehensions

A compact way to build a new list or dict directly from a loop, in one line:
```python
probabilities = {
    crop: float(probability)
    for crop, probability in zip(classes, probs)
}   # ml/crop_model.py, inside predict()
```
This means exactly the same thing as:
```python
probabilities = {}
for crop, probability in zip(classes, probs):
    probabilities[crop] = float(probability)
```
`zip(classes, probs)` pairs up the two lists element-by-element (first class with first
probability, second with second, ...), and the comprehension builds a dictionary mapping each
crop name straight to its predicted probability, in one readable line instead of four.

---

## Part 6 — JavaScript, React and HTML crash course

### 6.1 What HTML tags actually are

A web page is a text file made of **tags** — words wrapped in angle brackets, like `<div>`,
that tell the browser "this content is a block of this specific kind," so the browser knows how
to lay it out and (later) how CSS/JavaScript can target it. Most tags come in an opening/closing
pair, `<tag>...content...</tag>`; the browser draws whatever is between them. Every tag actually
used in this project's UI:

| Tag | Real-world meaning | Where you'll see it |
|---|---|---|
| `<div>` | A generic, meaning-free rectangular box — used purely for grouping/layout | Almost everywhere; e.g. `app-shell`, `hero-visual` |
| `<section>` | A generic block, but semantically "a distinct section of the page" (helps accessibility tools/screen readers understand page structure) | `<section className="hero section-pad">` |
| `<header>` (`<header>`… ) / `<footer>` | The top banner / bottom banner of the page | The top nav bar and the closing credit line |
| `<h1>`, `<h2>`, `<h3>` | Headings, in decreasing importance/size — `<h1>` is the single most important headline on the page | The hero title, panel titles, card titles |
| `<p>` | A paragraph of text | Descriptive copy under headings |
| `<span>` | Like `<div>` but for small *inline* pieces of text/content inside a line, not a whole block | Small labels, badges, numbers |
| `<button>` | A clickable button | "Demo values," "Analyze conditions," "Choose image" |
| `<form>` | A group of input fields meant to be submitted together | Wraps the 7 numeric fields + submit button |
| `<label>` | Text tied to a specific input, both for visual clarity and for accessibility (clicking the label focuses its input) | Every numeric field's name |
| `<input>` | A single editable field (`type="number"` for numbers, `type="file"` for file uploads) | The N/P/K/temperature/etc. boxes; the hidden file picker for soil photos |
| `<select>` / `<option>` | A dropdown menu / one choice inside it | "Specific crop to evaluate" and the catalog category filter |
| `<article>` | A self-contained, independently-meaningful chunk of content | Each individual crop card in the catalog grid |
| `<strong>` | Bold, semantically "important" text | Emphasized values throughout the result cards |
| `<small>` | De-emphasized, smaller-print text | Units (e.g. "mg/kg"), captions |

The general rule: pick the tag that best describes *what the content is*, and let CSS handle
*how it looks*. `<div>`/`<span>` are the deliberately meaning-free fallback when no more specific
tag fits.

### 6.2 JSX — HTML-like tags living inside JavaScript

[`frontend/src/App.jsx`](frontend/src/App.jsx) is not plain HTML and not plain JavaScript — it's
**JSX**, a syntax extension that lets you write tag-like markup directly inside JavaScript code.
A build tool (here, the `@vitejs/plugin-react` dependency you installed in Part 2.6) translates
JSX into plain `React.createElement(...)` function calls before the browser ever sees it — so
`<div className="app-shell">` is really shorthand for "call a function that creates a `div` DOM
node with that CSS class." Two JSX-specific quirks to know:

- You write `className` instead of HTML's `class` (because `class` is already a reserved word in
  JavaScript, used for defining classes as in Part 5.7's concept, just in JS instead of Python).
- Anything inside curly braces `{ }` inside JSX drops back into *live JavaScript* — so
  `{titleCase(result.prediction.crop)}` isn't literal text, it's a function call whose *result*
  gets displayed, recomputed automatically whenever the underlying data changes.

### 6.3 React components

A **component** is just a JavaScript function that returns JSX — a description of what should
be on screen for the data it's currently given. `App.jsx` defines four:

- `App()` — the top-level component: the whole page.
- `CropCatalog(...)` — the searchable grid of all 182 reference crops.
- `AnalysisResult(...)` — the block of cards shown after you click "Analyze."
- `SoilUpload(...)` / `ModelSummary(...)` — the soil-photo uploader and the "model health"
  stats card.

React re-runs a component's function and redraws only the parts of the page that actually
changed, automatically, any time the data it depends on changes — you never manually write code
to "find this box and change its text," you just describe what the box *should* show given the
current data, and React keeps the real page in sync with that description for you.

### 6.4 State — `useState`

A component needs to "remember" things between renders (e.g. what you've typed so far). This
project uses React's `useState`:
```jsx
const [values, setValues] = useState(DEFAULTS)   // App.jsx, line 65
```
This declares one piece of remembered data, `values` (initially set to the `DEFAULTS` object —
sample soil numbers), plus a function `setValues` that is the *only* correct way to change it.
Calling `setValues(...)` does two things: updates the stored value, and tells React "please
re-run this component's function again so the screen reflects the new value." `const [a, b] = x`
here is **array destructuring** — a shorthand for pulling the first and second items out of the
two-item array `useState` returns, and naming them `values` and `setValues` in one line.

Every editable piece of data on the page is its own `useState`: the seven number fields
(`values`), the chosen target crop (`targetCrop`), whether a request is currently in flight
(`loading`), any error message (`error`), the AI's last answer (`result`), and so on.

### 6.5 Side effects — `useEffect`

Some things need to happen "when the component first appears," outside of normal rendering —
most commonly, fetching data from the server. `useEffect` handles that:
```jsx
useEffect(() => {
  Promise.all([
    fetch('/api/crops').then((r) => r.json()),
    fetch('/api/metrics').then((r) => r.json()),
    fetch('/api/crop-catalog').then((r) => r.json()),
  ])
    .then(([cropData, metricData, catalogData]) => {
      setCrops(cropData.crops || [])
      setMetrics(metricData)
      setCropCatalog(catalogData.crops || [])
    })
    .catch(() => { /* the form remains usable even if this fails */ })
}, [])
```
`() => { ... }` is an **arrow function** — a compact way to write an anonymous (unnamed)
function in JavaScript, functionally similar to Python's `def`, just with different punctuation.
`fetch(url)` sends an HTTP request to that address and returns a **Promise** — a placeholder
standing in for "a value that will arrive later, asynchronously" (because a network request can
take anywhere from a few milliseconds to several seconds, and JavaScript refuses to freeze the
whole page waiting for it). `.then(...)` says "once that Promise finally resolves with real
data, run this next function using it." `Promise.all([...])` runs all three requests
*simultaneously* rather than one after another, then waits for all three to finish before
continuing — faster than doing them sequentially. The second argument to `useEffect`, the empty
list `[]`, tells React "only run this effect once, right when the component first appears," not
on every re-render — this is how the app fetches the crop list and metrics exactly once at
startup instead of re-fetching them constantly.

### 6.6 Rendering lists — `.map()`

To turn a JavaScript array into a row of JSX elements, React code universally uses the array's
built-in `.map()` method (conceptually identical to Python's list comprehension from 5.10 — "for
every item, produce one transformed output"):
```jsx
{FIELD_CONFIG.map(({ key, label, unit, icon: Icon }) => (
  <label className="field" key={key}> ... </label>
))}
```
For every object in the `FIELD_CONFIG` array (one entry per soil/weather field — Nitrogen,
Phosphorus, ...), this produces one `<label>` box, and React assembles all seven into the input
grid. The `key={key}` prop is not visual — it's a hint React uses internally to efficiently track
which item is which across re-renders, and React requires it (and will print a warning without
it) whenever you render a list this way.

### 6.7 Conditional rendering

JSX has no `if` tag; instead, JavaScript's own logical operators are reused directly inside the
curly-brace escape hatch:
```jsx
{loading ? <><Loader2 .../> Analyzing...</> : <>Analyze conditions <ArrowRight /></>}
```
This is a **ternary expression** — `condition ? valueIfTrue : valueIfFalse` — read as "if
`loading` is true, show the spinner and 'Analyzing...', otherwise show the normal button label."
Elsewhere you'll see the short-circuit form:
```jsx
{error && (<div className="error-banner">...</div>)}
```
In JavaScript, `&&` evaluates its left side first; if that's `false`/empty/`null` it stops right
there and the whole expression becomes that falsy value (so React renders nothing); only if the
left side is truthy does it go on to evaluate and render the right side. So this line means
"only show the error banner at all if `error` currently holds a non-empty message."

### 6.8 Sending data to the backend

```jsx
const response = await fetch('/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...values, target_crop: targetCrop || null }),
})
const data = await response.json()
```
(from the `submit` function in `App.jsx`)

`method: 'POST'` — HTTP requests come in different "verbs"; `GET` means "just give me data" (used
for `/api/crops`, `/api/metrics`, etc.), while `POST` means "here is data, please process it and
give me a result" (used for `/api/analyze`, which needs to send the 7 numbers). `{ ...values,
target_crop: ... }` uses the **spread operator** (`...`) to copy every field out of `values`
into a brand-new object, then adds one more field alongside them — a concise way to build "all of
this, plus one more thing" without mutating the original. `JSON.stringify(...)` converts that
JavaScript object into the JSON text format described in Part 1, ready to travel over the
network. `await` pauses this specific function (without freezing the rest of the page) until the
Promise it's attached to actually resolves — it's a cleaner, more readable alternative to
chaining `.then()` calls, doing the exact same underlying asynchronous work.

---

## Part 7 — The backend, file by file

### 7.1 `app/api.py` — defining the "doors" the outside world can knock on

Flask lets you attach a Python function to a specific web address using a **decorator** (recall
Part 5.7 — `@` immediately above a function, modifying it):

```python
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "crop_model": True,
        "soil_model": service.soil_available,
    })
```
`@app.get("/api/health")` means: "whenever an HTTP `GET` request arrives at exactly the address
`/api/health`, run the function immediately below this line, and send whatever it returns back
as the response." `jsonify(...)` takes a Python `dict` and converts it into a proper JSON HTTP
response (setting the correct `Content-Type` header along the way, so the browser knows to
interpret it as JSON). This particular endpoint is a simple health check — the frontend doesn't
actually call it, but it's a standard practice for any web service, letting you (or automated
monitoring) quickly confirm "is the server up, and are the models loaded?" by visiting it
directly.

Every endpoint defined in this file:

| Method + path | Purpose |
|---|---|
| `GET /api/health` | Is the server alive? Are the crop/soil models loaded? |
| `GET /api/crops` | List every crop name enabled in the UI, plus which subset the trained ML model itself recognizes |
| `GET /api/crop-catalog` | The full 182-entry reference catalog, tagged with `model_supported` per crop |
| `GET /api/metrics` | The crop model's saved accuracy/precision/recall numbers, straight from `models/crop_metrics.json` |
| `POST /api/analyze` | The main endpoint: send 7 numbers (+ optional target crop), get back a full recommendation |
| `POST /api/analyze-with-image` | Same as above, plus a soil photo, sent as a file upload (`multipart/form-data`) instead of plain JSON |

The `POST /api/analyze` handler shows the standard three-part shape used by both analysis
endpoints:
```python
@app.post("/api/analyze")
def analyze():
    try:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400
        values = {key: body.get(key) for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]}
        target = body.get("target_crop") or None
        return jsonify(service.analyze(values, target_crop=target))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "Analysis failed. Check the server logs for details."}), 500
```
1. **Parse and validate the incoming request.** `request.get_json(silent=True)` reads the JSON
   body the frontend sent; `silent=True` means "return `None` instead of crashing if it's not
   valid JSON," which the very next line checks for explicitly.
2. **Do the real work** by delegating to `service.analyze(...)` — the API layer itself contains
   *no* AI or business logic; its only job is translating an HTTP request into a plain Python
   function call, and translating the plain Python result back into an HTTP response. This
   separation means the actual logic in `app/services.py` can be tested, reused, or called from
   anywhere (a command-line script, a test file) without needing a running web server at all.
3. **Turn errors into proper HTTP status codes.** A bad/missing input raises `ValueError`
   (Part 5.8) inside `service.analyze`, caught here and turned into HTTP status `400`
   ("Bad Request" — the client's fault). Any *other*, unexpected kind of crash is caught by the
   broad `except Exception`, logged in full server-side for a developer to diagnose later, but
   only a generic message is sent back to the user as status `500` ("Internal Server Error" —
   the server's fault) — deliberately not leaking internal error details/stack traces to
   whoever's using the app.

Two more details worth knowing:
```python
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024   # MAX_UPLOAD_MB = 8
```
Rejects any uploaded file bigger than 8 MB (megabytes) before it's even fully received, to stop
one huge upload from tying up server memory. `1024 * 1024` converts megabytes to bytes (the
actual unit computers measure size in) — 1 KB = 1024 bytes, 1 MB = 1024 KB.

```python
@app.after_request
def add_cors_headers(response):
    response.headers.setdefault("Access-Control-Allow-Origin", "http://localhost:5173")
```
Browsers enforce a security rule called **CORS** (Cross-Origin Resource Sharing): by default, a
page loaded from one address (`localhost:5173`) is *not* allowed to receive responses from a
different address (`127.0.0.1:5001`) unless that second server explicitly says "I permit
requests from that specific address." This `@app.after_request` hook runs on the way out of
*every* response and adds that permission header. (In this project's actual dev setup this
header ends up not mattering day-to-day, because Vite's proxy — Part 15.4 — makes the browser
think it's talking to only one address in the first place; the header is a defensive fallback
for the case of calling the API directly.)

### 7.2 `app/crop_catalog.py` — loading the reference data

```python
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "india_crop_catalog.json"
```
`__file__` is a special Python variable automatically holding "the path to the file currently
running." `.resolve()` turns it into a full, unambiguous absolute path. `.parent.parent` walks
up two folder levels (out of `app/`, to the project root), and `/ "data" / "reference" / ...`
(the `/` here is overloaded by the `Path` type to mean "join this path segment on," not
division) builds the final path to the JSON file — this way the code works correctly no matter
what folder you happen to be standing in when you launch it, since it's always computed relative
to this specific file's own location rather than assuming a particular starting directory.

`catalog_metadata(model_crops)` loops over all 182 entries and, for each one, figures out
whether it maps onto one of the 22 crops the trained ML model actually knows via the `ALIASES`
dictionary (e.g. the catalog lists `"Maize / Corn"` but the trained model's class is just
`"maize"`) and tags it `model_supported: true/false` accordingly — this single flag is what
drives the "AI supported" badge you see on some catalog cards in the UI.

### 7.3 `app/services.py` — the orchestrator

This is the file that ties every other piece together. `AgricultureService.__init__` (the
constructor, Part 5.7) runs once, when the server starts:
```python
self.crop: CropBundle = load_bundle(self.models_dir)   # load the trained crop model from disk
self.catalog = load_catalog()                            # load the 182-crop reference list
...
try:
    from ml.soil_model import load
    soil_path = self.models_dir / "soil_mobilenetv2.keras"
    if soil_path.exists():
        self.soil = load(self.models_dir)
        self.soil_available = True
except Exception:
    self.soil = None
    self.soil_available = False
```
The soil model load is deliberately wrapped in `try/except` — TensorFlow is a large, sometimes
finicky dependency, and this project is explicitly designed to keep working (crop
recommendations still function normally) even if the soil-photo feature can't load for some
reason on a particular machine; `self.soil_available` becomes a flag the rest of the app checks
before trying to use it.

`analyze(...)` is the heart of the whole backend. Walking through it in order:

1. **Validate** the seven raw inputs (`validate_inputs`, shown fully in Part 5.8) — every value
   must be present, convertible to a number, and finite (not `NaN`/infinity, which can arise
   from bad math and would silently corrupt every downstream calculation).
2. **Rank all 182 catalog crops** by calling `recommend(...)` from `hybrid_recommender.py`
   (Part 11) — this single call is what makes crops outside the original 22-class dataset usable
   at all.
3. **Also run the raw supervised model** on its own (`ml_predict`) to get a second, ML-only
   ranking — used specifically for the broader "is this land suitable for agriculture at all"
   check (Part 12), which is deliberately based on the trained model's own learned feature
   ranges rather than the wider heuristic catalog.
4. **Resolve which crop is actually being evaluated** — either the one the hybrid recommender
   ranked first, or, if the user picked a specific crop from the dropdown, that one instead
   (`target_crop`).
5. **Branch on whether that crop is one of the 22 ML-trained classes** (`is_model_supported`):
   - If yes: use the *learned, empirical* per-crop quartile profile (built from real training
     data — see `crop_profiles()` in Part 9) via `feature_gap` + `target_suitability`.
   - If no: fall back to the transparent, rule-based category-profile score from the hybrid
     recommender instead, and builds an equivalent-shaped result by hand so the frontend can
     treat both cases identically without needing to know which path was taken.
   This dual path is the project's central design decision, and it's stated explicitly in the
   code comments and the original [README](#) copy this file replaces: the project is honest
   about the difference between "a crop the model was actually *trained* on real labeled data
   for" versus "a crop we only have general agronomic-category ranges for" — it never presents
   the second kind as if it had the same statistical backing as the first.
6. **Generate `reasons`** — every feature that's outside the ideal range, sorted worst-first by
   `severity`, each with a plain-English sentence.
7. **Generate `remedies`** — calls into `ml/suitability.py`'s `remedies()` (Part 12.4) to turn
   each `reasons` entry into an actionable instruction ("increase Nitrogen using...").
8. **Optionally run the soil-photo model**, if an image was uploaded, saving it to a temporary
   file first (neural network image-loading libraries expect a real file path, not raw upload
   bytes in memory) and always deleting that temporary file afterward in a `finally` block (code
   that runs whether or not an error happened — guaranteeing cleanup either way).
9. **Assemble and return one single JSON-shaped dictionary** containing everything the frontend
   needs to render the entire results screen in one round trip.

---

## Part 8 — Machine learning theory, from zero

### 8.1 What "machine learning" means here

Ordinarily, a program's logic is *hand-written* by a human: "if temperature is above 30, do X."
**Machine learning** flips that around: instead of a human writing the rules, you show the
computer thousands of *already-known, correct examples* (this soil/weather combination → this
crop grew well there), and an algorithm automatically works out a mathematical rule that fits
those examples — one that, ideally, *also* correctly predicts crops it has never seen the exact
numbers for before. The output of that process — the learned rule itself, plus all the numbers
it settled on — is called a **trained model**.

This project's crop model is a case of **supervised classification**:
- **Supervised** — the training data comes with the correct answer already attached (each row of
  the training spreadsheet already says which crop that soil/weather combination was recorded
  for), as opposed to *unsupervised* learning, which looks for patterns with no pre-given
  answers at all.
- **Classification** — the thing being predicted is one label out of a fixed, finite set of
  categories (one of 22 specific crop names), as opposed to *regression*, which predicts a
  continuous number (e.g. "predict tomorrow's exact temperature").

### 8.2 The training data

The crop model is trained on a public 2,200-row spreadsheet (`Crop_recommendation.csv`,
downloaded via `scripts/download_crop_dataset.py`, from the Kaggle dataset
`atharvaingle/crop-recommendation-dataset`). Every row is one example: 7 numbers (the
**features** — the model's inputs) and 1 crop name (the **label** — the correct answer for that
row). There are exactly 100 rows for each of the 22 crops, so the dataset is perfectly
**balanced** (no crop is over- or under-represented, which would otherwise bias the model toward
guessing whichever crop appears most often).

The 7 features (`FEATURES` in `ml/crop_model.py`), and why each one matters agronomically:

| Feature | Meaning | Typical unit |
|---|---|---|
| `N` | Nitrogen content in soil | mg/kg (ratio of nitrogen mass to soil mass) — drives leaf/stem growth |
| `P` | Phosphorus content in soil | mg/kg — drives root development and flowering |
| `K` | Potassium content in soil | mg/kg — drives overall plant vigor and disease resistance |
| `temperature` | Average ambient temperature | °C |
| `humidity` | Relative air humidity | % |
| `ph` | Soil acidity/alkalinity | 0 (very acidic) to 14 (very alkaline); 7 is neutral |
| `rainfall` | Total rainfall | mm |

### 8.3 Splitting the data: train / test / cross-validation

If you trained a model and then evaluated its accuracy on the *exact same rows* it just
memorized, you'd get a misleadingly perfect score that says nothing about how it handles new,
unseen situations — this failure mode is called **overfitting**. To catch it, this project's
training procedure (`train()` in `ml/crop_model.py`) uses three separate safeguards:

1. **Train/test split** — `train_test_split(X, y, test_size=0.20, random_state=42,
   stratify=y)`. 80% of the rows are set aside purely for training; the other 20% are held back,
   completely unseen by the model until the very end, purely for scoring it afterward.
   `stratify=y` guarantees the 80/20 split preserves the same proportion of each crop in both
   halves (so the test set isn't accidentally missing an entire crop). `random_state=42` fixes
   the randomness to a specific, repeatable sequence — without it, re-running training would
   shuffle the data differently every time and produce slightly different results each run,
   making results hard to compare or reproduce; `42` itself has no special meaning, it's simply
   a conventional arbitrary choice among programmers (a nod to *The Hitchhiker's Guide to the
   Galaxy*).
2. **Held-out test accuracy** — after training only on the 80% training rows, the model predicts
   on the 20% test rows it has genuinely never seen, and that accuracy is what actually gets
   reported and trusted.
3. **5-fold cross-validation** — `cross_val_score(build_model(), X, y, cv=StratifiedKFold(
   n_splits=5, shuffle=True, random_state=42), scoring="accuracy")`. This repeats the
   train/test idea five separate times: split the *entire* dataset into 5 equal chunks
   ("folds"), then five times in a row train on 4 of the folds and test on the 1 remaining fold
   (a different fold held out each time), and average the five resulting accuracy scores. This
   is a stronger check than a single 80/20 split because *every* row gets used as test data
   exactly once across the five runs, so the final average accuracy is far less sensitive to
   which particular rows happened to land in the test set.

### 8.4 The three underlying algorithms

**Logistic Regression.** The simplest of the three: for each possible crop, it learns one
numeric "weight" per feature (7 weights, one per feature, per crop), multiplies each input
feature by its learned weight, adds them all up, and squashes that sum through a **sigmoid**
function (an S-shaped curve that maps any number into a value between 0 and 1) so the output can
be read as a probability. During multi-class training this is generalized to a **softmax**
across all 22 crops, guaranteeing all 22 predicted probabilities add up to exactly 1. The weights
themselves are found by an optimization process (**gradient descent**): start with random
weights, measure how wrong the current predictions are using a **loss function**
(cross-entropy — a formula that heavily penalizes being confidently wrong), then nudge each
weight slightly in whichever direction reduces that loss, repeating thousands of times until the
loss stops improving. Because this algorithm is fundamentally a weighted *sum*, it is sensitive
to the raw scale of each feature — Nitrogen values range roughly 0–140 while pH ranges roughly
3.5–9.5, and without correction the model would wrongly treat Nitrogen as "more important" simply
because its numbers are bigger. That's exactly why Logistic Regression alone (not the other two
algorithms) is wrapped in a **`StandardScaler`**:
```python
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, random_state=42)),
])
```
`StandardScaler` rescales every feature to have mean 0 and standard deviation 1 (subtract each
column's average, then divide by its spread) before Logistic Regression ever sees it, putting
every feature on a fair, comparable footing. A `Pipeline` bundles the scaler and the classifier
into one object that always applies both steps together, in the same order, both during training
and later during prediction — preventing the easy mistake of forgetting to scale new data the
exact same way the training data was scaled.

**Random Forest.** Picture one single **decision tree**: a flowchart of yes/no questions about
the features ("Is rainfall > 200mm? → if yes, is temperature > 25°C? → ...") that a training
algorithm builds automatically by repeatedly picking whichever question best separates the
training examples into purer, more single-crop groups (measured by a formula like **Gini
impurity**, which is lowest when a group is all one class and highest when it's evenly mixed).
A single decision tree easily overfits — it can carve out an overly specific rule that fits the
training data perfectly but generalizes poorly. A **Random Forest** fixes that by training many
trees (`n_estimators=500` — 500 separate trees here) and averaging their opinions, where each
individual tree is deliberately trained on a random, slightly different subset of the training
rows and a random subset of features (a technique called **bagging** — "bootstrap aggregating"),
so the trees end up genuinely different from each other rather than all making the exact same
mistakes; averaging many diverse, individually-decent guessers reliably beats trusting a single
one. `class_weight="balanced"` further adjusts how much each class "counts" during training in
inverse proportion to how common it is, protecting against any residual imbalance.

**Gradient Boosting.** Also an ensemble of many small decision trees, but built completely
differently from Random Forest: instead of training all trees independently and averaging them,
it builds trees **one at a time, in sequence**, where each new tree is deliberately trained to
correct the specific mistakes ("residual errors") the trees built so far are still making — each
new tree nudges the combined prediction a little closer to correct, guided by the same
gradient-descent idea described under Logistic Regression, but applied to improve an ensemble of
trees rather than a single set of weights. This tends to produce very accurate models but is
more prone to overfitting than Random Forest if left unchecked (which is one reason it's used
here only as one voice among three, not alone).

**Why three different algorithms, combined?** Each algorithm makes different *kinds* of
mistakes — Logistic Regression is limited to relatively simple, roughly-linear boundaries between
crops; the two tree-based methods can capture more complex, jagged boundaries but can behave
erratically in slightly different situations from each other. Combining models with different
failure patterns tends to cancel out each other's individual weaknesses — this idea is called
**ensembling**, and it's why the code wraps all three inside one final:
```python
VotingClassifier(
    estimators=[("rf", rf), ("lr", lr), ("gb", gb)],
    voting="soft",
    weights=[2, 1, 2],
    ...
)
```
`voting="soft"` means the final answer isn't decided by each model casting one vote for its
single top guess (that would be "hard" voting); instead, each of the three models produces a
full probability for *every* crop, and the ensemble takes a weighted average of those
probabilities across all three models, crop by crop, before picking the highest combined
number. `weights=[2, 1, 2]` (matching the `rf, lr, gb` order listed just above it) means the
Random Forest's and Gradient Boosting's opinions each count twice as heavily as the Logistic
Regression's in that average — reflecting that the two tree-based models are generally
better-suited to this kind of data (their ability to model non-linear relationships between
features tends to fit real agronomic data, which rarely relates to yield in a straight line,
better than a purely linear model does).

### 8.5 Accuracy, precision, recall, F1 — the formulas, explained

After training, `train()` computes several standard scoring formulas via `scikit-learn`. For a
single crop class, imagine sorting every test-set prediction into four buckets:

- **True Positive (TP)** — model said *this crop*, and it really was this crop.
- **False Positive (FP)** — model said *this crop*, but it was actually a different one.
- **False Negative (FN)** — model said a *different* crop, but it was actually this one.
- **True Negative (TN)** — model correctly said "not this crop" for a row of a different crop.

```
Accuracy  = (correct predictions) / (total predictions)
Precision = TP / (TP + FP)     "Of everything I labeled X, how much really was X?"
Recall    = TP / (TP + FN)     "Of everything that really was X, how much did I catch?"
F1-score  = 2 × Precision × Recall / (Precision + Recall)      (their harmonic mean — a
                                                                  single balanced summary that
                                                                  only scores high if BOTH
                                                                  precision and recall are high)
```
`classification_report(...)` in `ml/crop_model.py` computes precision/recall/F1 *per crop*
(useful for spotting "the model is great overall but consistently confuses lentil with
mothbeans," for instance), and `confusion_matrix(...)` builds a 22×22 grid where cell
`[actual crop][predicted crop]` counts how many test rows of that actual crop got predicted as
that predicted crop — the diagonal (actual = predicted) should hold almost all the counts for a
good model; anything off the diagonal is a specific, visible mistake pattern.

**This project's actual, real numbers** (from `models/crop_metrics.json`, produced the last time
`train()` was run on the 2,200-row dataset):

| Metric | Value | What it means |
|---|---|---|
| Train accuracy | **100.0%** | Perfect on rows the model directly trained on — expected, and on its own not meaningful (see overfitting, above) |
| Test accuracy | **99.09%** | On the 20% of rows the model never trained on, it got roughly 218 of 220 exactly right |
| 5-fold CV accuracy | **99.50% ± 0.22%** | Averaged over five different train/test splits of the whole dataset — the most trustworthy single number, and it closely agrees with the held-out test accuracy, which is the sign of a genuinely well-generalizing model rather than a lucky split |

The small gap between 100% (train) and ~99% (test/CV) is a *normal, healthy* amount of
overfitting for a tree-ensemble on clean, well-separated data like this — a large gap (e.g. 100%
train vs. 70% test) would instead be a warning sign that the model memorized noise rather than
learning a real pattern.

### 8.6 What a per-crop "profile" is, and how the quartiles are computed

Beyond the classifier itself, training also computes, per crop, the **quartile range** of each
feature among that crop's own 100 training rows:
```python
def crop_profiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for crop, g in df.groupby(TARGET):
        item = {}
        for f in FEATURES:
            q1, med, q3 = g[f].quantile([0.25, 0.5, 0.75])
            item[f] = {"q1": float(q1), "median": float(med), "q3": float(q3)}
        profiles[crop] = item
    return profiles
```
`df.groupby(TARGET)` splits the full 2,200-row table into 22 smaller tables, one per crop label.
For each crop's own 100 rows, `.quantile([0.25, 0.5, 0.75])` finds three cut points along each
feature's sorted values: **Q1** (the value below which the lowest 25% of that crop's recorded
values fall), the **median** (the exact midpoint — 50% below, 50% above), and **Q3** (below
which 75% fall). The range **Q1 to Q3** is called the **interquartile range (IQR)** — it
describes where the "typical middle half" of that crop's real recorded conditions sat, resistant
to a few unusual outlier rows skewing the range the way a plain min/max would. This per-crop
profile — completely separate from the classifier's own learned weights — is what powers the
"your Nitrogen of 40 is below rice's typical range of 60–99" style feedback later in
[Part 12](#part-12--suitability-checks-and-remedies-mlsuitabilitypy).

---

## Part 9 — The crop model, line by line (`ml/crop_model.py`)

This file has already been quoted piece by piece above; here is the remaining connective detail
of how a *trained* model actually turns 7 fresh numbers into an answer at request time:

```python
def predict(bundle: CropBundle, values: dict[str, float], top_k: int = 8) -> dict[str, Any]:
    X = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
    probs = bundle.model.predict_proba(X)[0]
    classes = bundle.model.classes_.tolist()
    probabilities = {crop: float(probability) for crop, probability in zip(classes, probs)}
    order = np.argsort(probs)[::-1][:min(top_k, len(classes))]
    top_crops = [{"crop": classes[i], "probability": round(float(probs[i]), 6)} for i in order]
    best_index = int(np.argmax(probs))
    return {
        "predicted_crop": classes[best_index],
        "probability": round(float(probs[best_index]), 6),
        "probabilities": probabilities,
        "top_crops": top_crops,
    }
```
- `pd.DataFrame([{...}])` wraps the single incoming request as a one-row table — scikit-learn
  models expect a table-shaped input (rows × named columns) even for a single prediction, so
  this matches the exact shape the model was trained on.
- `.predict_proba(X)` is the key method: rather than only returning one crop name, it returns
  the full probability the trained `VotingClassifier` assigns to *every one* of the 22 classes
  for this input — this is what powers the "top 8 candidate crops" ranking in the UI, not just a
  single best guess.
- `np.argsort(probs)` returns the *indices* that would sort `probs` from smallest to largest;
  `[::-1]` is Python's slice syntax for "reverse the order" (largest to smallest); `[:top_k]`
  then keeps only the first 8 of those — together, "give me the indices of the 8 highest
  probabilities, highest first."
- `np.argmax(probs)` directly finds the index of the single highest probability — the model's
  best guess.

`load_bundle(...)` on the other side does the reverse of `train()`'s saving step: `joblib.load(
models_dir / "crop_voting.joblib")` deserializes the trained `VotingClassifier` Python object
straight back out of the binary file it was saved to (via `joblib.dump`) — this is *why* the
server can answer instantly on every request instead of retraining a model from scratch every
time, and why you never need to run any training script just to use the app: the file already on
disk in `models/` is a complete, ready-to-use snapshot of an already-trained model.

---

## Part 10 — Explaining the AI's decision (`ml/explainability.py`)

A machine-learning ensemble like this one is, mathematically, a black box: it's easy to ask "what
did it predict," but much harder to answer "*why* did it predict that" in a way a human can
sanity-check. **SHAP** (**SH**apley **A**dditive ex**P**lanations) is a technique, based on a
concept from cooperative game theory (the *Shapley value* — originally invented to fairly split a
group's collective winnings among individual contributors based on how much each one actually
added), adapted here to fairly attribute a prediction's outcome among the input features that
produced it: for each of the 7 features, SHAP estimates "if this feature's value had been
completely absent/neutral instead, how much would the predicted probability have moved?" — a
signed, feature-by-feature scorecard.

```python
explainer = shap.Explainer(bundle.model.predict_proba, background_rows)
exp = explainer(x, max_evals=200)
```
`shap.Explainer` needs two things: the function whose output it's explaining
(`bundle.model.predict_proba` — "given inputs, give probabilities"), and a small
**background dataset** — a reference point representing "typical"/"neutral" inputs, against
which every feature's individual contribution is measured relative to. This code builds that
background set out of the median feature values of each of the 22 crop profiles computed earlier
(Part 8.6) — a lightweight stand-in for a full random sample of the training data, kept small
(capped at 20 rows) purely so the explanation computation stays fast enough to run live on every
request rather than needing to be precomputed offline. `max_evals=200` caps how many internal
test evaluations SHAP is allowed to run while estimating the attribution, trading a small amount
of numerical precision for a fast enough response time for an interactive web request.

```python
if vals.ndim == 3:
    vals = np.abs(vals).mean(axis=2)
```
Because this is a *multi-class* model (22 possible outputs, not one), raw SHAP values come back
as a 3-dimensional array (one slice per crop class). Since the UI only wants one overall "how
much did Nitrogen matter" bar per feature (not 22 separate ones), this line takes the absolute
value of each class's contribution (so a feature that strongly pushed *toward* one crop and
strongly pushed *away* from another doesn't cancel out to a misleading near-zero) and averages
across all 22 classes, collapsing back down to one importance number per feature.

Finally:
```python
except Exception as exc:
    return [{"feature": "unavailable", "impact": 0.0, "importance": 0.0, "note": str(exc)}]
```
SHAP is deliberately treated as a "nice-to-have," not a hard requirement — if it fails for any
reason (it can be a finicky, version-sensitive library), the function catches literally *any*
exception and hands back a clearly-labeled placeholder rather than crashing the entire analysis
request over what is ultimately a secondary feature.

---

## Part 11 — The hybrid recommender for 182 crops (`ml/hybrid_recommender.py`)

### 11.1 The problem this file solves

The trained ML classifier only ever learned about **22 crops** — because that's the only crops
the labeled training spreadsheet contains real recorded soil/weather data for. But the project's
full reference catalog (`data/reference/india_crop_catalog.json`) lists **182** Indian crops.
Rather than either (a) pretending the model "knows" all 182 crops (which would be dishonest — it
has no trained evidence for the other 160), or (b) simply hiding the other 160 from the UI
entirely, this file implements a middle ground: a transparent, clearly-labeled, rule-based
scoring method for exactly those 160 extra crops, built from broad agronomic knowledge about
each crop's *category* (cereals, pulses, fruits, spices, ...) rather than crop-specific measured
data — and the code is careful to tag every single score with which of the two methods produced
it, all the way out to the final API response (`model_supported: true/false`,
`source: "hybrid_ml_and_profile"` vs `"reference_profile_suitability"`).

### 11.2 The suitability-score formula, explained

```python
def _score_dimension(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        mid = (low + high) / 2.0
        half = max((high - low) / 2.0, 1e-9)
        return 1.0 - 0.25 * min(abs(value - mid) / half, 1.0)
    distance = (low - value) if value < low else (value - high)
    span = max(high - low, 1e-9)
    return max(0.0, 1.0 - 0.75 * (distance / span))
```
This scores exactly *one* feature (say, temperature) against one crop category's acceptable
range `[low, high]`, producing a single number from 0 (terrible fit) to 1 (perfect fit):

- **Inside the range:** the score starts at 1.0 (perfect) right at the exact midpoint of the
  range, and gradually drops toward 0.75 as the value approaches either edge of the range
  — reflecting "technically acceptable, but a plant does better closer to the ideal middle than
  right at the edge of tolerable conditions." `abs(value - mid) / half` measures how far from
  the center the value sits, as a fraction from 0 (dead center) to 1 (right at the edge); `min(
  ..., 1.0)` clamps that fraction so it can't exceed 1 even from floating-point rounding noise;
  multiplying by `0.25` scales the maximum possible penalty for being inside-but-off-center down
  to a gentle quarter-point deduction.
- **Outside the range:** the score drops below 0.75, falling further the *farther outside* the
  range the value is, until it hits a hard floor of 0.0 once the distance outside equals the
  entire width of the range itself (`distance / span >= 1.0` combined with `max(0.0, ...)`) — so
  a value only *slightly* outside the acceptable range is scored as "borderline poor" rather than
  "as bad as it can possibly get," but a value wildly outside gets clamped to the worst possible
  score rather than going negative.
- `max(..., 1e-9)` in both branches is a standard defensive technique to prevent a
  **divide-by-zero** crash in the rare edge case where `low` and `high` happen to be equal —
  `1e-9` (scientific notation for `0.000000001`) is small enough to have no real effect on the
  math while guaranteeing the denominator is never literally zero.

`build_catalog_profiles(...)` decides, per catalog crop, which set of ranges to score it
against: if the crop's name maps onto one of the 22 trained classes (via the `_alias(...)` lookup
table, which handles cases like the catalog saying `"Grape"` while the trained model's class is
spelled `"grapes"`), it reuses that crop's *own, empirically measured* quartile profile from
Part 8.6 — the best available data. Otherwise, it falls back to `CATEGORY_PROFILES`, a
hand-curated dictionary of broad, deliberately wide agronomic ranges per major crop category
(defined explicitly at the top of the file, with a code comment noting they are "intentionally
broad, indicative profiles... NOT presented as measured training data").

### 11.3 Combining per-feature scores into one crop score, and blending in the ML model

```python
heuristic = float(np.mean(dims)) if dims else 0.0
...
if entry["model_supported"]:
    final_score = 0.60 * heuristic + 0.40 * ml_probability
    source = "hybrid_ml_and_profile"
else:
    final_score = heuristic
    source = "reference_profile_suitability"
```
`np.mean(dims)` averages the 7 individual per-feature scores from Part 11.2 into one overall
"heuristic" fit score for that crop. Then, **only for the 22 crops the trained model actually
knows**, that heuristic is blended 60/40 with the trained classifier's own predicted probability
for that crop (from Part 9's `predict()`) — giving real statistical training evidence a
majority-but-not-total say, while still keeping the profile-range check in the mix (a crop the
model is statistically confident about but whose specific numeric inputs are agronomically
implausible still gets pulled down somewhat). For the 160 crops with no trained-model opinion to
blend in at all, the heuristic category-range score stands entirely on its own.

```python
total = sum(max(x["score"], 0.0) for x in scored)
if total > 0:
    for x in scored:
        x["probability"] = x["score"] / total
```
Finally, all 182 raw scores are **normalized** — each one divided by the sum of all of them — so
that the full set of "probabilities" sums to exactly 1.0 across all 182 crops, making them
directly comparable to (and displayable identically alongside) genuine model probabilities, even
though, for the 160 reference-only crops, they are really just a rescaled heuristic rather than a
statistically calibrated probability — again, this distinction is preserved and exposed via the
`source` field rather than hidden.

---

## Part 12 — Suitability checks and remedies (`ml/suitability.py`)

### 12.1 `feature_gap` — comparing your numbers against a specific crop's real learned profile

Already fully quoted and explained in Part 5.4 (the `if`/`elif`/`else` example) — it computes,
per feature, whether your value falls `"low"`, `"high"`, or `"within"` relative to that crop's
empirical Q1–Q3 range (Part 8.6), plus a numeric `severity`: how many "IQR-half-widths" away from
the boundary you are (e.g. severity `2.0` roughly means "twice as far outside the typical range
as the range's own typical half-width" — a scale-free way of comparing severity across features
with very different natural units, like pH's 0–14 scale versus rainfall's 0–3000mm scale).

### 12.2 `agricultural_suitability` — "is this land viable for farming at all?"

```python
def agricultural_suitability(values, ranked, bundle):
    supported = 0
    for crop, p in profiles.items():
        ok_count = 0
        for f in FEATURES:
            q1, q3 = p[f]["q1"], p[f]["q3"]
            if q1 <= float(values[f]) <= q3:
                ok_count += 1
        if ok_count >= 4:
            supported += 1
    top_prob = float(ranked[0]["probability"]) if ranked else 0.0
    suitable = supported > 0 and top_prob >= 0.20
    ...
```
This is a broader, coarser check than the per-crop target check below — it asks "across *all* 22
known crops, is there at least one whose typical range this land satisfies on a majority (4 of
7) of features, *and* is the model's own top pick reasonably confident (≥20%) about something?"
It's deliberately generous (a crop only needs 4 of 7 features within range to count as
"supported," and 20% is a low confidence bar out of 22 possible classes — for context, guessing
uniformly at random across 22 classes would only be about 4.5% each) — the intent is to flag
truly extreme, unfarmable input combinations, not to be a strict per-crop judgment (that's what
`target_suitability`, next, is for).

### 12.3 `target_suitability` — "will *this specific* crop grow here?"

```python
suitable = prob >= 0.35 and len([g for g in major if g["severity"] > 1.0]) <= 2
```
A stricter, crop-specific version: the trained model's predicted probability for *this exact
crop* must be at least 35%, **and** at most 2 of its features are allowed to be "severely" out of
range (`severity > 1.0`, i.e. further outside the typical range than the range's own typical
half-width). Both conditions must hold (`and`) — a crop can't pass just by being the model's
statistically top guess if several of its actual input numbers are agronomically implausible for
it, and it can't pass purely on "only 1 bad feature" if the model's own confidence in it is
low.

### 12.4 `remedies` — turning a numeric gap into a plain-English recommendation

```python
if f in {"N", "P", "K"}:
    nutrient = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}[f]
    action = (f"Increase {nutrient} using a soil-test-guided fertilizer or organic amendment. "
              if direction == "low" else
              f"Reduce additional {nutrient} inputs and avoid over-fertilization; re-test the soil before applying more.")
```
`{"N", "P", "K"}` here is a **set literal** — an unordered collection used purely to efficiently
answer "is this value one of these three?" (`in` membership checks). For each out-of-range
feature, the function picks one of a small number of hand-written, direction-aware
(`"low"` vs `"high"`) template sentences — deliberately conservative, actionable, real-world
agronomic advice ("soil-test-guided," "consult an agronomist") rather than overconfident
one-line commands, reflecting that this whole layer is decision *support*, not an authoritative
agronomic prescription (a caveat the frontend also displays directly to the user, in the input
panel's footnote).

---

## Part 13 — The soil photo classifier: a neural network (`ml/soil_model.py`)

### 13.1 Why this needs a completely different kind of algorithm

Random Forest, Logistic Regression, and Gradient Boosting all expect a small, fixed number of
meaningful *named* numeric inputs (7, here) — the kind of structured data you'd find in a
spreadsheet. A photograph is nothing like that: a single 224×224 pixel color photo is really
224 × 224 × 3 (three color channels: Red, Green, Blue) = **150,528 individual raw numbers**, and
critically, no single one of those numbers is individually meaningful the way "Nitrogen level"
is — what matters is the *spatial pattern* across many neighboring pixels (an edge, a texture, a
color gradient). This is exactly the kind of problem **Convolutional Neural Networks (CNNs)**
were designed for.

### 13.2 What a neural network is, conceptually

A **neural network** is built from layers of simple units ("neurons"), each one computing a
weighted sum of its inputs (conceptually similar to the Logistic Regression weighted sum from
Part 8.4) followed by a small non-linear "activation" step, then passing its output onward to
the next layer. Stacking many such layers lets the network build up increasingly abstract
representations — early layers might learn to detect simple edges and color blobs, middle layers
combine those into textures and shapes, and later layers combine *those* into whole recognizable
concepts. Like the classical models in Part 8, the network's many weights (here: millions of
them) are learned automatically from labeled examples via gradient descent — nobody hand-writes
"if there are more than 40% reddish pixels, it's Red soil"; the network discovers useful visual
patterns entirely on its own by repeatedly being shown labeled photos and adjusting its internal
weights to reduce prediction error.

A **Convolutional** layer, specifically, is a neural-network layer specialized for images: rather
than connecting every input pixel to every neuron (which would be both enormous and wasteful),
it slides a small grid of learned weights (a "filter" or "kernel," e.g. 3×3 pixels) across the
entire image, reusing the *exact same* small set of weights at every position — this exploits the
fact that a useful pattern like "an edge" is a small, local shape that can usefully be detected
the same way no matter *where* in the photo it appears, drastically reducing the number of
weights the network needs to learn compared to a naively fully-connected approach.

### 13.3 Transfer learning: reusing a network someone else already trained

Training a large CNN completely from scratch normally requires millions of labeled images and
substantial computing power — far more than this project's few hundred soil photos could ever
support well on their own. Instead, this project uses **transfer learning**:
```python
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base.trainable = False
```
**MobileNetV2** is a well-known, publicly available CNN architecture that Google originally
trained on **ImageNet** — a famous benchmark dataset of over a million photos across 1,000
everyday object categories (dogs, cars, chairs, and so on). `weights="imagenet"` downloads that
*already-trained* set of millions of weights rather than starting from random values. The
intuition: a network that has already learned to recognize general-purpose edges, textures, and
color patterns from a million diverse photos has effectively already learned most of the
low-level visual vocabulary any image task needs — including telling apart soil types, even
though MobileNetV2 was never shown a single labeled soil photo during its original training.
`include_top=False` strips off MobileNetV2's *own* final classification layer (which was
originally wired to output one of the 1,000 ImageNet categories — meaningless here), keeping
only the general-purpose visual "feature extractor" underneath it. `base.trainable = False`
**freezes** all of those millions of pretrained weights — they are used as-is and never
adjusted during this project's own training, both because there isn't nearly enough soil-photo
data to safely retrain such a large network without it just memorizing the small dataset, and
because it makes training vastly faster (only the small new layers added on top actually need
to learn anything).

```python
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.08)(x)
x = layers.RandomZoom(0.1)(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(4, activation="softmax")(x)
```
Reading this top to bottom, in the order data actually flows through it:

- **`RandomFlip` / `RandomRotation` / `RandomZoom`** — **data augmentation**: during training
  only, each photo is randomly, slightly flipped/rotated/zoomed *before* being shown to the
  network, creating many effectively-different training examples out of one real photo. This
  artificially grows the effective size and diversity of a small dataset and teaches the network
  that soil type shouldn't depend on the exact camera angle or zoom level a particular photo
  happened to be taken at — improving how well it **generalizes** to new photos taken differently
  from the training set.
- **`preprocess_input`** — rescales raw 0–255 pixel values into the specific numeric range
  MobileNetV2 was originally trained expecting (roughly -1 to 1) — every pretrained network
  expects inputs prepared exactly the way its own original training data was prepared.
- **`base(x, training=False)`** — runs the frozen, pretrained MobileNetV2 feature extractor,
  producing a compact grid of abstract visual features rather than raw pixels.
- **`GlobalAveragePooling2D`** — collapses that grid of spatial features down into a single flat
  list of numbers by averaging each feature map, discarding the leftover "where in the photo"
  positional detail (soil-type classification doesn't care *where* in the frame the soil texture
  appears, only that it does).
- **`Dropout(0.25)`** — a well-known **regularization** technique used only during training: on
  each training step, it randomly and temporarily "turns off" 25% of the neurons at that point in
  the network, forcing the remaining ones to not overly rely on any single specific neuron always
  being present — another defense against overfitting on a small dataset.
- **`Dense(4, activation="softmax")`** — the actual new, trainable part: a plain fully-connected
  layer producing exactly 4 numbers (one per soil class: Alluvial, Black, Clay, Red), passed
  through **softmax** (the same all-outputs-sum-to-1 probability squashing function mentioned
  under Logistic Regression in Part 8.4, generalized here to 4 categories instead of 22).

### 13.4 Compiling and training

```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
**`Adam`** is a specific, very widely-used variant of the gradient-descent optimization algorithm
introduced back in Part 8.4 — it adapts how big a step to take for each individual weight based
on that weight's own recent history of updates, generally converging faster and more reliably
than plain gradient descent on deep networks. `1e-3` (`0.001`) is the **learning rate** — a
knob controlling how large each weight-adjustment step is; too high and training can overshoot
and never settle down, too low and training crawls unnecessarily slowly.
**`sparse_categorical_crossentropy`** is the loss function (recall Part 8.4: a formula scoring
"how wrong were the predictions") appropriate for multi-class classification where the true
label is given as a plain integer class index (0, 1, 2, or 3) rather than already expanded into a
one-hot vector like `[0,0,1,0]` — "sparse" here refers to that integer-label input format, not to
the data itself.

```python
cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(models_dir / "soil_mobilenetv2.keras", monitor="val_accuracy", save_best_only=True),
]
hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
```
Training runs for up to `epochs` full passes over the training photos (one **epoch** = one
complete pass through the entire training set). `validation_split=0.2, subset="both"` (set when
the dataset was first loaded, further up in the file) held back 20% of the soil photos purely
for checking progress during training, never trained on directly — the same train/held-out-set
principle from Part 8.3, applied here to images. **`EarlyStopping`** watches the held-out
validation accuracy after every epoch and automatically stops training early if it hasn't
improved for `patience=5` epochs in a row — a direct, practical defense against overfitting
(more epochs isn't automatically better; past some point, more training just means more
memorizing of the training photos specifically, at the cost of accuracy on new ones), and
`restore_best_weights=True` ensures the network actually ends up keeping the weights from
whichever epoch scored best on validation, not simply whatever epoch it happened to stop on.
**`ModelCheckpoint`** separately saves a copy of the model to disk (`save_best_only=True`)
every time validation accuracy hits a new best, so the final `.keras` file on disk is always the
single best-performing version seen across the entire run.

**This project's actual soil-model numbers** (from `models/soil_metrics.json`, trained on 529
total photos — 371 training / 158 held-out validation — across the 4 classes): **test/validation
accuracy ≈ 88.0%**. Notably lower than the crop model's ~99% — expected, and explained directly
by the training data itself: there are only 529 total photos spread across 4 classes (compared
to 2,200 rows across 22 classes for the crop model), and the classes are visibly imbalanced
(56 Clay training photos versus 125 Red), both of which make this a meaningfully harder learning
problem with less evidence to learn from.

### 13.5 Using the trained model on a new photo

```python
def predict(model, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(tf.argmax(probs))
    return {"soil_type": CLASS_NAMES[idx], "probability": float(probs[idx]), ...}
```
`load_img(..., target_size=IMG_SIZE)` opens the uploaded file and resizes it to the same 224×224
size every training photo was resized to (a network's input layer has a fixed, unchangeable
shape). `img_to_array` converts the image into the raw numeric pixel grid the network actually
operates on. `tf.expand_dims(arr, 0)` adds an extra dimension at the front purely because, exactly
like the crop model's `predict()` from Part 9, the network expects a *batch* of images (even a
batch of exactly one) rather than a single bare image. `tf.argmax(probs)` picks out the single
highest-probability class, and `CLASS_NAMES[idx]` (`["Alluvial", "Black", "Clay", "Red"]`) turns
that numeric index back into the human-readable soil-type name shown in the UI.

---

## Part 14 — The 182-crop reference catalog

[`data/reference/india_crop_catalog.json`](data/reference/india_crop_catalog.json) is plain,
static reference data — not learned by any model, just a curated list. Each entry looks like:
```json
{
  "crop": "Rice",
  "major_category": "Field Crops – Cereals",
  "crop_group": "Rice",
  "varieties": "Basmati; Pusa Basmati 1121; ...",
  "system_notes": "Wetland/paddy systems; irrigated or rainfed; kharif and some rabi/summer systems."
}
```
It's organized into 12 major categories (Cereals, Millets, Pulses, Oilseeds, Fibre Crops, Fruits,
Vegetables, Plantation Crops, Spices, Specialized Horticulture, Forage Crops, and Pseudo-cereals)
totaling exactly 182 entries — a count and per-category breakdown that
[`scripts/validate_crop_catalog.py`](scripts/validate_crop_catalog.py) exists purely to
double-check (it simply loads the JSON, counts entries per category, and uses Python's `assert`
statement — a line that instantly crashes the program with an error if the condition after it is
`False` — to confirm nothing was accidentally added, removed, or miscategorized). It's this file
that the crop catalog browser at the bottom of the webpage reads and displays, and that
`app/crop_catalog.py` cross-references against the trained model's 22 known classes.

---

## Part 15 — The frontend, file by file

### 15.1 `index.html` — the empty shell

```html
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
```
This is the *only* real, literal HTML file in the whole frontend. Notice it's almost completely
empty — just one `<div id="root">` and one `<script>` tag. Everything you actually see on the
page is generated dynamically by JavaScript *after* the page loads — this pattern is called a
**Single Page Application (SPA)**. `type="module"` tells the browser this script uses the modern
`import`/`export` JavaScript module system (letting `main.jsx` itself `import` from other files,
exactly like Python's `import` from Part 5.6).

### 15.2 `main.jsx` — mounting React onto the page

```jsx
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```
`document.getElementById('root')` finds that empty `<div id="root">` from `index.html`.
`ReactDOM.createRoot(...).render(<App />)` tells React "take over this specific empty box, and
continuously keep its contents in sync with whatever the `App` component currently describes" —
this one call is the entire bridge between the static HTML file and the dynamic React
application. `<React.StrictMode>` wraps the app in an extra development-only safety net that
deliberately runs certain functions twice to help surface accidental bugs early (side effects
that aren't safely repeatable) — it has no effect on the final production build.

### 15.3 `App.jsx` — already covered

The component structure, state, effects, list rendering, and conditional rendering were all
covered with real snippets from this exact file throughout [Part 6](#part-6--javascript-react-and-html-crash-course).
The one remaining thing worth calling out explicitly is how tightly the JSX mirrors the backend's
JSON response shape from `app/services.py` — for example `result.agriculture_suitability.status`,
`result.selected_crop.details`, and `result.explanation` in `AnalysisResult(...)` are reading
*exactly* the field names built by `analyze()` in Part 7.3, field for field — the frontend and
backend agree on this shape entirely by convention (there's no formal schema file enforcing it);
if a backend field is ever renamed, the corresponding frontend reference has to be updated by
hand to match, or that piece of the UI will silently show nothing/`undefined`.

### 15.4 `vite.config.js` — the dev server and the proxy

```js
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:5001' } },
  build: { outDir: 'dist', emptyOutDir: true },
})
```
`plugins: [react()]` wires in the JSX-to-JavaScript translation mentioned in Part 6.2.
`proxy: { '/api': ... }` is what makes the browser able to call `fetch('/api/analyze')` (a
relative address, no host at all) throughout `App.jsx` without ever hardcoding a full backend
URL: any request the browser makes to Vite's own dev server starting with `/api` is silently
forwarded by Vite itself to the real Flask backend at `127.0.0.1:5001`, and Vite relays the
response straight back — from the browser's point of view, it only ever talked to one single
address the whole time. `build: { outDir: 'dist' }` configures what happens when you eventually
run `npm run build` instead of `npm run dev` — rather than a live dev server, this bundles the
entire React app down into a folder of optimized, static `.html`/`.js`/`.css` files (named
`dist/`, hence `outDir`) that could be hosted anywhere, including served directly by the Flask
backend itself — which is exactly what the fallback logic at the bottom of `app/api.py`
(`react_app(path)`, checking `if not FRONTEND_DIST.exists()`) is prepared to do, letting the
whole project run as a single combined server in a production deployment instead of two separate
dev processes.

### 15.5 `styles.css`

Plain CSS (Cascading Style Sheets — the language that controls a web page's visual appearance:
colors, spacing, fonts, layout — completely separate from HTML's job of describing *content
structure*). A few patterns worth knowing the names of, since they're used throughout this file:
- **CSS custom properties** (variables), declared once and reused everywhere:
  `--green: #0f6b45;` defined inside `:root { ... }`, then reused elsewhere as `var(--green)` —
  change the color in exactly one place and it updates everywhere it's referenced, rather than
  having to find and replace the same color code dozens of times throughout the file.
- **Flexbox** (`display: flex`) and **CSS Grid** (`display: grid`) — two different modern
  layout systems for arranging boxes in a row/column (`flex`) or a full two-dimensional
  grid (`grid`) without needing old-fashioned manual pixel positioning.
- **Media queries** (`@media (max-width: 760px) { ... }`) — CSS rules that only apply when the
  browser window is narrower than a given width, which is how this page rearranges itself into a
  single readable column on a phone screen instead of staying in its multi-column desktop layout.

---

## Part 16 — The full API reference

All endpoints are served from the Flask backend, normally reachable during development through
Vite's `/api` proxy from the browser at `http://localhost:5173/api/...`.

### `GET /api/health`
```json
{ "status": "ok", "crop_model": true, "soil_model": true }
```

### `GET /api/crops`
```json
{ "crops": ["Rice", "Wheat", "...(all 182)"], "model_crops": ["apple", "banana", "...(22)"] }
```

### `GET /api/crop-catalog`
Returns the full 182-entry catalog, each tagged `model_supported` and `enabled`, plus summary
counts:
```json
{
  "count": 182,
  "unique_crops": 182,
  "model_classes": 22,
  "categories": ["Commercial & Fibre Crops", "..."],
  "crops": [ { "crop": "Rice", "major_category": "Field Crops – Cereals", "model_supported": true, "..." } ]
}
```

### `GET /api/metrics`
Returns the raw contents of `models/crop_metrics.json` verbatim (training sample count, class
list, train/test/CV accuracy, full classification report, confusion matrix).

### `POST /api/analyze`
Request body (JSON):
```json
{ "N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202, "target_crop": null }
```
Response (abbreviated — the real response also includes `top_crops`, `agriculture_suitability`,
`reasons`, `remedies`, `explanation`, and `soil_image_analysis`, all described in Part 7.3):
```json
{
  "inputs": { "N": 90, "...": "..." },
  "prediction": { "crop": "rice", "probability": 0.91, "source": "hybrid_ml_and_profile" },
  "selected_crop": { "crop": "rice", "suitable": true, "status": "Suitable for the selected crop", "details": ["..."] }
}
```

### `POST /api/analyze-with-image`
Identical to `/api/analyze`, but sent as `multipart/form-data` (a file-upload-capable request
format) instead of plain JSON, with the same 7 fields sent as individual form fields plus a
`soil_image` file field. Adds a populated `soil_image_analysis` in the response:
```json
{ "soil_type": "Red", "probability": 0.94, "probabilities": { "Alluvial": 0.02, "Black": 0.03, "Clay": 0.01, "Red": 0.94 } }
```

---

## Part 17 — Training your own models from scratch

You do **not** need to do this to use the app — pre-trained files already exist in `models/`.
This is only for retraining on new/different data.

**Retraining the crop classifier:**
```bash
python scripts/download_crop_dataset.py     # downloads the public Kaggle CSV via kagglehub
python scripts/train_crop_model.py          # trains and overwrites models/crop_voting.joblib etc.
```
The first command needs a (free) Kaggle account and API credentials configured for `kagglehub`
to authenticate — see Kaggle's own documentation for generating an API token if prompted.

**Retraining the soil classifier:** requires your own labeled photo dataset first, arranged as:
```
data/raw/soil/
├── Alluvial/  (photo files)
├── Black/     (photo files)
├── Clay/      (photo files)
└── Red/       (photo files)
```
(the folder name for each subfolder *is* the label — this is what `labels="inferred"` in
`ml/soil_model.py`'s `train()` relies on), then:
```bash
python scripts/train_soil_model.py
```

**Expanding beyond the 22 supervised classes:**
[`scripts/download_mendeley_agro_data.py`](scripts/download_mendeley_agro_data.py) downloads a
larger public agro-meteorological reference dataset (107 crops) from Mendeley Data, intended as
raw material for a *future* effort to expand real supervised training coverage beyond today's 22
crops — as of this version of the project, that data is downloaded but not yet wired into
`train_crop_model.py`; the 160 non-supervised catalog crops still use the
category-profile method from [Part 11](#part-11--the-hybrid-recommender-for-182-crops-mlhybrid_recommenderpy)
until/unless that expansion happens.

---

## Part 18 — Automated tests

A **test** is a small program written specifically to check that another piece of code behaves
correctly — instead of a human manually clicking through the app after every change to check
nothing broke, you write the check once as code, and re-run it instantly, as many times as you
like, forever. This project uses **`pytest`**, a Python testing framework that automatically
discovers and runs any function in the `tests/` folder whose name starts with `test_`.

```python
def test_health():
    app = create_app()
    client = app.test_client()
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
```
`app.test_client()` creates a fake, in-memory version of the Flask app that behaves exactly like
a real running server, without actually opening a network port — letting a test send it a
request and inspect the response instantly, without starting `run.py` separately. **`assert`**
is a Python statement (already mentioned in Part 14) that does nothing at all if the condition
after it is `True`, but immediately stops the program with a clear, specific error the moment it
is `False` — the fundamental building block of essentially all automated testing: state exactly
what *should* be true, and let the program itself catch you immediately if reality disagrees.

Run every test with:
```bash
pytest
```
from the project root, with the virtual environment active. `test_api_contract.py` confirms the
health check and a real `/api/analyze` call (using the exact same demo values shown as the
frontend's default input values) both succeed and return the expected shape —
`assert payload['prediction']['crop'] == 'rice'` specifically locks in that this project's known
demo inputs really do predict rice, so if a future code change ever accidentally broke that,
this test would immediately fail and say so. `test_suitability.py` checks the remedy-generation
logic from Part 12.4 in isolation, with no Flask server or trained model involved at all —
testing that one small function's behavior directly and cheaply, which is generally preferable
to testing everything only through the full, slower, end-to-end app whenever possible.

---

## Part 19 — Troubleshooting

**Backend seems to start but nothing ever responds (macOS).** Apple's AirPlay Receiver silently
listens on port 5000 by default, which is Flask's traditional default port. This project's
`run.py` is already configured for port **5001** specifically to avoid this — if you ever change
it back to 5000 on a Mac and hit this, either turn off System Settings → General → AirDrop &
Handoff → "AirPlay Receiver," or just pick a different port (and update
`frontend/vite.config.js`'s proxy to match — the two files must always agree on which port the
backend is running on).

**`InconsistentVersionWarning` from scikit-learn on startup.** The saved `models/crop_voting.
joblib` file was created with a slightly older scikit-learn version than the one currently
installed in your virtual environment. This is just a warning, not an error — the model still
loads and predicts correctly; scikit-learn simply can't guarantee that will always remain true
across every possible future version gap, so it flags it defensively. If it ever becomes a real
problem, re-running `scripts/train_crop_model.py` regenerates the file with your current
library version.

**`pip install -r requirements.txt` fails to find a matching `pandas`/`numpy` version.** Those
two libraries' newest versions require a newer Python than you have installed (this project was
written expecting **Python 3.11** specifically — not the very newest available Python release,
and not older releases like 3.9/3.10 either). Recreate the virtual environment using `python3.11`
explicitly, as shown in Part 2.4.

**The frontend loads but every request fails / shows a network error.** Almost always means the
backend terminal isn't actually running, or crashed. Check that terminal's own output for a
Python error message — it will usually say exactly what went wrong.

**"Address already in use" when starting the backend.** Some other program (possibly a previous,
still-running copy of this same backend you forgot to stop) is already using port 5001. Find and
stop it, or change the port in both `run.py` and `frontend/vite.config.js` together.

---

## Part 20 — Glossary of every technical word used above

- **API (Application Programming Interface)** — a defined set of addresses/messages one program
  exposes so other programs can ask it to do things, without needing to know how it works
  internally.
- **Argument / parameter** — a value handed into a function when it's called.
- **Array / list** — an ordered collection of values.
- **Backend** — the server-side program that holds logic/data and answers requests; has no
  visual interface of its own.
- **Class / object** — a blueprint (class) and a concrete thing built from it (object);
  Part 5.7.
- **CLI (Command-Line Interface)** — controlling a program by typing text commands in a
  terminal, instead of clicking a visual interface.
- **Component (React)** — a JavaScript function that returns a description of part of a UI;
  Part 6.3.
- **CORS** — a browser security rule restricting which web address a page is allowed to receive
  responses from; Part 7.1.
- **CSV (Comma-Separated Values)** — a plain-text spreadsheet format, one row per line, columns
  separated by commas.
- **Dependency / library / package** — pre-written, reusable code your own program relies on;
  Part 0.
- **Dict / dictionary / object (JS)** — a collection of `key: value` pairs looked up by name.
- **Ensemble** — a model made of several smaller models combined, aiming for better accuracy
  than any one alone; Part 8.4.
- **Epoch** — one complete pass through an entire training dataset during neural-network
  training; Part 13.4.
- **Feature** — one named numeric input to a model (Nitrogen, temperature, etc.).
- **Frontend** — the part of an app that runs in the user's browser and draws what they see.
- **Function** — a named, reusable block of instructions; Part 5.5.
- **Gradient descent** — the general algorithm used to train most ML models, by repeatedly
  nudging internal numbers in whichever direction reduces prediction error; Part 8.4.
- **HTTP / HTTPS** — the protocol (agreed-upon message format) browsers and servers use to talk
  to each other.
- **JSON (JavaScript Object Notation)** — a plain-text format for structured data, used
  virtually everywhere APIs exchange information; Part 1.
- **JSX** — HTML-like tag syntax written directly inside JavaScript, translated by a build tool
  before running in the browser; Part 6.2.
- **Label** — the known-correct answer attached to a training example (a crop name, a soil-type
  name).
- **Loop (`for`)** — repeating a block of instructions once per item in a collection; Part 5.3.
- **Model (trained)** — the saved output of a machine-learning training process: a set of
  learned numbers plus the rule for combining them into a prediction.
- **Module / package (Python)** — a reusable `.py` file, or a folder of them; Part 5.6.
- **Neural network / CNN** — a layered model of simple weighted-sum units, specialized for
  images in the "Convolutional" case; Part 13.2.
- **Overfitting** — when a model has effectively memorized its specific training examples rather
  than learning a genuinely general pattern, so it performs far worse on new, unseen data;
  Part 8.3.
- **Port** — a number identifying which specific program on a computer a network message is
  meant for.
- **Probability** — in this project's classifiers, a number from 0 to 1 (or shown as a
  percentage) expressing the model's estimated confidence in one particular answer, with every
  possible answer's probabilities summing to 1.
- **Quartile (Q1 / median / Q3)** — cut points splitting sorted data into quarters; Part 8.6.
- **Regularization** — any technique (like Dropout, EarlyStopping) deliberately used to reduce
  overfitting during training.
- **SHAP** — a technique for attributing a model's prediction back to individual input features;
  Part 10.
- **State (React)** — data a component remembers between renders, managed via `useState`;
  Part 6.4.
- **Supervised learning** — training a model on examples that already have the correct answer
  attached; Part 8.1.
- **Terminal / shell / command line** — a text-based window for typing commands to run programs
  directly; Part 0.
- **Type hint (Python)** — an optional annotation noting what kind of value a variable/function
  is expected to hold, for human readability/tooling, not strictly enforced at runtime;
  Part 5.5.
- **Variable** — a named storage location holding a value; Part 5.1.
- **Virtual environment** — an isolated, private copy of Python (and its installed libraries)
  for one specific project; Part 2.4.
