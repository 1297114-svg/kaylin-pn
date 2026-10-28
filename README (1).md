# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit app for exploring World Bank historical indicators and using polynomial models and a function-analysis lab for Honors Algebra 2, Precalculus, and Calculus.

## Files

- `app.py` — complete Streamlit app
- `requirements.txt` — minimal, unpinned dependencies
- `sources.md` — data source and indicator references

## Run locally

Use Python 3.10 or newer in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The World Bank API requires internet access. The app caches successful responses for one hour.

## Deploy with GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and connect your GitHub account.
4. Create an app from the repository, select the branch, and set the main file path to `app.py`.
5. Deploy. Review the app logs if installation or startup fails.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist

- [ ] Check country, indicator, unit, definition, retrieval date, and World Bank API link.
- [ ] Confirm missing API values remain missing and are not replaced with fabricated values.
- [ ] Edit a value, confirm the scenario label changes, then use **Reset to source data**.
- [ ] Select a degree for which there are fewer than degree + 1 distinct years; confirm the fit is blocked.
- [ ] Add a duplicate year or nonnumeric value; confirm validation messages appear.
- [ ] Calculate one year and include that same year in a batch; estimates should agree.
- [ ] Test reversed batch ranges, malformed custom years, and years outside 1950–2100.
- [ ] Change sample increment and forward horizon; confirm the graph changes.
- [ ] Change comparison country; confirm its line and coverage message update.
- [ ] Download and open the CSV, Excel workbook, and PDF.
- [ ] In Function Lab, test `x**2`, `1/x`, `log(x)`, `exp(-k*x)`, and `sin(w*x)` with suitable intervals.
- [ ] Inspect derivative plots and numerical critical/inflection point scans; compare with hand calculations.
- [ ] Deploy and check the service logs for missing packages, port issues, and API timeouts.

## Important limitations

- I have not executed this app in the user's deployment environment. Treat it as code to verify, not as a claim of tested deployment.
- Polynomial models are descriptive, not causal explanations. High-degree fits may oscillate and extrapolate implausibly.
- R² and MAE shown for the fitted observations are in-sample. The chronological holdout MAE is separate and only appears when enough observations are available.
- Numerical derivative scans can miss repeated roots and do not prove exact extrema or inflection points. The app labels detected points as numerical approximations.
- Function input is restricted to a limited mathematical expression syntax. It is not a full computer algebra system or a security boundary for hosting untrusted code.
- The app's contextual examples are educational models. Users must choose units and interpretations that make sense for their scenario.
- The current function lab includes function, derivative, tangent/secant, and numerical calculus exploration. It does not implement a continuous autonomous animation loop; Streamlit controls update on interaction.
