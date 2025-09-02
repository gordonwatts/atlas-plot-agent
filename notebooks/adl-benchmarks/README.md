# ADL Benchmarks

The files in this folder contain examples of using various AI systems to generate code for the [ADL benchmarks](https://github.com/iris-hep/adl-benchmarks-index) along with 6 extra questions proposed by @rrutaa and @davidlange6.

## Questions

See the original [site](https://github.com/iris-hep/adl-benchmarks-index) for the raw version of the questions. Of course, those questions were never meant to be fed to an AI, so some further context (e.g. datasets) had to be added. The questions are listed below:

1. Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the pT of all jets in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the pT of jets with |η| < 1 in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the ETmiss of events that have at least two jets with pT > 40 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the ETmiss of events that have an opposite-charge muon pair with an invariant mass between 60 and 120 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697.
1. For events with at least three jets, plot the pT of the trijet four-momentum that has the invariant mass closest to 172.5 GeV in each event and plot the maximum b-tagging discriminant value among the jets in this trijet in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697.
1. Plot the scalar sum in each event of the pT of jets with pT > 30 GeV that are not within 0.4 in ΔR of any light lepton with pT > 10 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697.
1. For events with at least three light leptons and a same-flavor opposite-charge light lepton pair, find such a pair that has the invariant mass closest to 91.2 GeV in each event and plot the transverse mass of the system consisting of the missing transverse momentum and the highest-pT light lepton not in this pair in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697.
1. Plot the number of jets per event in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the electron transverse momentum for electrons with p_T > 25 GeV in the first 20000 events of the rucio dataset mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697.
1. Plot the leading jet p_T (highest p_T jet per event) in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.
1. Plot the invariant mass of all muon pairs with the same charge in the mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697.
1. Plot the missing transverse energy for events with no jets in the mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697.
1. Plot delta-R between the two leading jets for events with at least two jets in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

Note: these DID's had files in them when the tests were run! Sample deletion comes for us all!

Note: The above questions are parsed by code. Keep the format the same (each question is on one line, starts with a `1.`.)

## Process

Two experiments are cataloged here: a steered query and a cut-and-paste query.

1. Steered Query - a python script interacts with a LLM API and a local docker container to try to produce the plots.
1. Cut-And-Paste Query - use a web-based LLM, paste the questions (and hint files), take answer and run it in a local notebook, making modifications until it works.

The _Steered Query_ is much more developed. The `Cut-And-Paste` query was a by-hand experiment.

In all cases, the LLM is guided to do the following:

1. Use [`ServiceX`](https://servicex-frontend.readthedocs.io/en/stable/) to fetch the data from the dataset on the GRID
1. Use [`Awkward`](https://awkward-array.org/doc/main/index.html) and [`vector`](https://vector.readthedocs.io/en/latest/index.html) to process the data (filter, combine, 4-vector math, etc.)
1. Use [`hist`](https://hist.readthedocs.io/en/latest/) to build the histogram and then plot it.

### Steered Query

You can find results for two types of steered queries:

1. [Direct Query](direct-query.py): The query contains everything and the LLM is asked to produce the code start-to-finish in one go ([config-info](direct-query-config.yaml)). **Results** can be found in [this notebook](results/direct-query/analysis.ipynb), and [this directory](results/direct-query/) contains a markdown file for each question with entries for every model, including LLM response, code, and results of running the code. If the code fails to run, the LLM has a chance to update it.
1. [Plan Query](plan-query.py): The plotting is broken down into steps: planning, implementing the data fetch, data manipulation, and plotting ([config-info](plan-query-config.yaml)). **Results** can be found in [this notebook](results/plan-query/analysis.ipynb), and [this directory](results/plan-query/) contains a markdown file for each question with entries for every model, including LLM response, code, and results of running the code. If the code fails to run, the LLM has a chance to update it.

Notes:

* [Hint files are available here](https://github.com/gordonwatts/hep-programming-hints) (and are used by everything, including the cut-and-paste).

#### Running the analysis

You can use [`direct-query.py`](direct-query.py) and [`plan-query.py`](plan-query.py) to run a model and question (use --help to see options). Use [`query-all.py`](query-all.py) to run against all questions listed in this `README.md` file. You'll need `docker` installed in your directory, and you'll need to do a `docker build` on the `Dockerfile`. A docker volume is used to cache downloads from servicex to improve re-running speed.

Start to finish running everything and updating the reports:

1. Run the `query-all` to re-run the direct or plan queries
1. Copy the `.md` files and the `img` directory into the `notebooks/adl-benchmark/results/xxx` directory
1. Look through each output `.md` file and look for hallucinations. Record these in the `query-analysis.ipynb` file you find in this directory.
1. Run the following `papermill` commands to generate the _reports_ in each results directory:

```bash
cd notebooks/adl-benchmarks
papermill query-analysis.ipynb results/direct-query/analysis.ipynb -p results_dir direct-query
papermill query-analysis.ipynb results/plan-query/analysis.ipynb -p results_dir plan-query
```

How did I come up with what models to test?

* Ones that I have used before
* Ones recommended by others
* The [openrouter](https://openrouter.ai/rankings?category=programming#categories) leader board for code.

The complete list can be pulled from the [models.yaml](models.yaml) file.

### Cut-and-Paste Query

This is older and how I tested out this work by hand.

#### Codespaces

**Models:** Used a variety of models (`o4-mini`, `gpt-4.1`, `Claude Sonnet 4`). The best was probably `Claude Sonnet 4`. Questions 1-5 were simple enough that `4.1` could handle the task. After that `o4-mini` or `Claude`. That said, no one could do 6-8 in a single shot. All required modifications in order to make them run. Errors ranged from simple `ak.stack` rather than `np.stack`, so more subtle masking errors. Errors were often ones I'd have made as a person writing this code. Another interesting fact - if a model generated an answer I didn't like, I could re-run it with the same prompt and it would generate a different answer.

**Hints:** Often had to update hint files after catching the model's mistakes. The updates were in two forms: fixes where the hint files were incorrect, and trying to emphasize taking one approach or the other. Most models were good at following the hints, despite how big the hint files have gotten. That said, there are some things (like `ak.stack`) that seem to be totally baked into the model's training and nothing I can do will dissuade them. Might be one place where fine tuning our own models would be an advantage.

**General Comments & Future Directions:**

* Writing some of this code to manipulate arrays is subtle! And some of it is not obvious why you need to do it, even after you've written it (at least to me).
* Complex queries will need a planning step.
* The approach taken for ServiceX will affect things downstream. So probably best to split the task in two, tackle the ServiceX step, and then go back and revisit the downstream method.
* Some of the awkward code will need to be run in order to sort out what went wrong.
* Running on many files will require a different strategy than running on one file. But it might be worth getting the 1 file run working and then translating to a many file run. Much like how we as humans do it.
