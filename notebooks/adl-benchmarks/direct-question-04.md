# Plot the ETmiss of events that have at least two jets with pT > 40 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.



## Model: gpt-4.1

- Selected the rucio dataset: `mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697`
- Built a ServiceX query to select only those events with at least 2 jets with $p_T > 40$ GeV.
- For these selected events, fetched the ETmiss (missing transverse energy) value in GeV.
- Used Awkward Array to receive and represent the results.
- Used Hist to histogram the ETmiss distribution, and made a plot.
- Saved the resulting plot to `etmiss.png`.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1. Prepare the ServiceX query
query = (
    FuncADLQueryPHYSLITE()
    .Where(
        lambda e: e.Jets()
                   .Where(lambda j: j.pt()/1000.0 > 40.0)
                   .Count() >= 2
    )
    .Select(lambda e: {"etmiss": e.MissingET().First().met()/1000.0})  # GeV
)

# 2. Dataset name
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# 3. Fetch the data (for quick testing, NFiles=1; remove this for full scale analysis)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="etmiss_selection",
                    Dataset=dataset.Rucio(ds_name),
                    Query=query,
                    NFiles=1,  # Remove or increase for full dataset
                )
            ]
        )
    )
)

# 4. Extract the ETmiss awkward array
etmiss = results["etmiss_selection"].etmiss

# 5. Build the histogram
etmiss_hist = (
    Hist.new.Reg(40, 0, 400, name="etmiss", label="E$_T^{miss}$ [GeV]")
        .Int64()
)

etmiss_hist.fill(etmiss=ak.to_numpy(etmiss))

# 6. Plot the histogram
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
etmiss_hist.plot(histtype="step", linewidth=1.5, ax=ax, label="$E_T^{miss}$")
ax.set_xlabel("$E_T^{miss}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("ETmiss in events with $\geq$2 jets ($p_T>40$ GeV)")
ax.legend()
plt.tight_layout()
plt.savefig("etmiss.png")
plt.close()
```


## Model: gpt-5

- Used ServiceX with a func-adl Release 25 PHYSLITE query to fetch only events that have at least two jets with pT > 40 GeV (applied at ServiceX to minimize data volume).
- Selected the event MissingET and converted to GeV directly in the query.
- Converted the delivery to Awkward Array, then used Hist to make a 1D histogram of ETmiss and plotted with mplhep ATLAS style.
- Saved the plot to PNG: etmiss_2jets40_mc23_JZ2.png.
- For quick testing the code runs on NFiles=1; remove the NFiles parameter to run on the full dataset.

```python
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# ---------------------------------------
# 1. Define the dataset and the query
# ---------------------------------------
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
    "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Base query against PHYSLITE
base = FuncADLQueryPHYSLITE()

# Event-level selection: require >= 2 jets with pT > 40 GeV (GeV units in the query)
# Then select MissingET (first element) and convert to GeV
query = (
    base
    .Where(lambda e:
           e.Jets()
            .Where(lambda j: j.pt() / 1000.0 > 40.0)
            .Count() >= 2)
    .Select(lambda e: {
        "met": e.MissingET().First().met() / 1000.0,  # ETmiss in GeV
    })
)

# ---------------------------------------
# 2. Deliver with ServiceX (start with 1 file for testing)
#    Remove NFiles to run on the full dataset.
# ---------------------------------------
sx_results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="etmiss_2jets40",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Remove this line to run over the full dataset
                    Query=query,
                )
            ]
        )
    )
)

# ---------------------------------------
# 3. Extract Awkward arrays
# ---------------------------------------
etmiss = sx_results["etmiss_2jets40"].met  # 1D awkward array of ETmiss [GeV]
# Ensure a flat numeric array for histogram fill
etmiss_np = ak.to_numpy(ak.flatten(etmiss, axis=None))

# ---------------------------------------
# 4. Make histogram with Hist
# ---------------------------------------
h = (
    Hist.new
    .Reg(50, 0.0, 500.0, name="met", label=r"$E_T^{\mathrm{miss}}$ [GeV]")
    .Int64()
)
h.fill(met=etmiss_np)

# ---------------------------------------
# 5. Plot and save to PNG
# ---------------------------------------
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(7, 5))
h.plot(ax=ax, histtype="step", linewidth=1.5, label=r"$\geq 2$ jets with $p_T > 40$ GeV")
ax.set_ylabel("Events")
ax.legend()
ax.set_title("mc23 13.6 TeV JZ2 (PHYSLITE)")
plt.tight_layout()
plt.savefig("etmiss_2jets40_mc23_JZ2.png", dpi=150)
plt.close(fig)
```


## Model: gpt-5-mini

- I filtered events at the ServiceX/func_adl level to keep only events with at least two jets with pT > 40 GeV (this reduces how much data is moved out of ServiceX).
- I fetched the event MissingET (MET) from the PHYSLITE xAOD dataset you specified, converting MET to GeV.
- I filled a 1D histogram of MET using hist and plotted it with an ATLAS-like style, saving the plot to met_hist.png.
- Note: the code uses NFiles=1 for a fast test run. To run over the full dataset remove the NFiles argument (or set it to a larger number) — that will process the entire Rucio dataset but take longer.

```python
# Fetch MET for events with >= 2 jets (pT > 40 GeV), histogram and save plot.
# Requires: func_adl_servicex_xaodr25, servicex, servicex_analysis_utils, awkward, hist, mplhep, matplotlib, numpy

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk

import awkward as ak
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Build the func_adl query (PHYSLITE)
base = FuncADLQueryPHYSLITE()

# Select events that have at least two jets with pT > 40 GeV, then get the event MissingET value
query = (
    base
    # jets pT in func_adl/xAOD are in MeV, so convert to GeV when comparing
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 > 40.0).Count() >= 2)
    # MissingET is a sequence; take the First() object and then its met()
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {"met": m.met() / 1000.0})  # convert MeV -> GeV
)

# Rucio dataset asked for
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Deliver via ServiceX. Use NFiles=1 for a quick test run; remove NFiles to run over the full dataset.
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="met_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # start with 1 file for testing; remove or increase to run full dataset
                    Query=query,
                )
            ]
        )
    )
)

# 'results' is a dict keyed by Sample.Name
met_awk = results["met_fetch"].met  # awkward array of MET values in GeV (one per selected event)

# Basic info (print number of events returned)
print("Number of events returned by ServiceX (NFiles=1 test):", len(met_awk))

# Build a histogram: 50 bins from 0 to 200 GeV
h = Hist.new.Reg(50, 0.0, 200.0, name="met", label=r"$E_{T}^{miss}$ [GeV]").Int64()

# Fill histogram using the Awkward array (flatten in case of nesting)
met_vals = ak.flatten(met_awk)       # remains an Awkward Array
# Convert to NumPy for hist.fill (this does not use Python lists for the physics manipulations)
met_numpy = ak.to_numpy(met_vals)

h.fill(met=met_numpy)

# Plot using mplhep ATLAS style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8, 6))
# hist.Hist objects have a .plot method that works with matplotlib axes
h.plot(ax=ax, histtype="stepfilled", label="Events with ≥2 jets (pT > 40 GeV)", edgecolor="black", alpha=0.6)

ax.set_xlabel(r"$E_{T}^{miss}$ [GeV]")
ax.set_ylabel("Events")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)

# Save the plot to a PNG file
outname = "met_hist.png"
plt.tight_layout()
plt.savefig(outname, dpi=150)
print(f"Saved MET histogram to {outname}")
```


## Model: gpt-5-nano

- I implemented a ServiceX-based workflow that targets the PHYSLITE DAOD dataset you specified, filtering events at the ServiceX level to only keep those with at least two jets having pT > 40 GeV.
- I fetch the event MissingET (ETmiss) values in GeV (converting from MeV as needed) and expose them as an Awkward array for downstream processing.
- I use Awkward Array for data manipulation and Hist to create a 1D histogram of ETmiss, then plot and save the figure as a PNG file.
- The code uses NFiles=1 for testing as recommended, and explains the pipeline inline via comments.

```python
# Imports for the ServiceX/FuncADL flow and plotting
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
import awkward as ak

# Typing/plotting imports
import hist
import matplotlib.pyplot as plt
import mplhep as hep

# Dataset name (as provided)
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE."
           "e8514_e8528_a911_s4114_r15224_r15225_p6697")

# Base query against PHYSLITE
base_query = FuncADLQueryPHYSLITE()

# MET query: filter events with at least 2 jets with pt > 40 GeV (pt is in MeV)
# and select MissingET().First().met() converted to GeV
met_query = (
    base_query
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 > 40.0).Count() >= 2)
    .Select(lambda e: {"met": e.MissingET().First().met() / 1000.0})
)

# Fetch data via ServiceX (test with a single file)
met_fetch = deliver(
    ServiceXSpec(
        Sample=[
            Sample(
                Name="etmiss_fetch",
                Dataset=dataset.Rucio(ds_name),
                NFiles=1,
                Query=met_query,
            )
        ]
    )
)

# Convert to Awkward Array (only Awkward operations thereafter)
et_values = to_awk(met_fetch)
met_vals = et_values["etmiss_fetch"].met

# Build a histogram for ETmiss in GeV
h = (
    hist.Hist.new
    .Reg(50, 0, 500, name="met", label="Missing ET [GeV]")  # 50 bins from 0 to 500 GeV
    .Weight()  # store integer counts
)

# Fill the histogram with the ETmiss values
h.fill(met=met_vals)

# Plotting: ATLAS style
plt.style.use(hep.style.ATLAS)
plt.figure(figsize=(8, 6))
# Simple step-style plot of the histogram
h.plot(histtype="step", linewidth=1, label="ETmiss, events with ≥2 jets pT>40 GeV")
plt.xlabel("Missing ET [GeV]")
plt.ylabel("Entries")
plt.legend()
plt.tight_layout()

# Save to PNG
plt.savefig("etmiss_two_jets_gt40GeV.png")
```


## Model: gpt-4o

- Constructed a query using `FuncADLQueryPHYSLITE` to select events with at least two jets having transverse momentum (pT) greater than 40 GeV.
- Extracted the missing transverse energy (ETmiss) for these events using the `MissingET` collection.
- Used ServiceX to fetch the filtered data from the specified `rucio` dataset.
- Processed the data with Awkward arrays to ensure efficient handling of high-dimensional data.
- Plotted the ETmiss values using `Hist` and saved the plot as a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import matplotlib.pyplot as plt
from hist import Hist
import mplhep as hep

# The base query should run against PHYSLITE.
base_query = FuncADLQueryPHYSLITE()

# Query: Select only events with at least 2 jets with pt > 40 GeV and get ETmiss
etmiss_query = (
    base_query
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 > 40.0).Count() >= 2)
    .Select(lambda e: {'etmiss': e.MissingET().First().met() / 1000.0})
)

# Define the dataset
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Fetch data
etmiss_data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="etmiss_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1
                )
            ],
            Query=etmiss_query
        )
    )
)
# Extract ETmiss array
etmiss_array = etmiss_data["etmiss_fetch"].etmiss

# Plot ETmiss
etmiss_hist = Hist.new.Reg(50, 0, 500, name="etmiss", label="ETmiss [GeV]").Double()
etmiss_hist.fill(etmiss=etmiss_array)

plt.style.use(hep.style.ATLAS)
etmiss_hist.plot(histtype='fill', linewidth=2, label='ETmiss')
plt.xlabel('ETmiss [GeV]')
plt.ylabel('Events')
plt.legend()
plt.savefig("etmiss_histogram.png")
plt.show()
```


## Model: o4-mini

- Defined a FuncADL query on the PHYSLITE sample that:
  • Filters events to require at least two jets with pT > 40 GeV  
  • Extracts the event missing transverse energy (MET) in GeV  
- Delivered the query via ServiceX (using 1 file for quick testing) and converted the result to an Awkward array  
- Built a 1D histogram of MET with 100 bins from 0 to 500 GeV using `hist`  
- Plotted the MET distribution with ATLAS style and saved it to `etmiss.png`  

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Build the FuncADL query
base = FuncADLQueryPHYSLITE()
met_query = (
    base
    # Require ≥2 jets with pT > 40 GeV
    .Where(lambda e:
        e.Jets()
         .Where(lambda j: j.pt() / 1000.0 > 40.0)
         .Count() >= 2
    )
    # Grab the MissingET object and convert to a flat field
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {"met": m.met() / 1000.0})
)

# 2) Deliver via ServiceX (use NFiles=1 for testing)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV."
    "801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_"
    "r15224_r15225_p6697"
)
samples = [
    Sample(
        Name="met_fetch",
        Dataset=dataset.Rucio(ds_name),
        NFiles=1,
        Query=met_query,
    )
]
result = to_awk(deliver(ServiceXSpec(Sample=samples)))

# 3) Extract the MET array
met_array = result["met_fetch"].met  # Awkward array of MET values in GeV

# 4) Create and fill a histogram
h = (
    hist.Hist.new
        .Reg(100, 0, 500, name="met", label="Missing ET [GeV]")
        .Int64()  # integer counts
)
h.fill(met=met_array)

# 5) Plot with ATLAS style and save
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(
    ax=ax,
    histtype="step",
    label="Events with ≥2 jets (pT > 40 GeV)",
    color="black",
)
ax.set_xlabel("Missing ET [GeV]")
ax.set_ylabel("Events / bin")
ax.legend()
fig.savefig("etmiss.png")
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 8.14 | 11653 | 705 | 12358 | 0.0289 |
| gpt-5 | 62.46 | 11652 | 2958 | 14610 | 0.0441 |
| gpt-5-mini | 24.77 | 11652 | 2562 | 14214 | 0.0080 |
| gpt-5-nano | 17.54 | 11652 | 4782 | 16434 | 0.0025 |
| gpt-4o | 8.76 | 11653 | 580 | 12233 | 0.0349 |
| o4-mini | 12.62 | 11652 | 1397 | 13049 | 0.0190 |
