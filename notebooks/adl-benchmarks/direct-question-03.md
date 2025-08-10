# Plot the pT of jets with |η| < 1 in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.



## Model: gpt-4.1

- Accessed the specified rucio PHYSLITE dataset via ServiceX.
- Queried only jet $p_T$ and $\eta$ to minimize data movement.
- Filtered jets at the query level to select only those with $|\eta| < 1$.
- Used Awkward Array to extract the resulting jet $p_T$ values (in GeV).
- Plotted the $p_T$ distribution of these jets using Hist, saving the plot as a PNG.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

# Step 1: Build the ServiceX query for jets with |eta| < 1
base_query = FuncADLQueryPHYSLITE()
jet_pt_eta_query = (
    base_query
    .SelectMany(lambda evt: evt.Jets())
    .Where(lambda jet: abs(jet.eta()) < 1.0)
    .Select(lambda jet: {"pt": jet.pt() / 1000.0})  # Convert pt to GeV
)

# Step 2: Run the query on one file of the dataset for testing.
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jets",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Use a single file for quick test; increase as needed
                    Query=jet_pt_eta_query,
                )
            ]
        ),
    )
)
# Step 3: The resulting awkward array
jet_pts = result["jets"].pt  # flat array of pT [GeV]

# Step 4: Plot the histogram with Hist and mplhep
plt.style.use(hep.style.ATLAS)

hist_bins = Hist.new.Reg(50, 0, 500, name="pt", label="Jet $p_T$ [GeV]").Double()
hist_bins.fill(pt=jet_pts)

hist_bins.plot(histtype="fill", linewidth=1, edgecolor="gray", label="Jets $|\eta|<1$")
plt.xlabel("Jet $p_T$ [GeV]")
plt.ylabel("Jets / bin")
plt.legend()
plt.title("Jet $p_T$ Distribution ($|\eta|<1$)")
plt.savefig("jet_pt_eta_lt1.png")
plt.close()
```


## Model: gpt-5

- Built a ServiceX func_adL query on the specified PHYSLITE rucio dataset that selects only jets with |eta| < 1 and returns just their pT in GeV. The eta cut is applied inside ServiceX to minimize data movement.
- Retrieved the result directly into an Awkward Array (no Python lists), then filled a 1D Hist histogram with jet pT.
- Plotted and saved the histogram to a PNG file: jet_pt_eta_lt1.png.
- For quick testing, the code requests only 1 file (NFiles=1). Remove the NFiles argument to run on the full dataset.

```python
import awkward as ak
import numpy as np
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

from hist import Hist
import matplotlib.pyplot as plt

# 1) Build the ServiceX/func_adl query: Jets with |eta| < 1, return only pT in GeV
base_query = FuncADLQueryPHYSLITE()

jet_pt_query = (
    base_query
    .SelectMany(lambda e: e.Jets())
    .Where(lambda j: (j.eta() > -1.0) & (j.eta() < 1.0))
    .Select(lambda j: {"jet_pt": j.pt() / 1000.0})  # GeV
)

# 2) Define the dataset (PHYSLITE MC23 sample)
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# 3) Deliver to Awkward Array (start with NFiles=1 for a quick test; remove NFiles to run the full dataset)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jets_eta_lt1",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # remove this line to run over the full dataset
                    Query=jet_pt_query,
                )
            ]
        ),
    )
)

# 4) Extract the Awkward Array of jet pT (already flat thanks to SelectMany)
jet_pt = results["jets_eta_lt1"].jet_pt  # ak.Array of floats (GeV)

# 5) Make and fill a histogram (Hist)
h = (
    Hist.new
    .Reg(100, 0.0, 500.0, name="pt", label="Jet pT [GeV]")
    .Int64()
)
# Fill using numpy conversion (no Python lists)
h.fill(pt=ak.to_numpy(jet_pt))

# 6) Plot and save
fig, ax = plt.subplots(figsize=(7, 5))
h.plot(ax=ax, histtype="fill", alpha=0.7, color="tab:blue", edgecolor="black", label="|η| < 1")
ax.set_xlabel("Jet pT [GeV]")
ax.set_ylabel("Entries")
ax.set_title("Jet pT for |η| < 1")
ax.legend()
plt.tight_layout()
plt.savefig("jet_pt_eta_lt1.png", dpi=150)
plt.close(fig)
```


## Model: gpt-5-mini

- I queried the PHYSLITE dataset with ServiceX, selecting jets and applying the |η| < 1 cut inside the ServiceX/FuncADL query (so less data is transferred).
- I converted jet pt from MeV to GeV in the query (dividing by 1000).
- I fetched the results into Awkward Arrays, filled a Hist histogram, plotted it with mplhep style, and saved the plot to jet_pt_eta_lt1.png.
- For testing I set NFiles=1 in the ServiceX Sample. To run over the full dataset, remove the NFiles argument (or increase it).

```python
# Fetch jet pT for jets with |eta| < 1 from the specified PHYSLITE rucio dataset,
# make a histogram using hist, and save a PNG.

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Build the FuncADL query (filter jets by |eta| < 1 inside ServiceX)
base_query = FuncADLQueryPHYSLITE()

# Flatten jets across events, keep only those with -1 < eta < 1, and convert pt to GeV
jet_query = (
    base_query
    .SelectMany(lambda evt: evt.Jets())
    .Where(lambda j: (j.eta() < 1.0) & (j.eta() > -1.0))
    .Select(lambda j: {
        "jet_pt": j.pt() / 1000.0,   # MeV -> GeV
        "jet_eta": j.eta()
    })
)

# 2) Define the dataset (PHYSLITE) and ask ServiceX to deliver results.
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# NOTE: NFiles=1 for quick testing. Remove NFiles to run on the full dataset.
spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="jets_eta_lt1",
            Dataset=dataset.Rucio(ds_name),
            NFiles=1,
            Query=jet_query,
        )
    ]
)

# Deliver and convert to awkward arrays
result = to_awk(deliver(spec))

# Access the returned awkward array of jets
jets = result["jets_eta_lt1"].jet_pt  # this should be a flat awkward array of jet pT values (GeV)

# Ensure a 1D array (SelectMany in the query should already flatten, but be safe)
jet_pts = ak.flatten(jets)

# Optionally report some basic info (kept as awkward operations)
n_jets = ak.num(jet_pts, axis=0) if ak.num(jet_pts, axis=0).size == 1 else ak.count(jet_pts)
# ak.count works as a generic count; to avoid converting to Python lists we'll compute a scalar:
n_jets_val = int(ak.sum(ak.ones_like(jet_pts))) if ak.to_list(jet_pts) else 0

print(f"Number of jets retrieved (|eta|<1): {len(jet_pts)}")

# 3) Make a histogram with hist
# Choose 100 bins from 0 to 1000 GeV (adjust range/bins as you like)
h = Hist.new.Reg(100, 0.0, 1000.0, name="pt", label="Jet $p_T$ [GeV]").Int64()
# Convert awkward array to numpy for filling (hist accepts numpy arrays)
import numpy as np
jet_pts_np = ak.to_numpy(jet_pts)  # 1D numpy array

# Fill histogram
h.fill(pt=jet_pts_np)

# 4) Plot and save as PNG using mplhep ATLAS style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8, 6))
h.plot(histtype="step", ax=ax, label=r"$| \eta | < 1$")
ax.set_xlabel("Jet $p_T$ [GeV]")
ax.set_ylabel("Counts")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.title("Jet $p_T$ for |η| < 1")

outname = "jet_pt_eta_lt1.png"
plt.savefig(outname, dpi=150)
print(f"Saved histogram to {outname}")
```


## Model: gpt-5-nano

- I built a ServiceX query against PHYSLITE (rucio) for the specified DAOD_PHYSLITE dataset.
- I filter jets to |η| < 1.0 at the ServiceX level to minimize data transfer.
- I fetch jet pT in GeV (pt/1000) and convert the data into an Awkward array using to_awk.
- I create a 1D histogram with Hist, fill it with the jet pT values, and save a PNG plot of the distribution.

```python
# Required imports
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
import awkward as ak
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Define the base query (PHYSLITE)
base_query = FuncADLQueryPHYSLITE()

# 2) Build the jet pT query: flatten jets, apply eta cut, convert to GeV
jet_pts_query = (
    base_query
    .SelectMany(lambda e: e.Jets())
    .Where(lambda j: abs(j.eta()) < 1.0)
    .Select(lambda j: {"pt": j.pt() / 1000.0})  # GeV
)

# 3) Dataset specification (provided dataset)
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# 4) Deliver and convert to Awkward array (NFiles=1 for testing)
all_jet_pts = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_pt_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=jet_pts_query,
                )
            ]
        ),
    )
)

# 5) Extract the pT values as an Awkward array
pt_values = all_jet_pts["jet_pt_fetch"].pt  # Awkward array of jet pT in GeV

# 6) Create a 1D histogram for jet pT
h = (
    Hist.new
        .Reg(100, 0, 400, name="pt", label="Jet pT [GeV]")
        .Int64()  # store counts as 64-bit integers
)

# 7) Fill the histogram (Awkward array input)
h.fill(pt=pt_values)

# 8) Plot and save to PNG
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(6, 4.5))
h.plot(histtype="step", ax=ax, label="Jets with |η|<1.0")
ax.set_xlabel("Jet pT [GeV]")
ax.set_ylabel("Entries")
ax.legend()
plt.tight_layout()
plt.savefig("jets_pt_eta_lt1.png")
plt.close(fig)
```


## Model: gpt-4o

- Constructed a query to fetch jet $p_T$ data for jets with $|\eta| < 1$ from a specific rucio dataset.
- Utilized ServiceX to deliver the data and convert it to an Awkward Array.
- Created a histogram to visualize the distribution of jet $p_T$.
- Saved the plot to a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Define the dataset name
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697")

# The base query should run against PHYSLITE.
base_query = FuncADLQueryPHYSLITE()

# Query: get jet pT for jets with |eta| < 1
jet_pts_query = (base_query
    .SelectMany(lambda evt: evt.Jets())
    .Where(lambda jet: abs(jet.eta()) < 1.0)
    .Select(lambda jet: {
        "jet_pt": jet.pt() / 1000.0,
    })
)

# Do the fetch
all_jet_pts = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_pt_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=jet_pts_query,
                )
            ]
        ),
    )
)

# Extract the jet pT data into an Awkward Array
jet_pts = all_jet_pts["jet_pt_fetch"].jet_pt

# Create the histogram
h = (
    Hist.new
    .Reg(50, 0, 500, name="jet_pt", label="Jet $p_T$ [GeV]")
    .Double()
)

# Fill the histogram with the jet pT data
h.fill(jet_pt=ak.flatten(jet_pts))

# Plotting
plt.style.use(hep.style.ATLAS)  # Use ATLAS style for plots
h.plot(histtype='step', linewidth=2)
plt.xlabel("Jet $p_T$ [GeV]")
plt.ylabel("Number of Jets")
plt.title("Jet $p_T$ Distribution with $|\eta| < 1$")
plt.savefig("jet_pt_distribution.png")
plt.show()
```


## Model: o4-mini

- Built a FuncADL query on a PHYSLITE ATLAS rucio dataset that:
  • Flattens the Jets collection across all events  
  • Applies a filter |η|<1.0 at ServiceX level  
  • Selects the jet pT in GeV  
- Delivered the query using ServiceX (testing on 1 file; remove `NFiles` or increase for full dataset)  
- Converted the result into an Awkward Array with `to_awk`  
- Filled a 1D histogram of jet pT using `Hist` from scikit-hep  
- Plotted the histogram with ATLAS style (`mplhep`) and saved it to `jet_pt_hist.png`  
```python
# Imports for ServiceX + FuncADL
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Imports for histogramming and plotting
import awkward as ak
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Define the rucio dataset name
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2"
    ".deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# 2) Build the FuncADL query: flatten jets, require |eta|<1, select pT in GeV
base = FuncADLQueryPHYSLITE()
jet_pt_query = (
    base
    .SelectMany(lambda evt: evt.Jets())
    .Where(lambda j: abs(j.eta()) < 1.0)
    .Select(lambda j: {"pt": j.pt() / 1000.0})
)

# 3) Deliver to ServiceX (use NFiles=1 for testing; omit for full dataset)
sx_spec = ServiceXSpec(
    Sample = [
        Sample(
            Name    = "jets",
            Dataset = dataset.Rucio(ds_name),
            NFiles  = 1,
            Query   = jet_pt_query,
        )
    ]
)
awk_data = to_awk(deliver(sx_spec))

# 4) Extract the pT array
jet_pts = awk_data["jets"].pt  # awkward array of floats (GeV)

# 5) Create and fill a histogram of jet pT
h = (
    Hist.new.Reg(50, 0, 500, name="pt", label="Jet $p_{T}$ [GeV]")
       .Int64()
)
h.fill(pt=jet_pts)

# 6) Plot with ATLAS style and save to file
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(ax=ax, histtype="stepfilled", edgecolor="black", label="Jets $|η|<1$")
ax.legend()
ax.set_xlabel("Jet $p_{T}$ [GeV]")
ax.set_ylabel("Entries")
fig.savefig("jet_pt_hist.png")
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 6.47 | 11646 | 612 | 12258 | 0.0282 |
| gpt-5 | 51.13 | 11645 | 2166 | 13811 | 0.0362 |
| gpt-5-mini | 29.42 | 11645 | 2763 | 14408 | 0.0084 |
| gpt-5-nano | 15.44 | 11645 | 3706 | 15351 | 0.0021 |
| gpt-4o | 8.09 | 11646 | 584 | 12230 | 0.0350 |
| o4-mini | 12.37 | 11645 | 1414 | 13059 | 0.0190 |
