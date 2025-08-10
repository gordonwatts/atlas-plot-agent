# Plot the pT of all jets in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.



## Model: gpt-4.1

- I used ServiceX and func_adl to fetch all jet $p_T$ values from the specified ATLAS PHYSLITE Rucio dataset.
- The code uses only one file for demonstration/testing speed. Remove or increase `NFiles` for larger statistics.
- I converted the raw jet $p_T$ units from MeV to GeV as recommended.
- The Awkward Array is used to handle and process the jet $p_T$ data.
- I filled a 1D histogram (using hist) with these $p_T$ values, using reasonable binning for jets.
- The plot is made and saved to a PNG file named `jet_pt.png`.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# Use ATLAS plot style
plt.style.use(hep.style.ATLAS)

# Dataset name as requested
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
    "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Construct the query for all jet pt (in GeV)
base_query = FuncADLQueryPHYSLITE()
jet_pts_query = (
    base_query
    .SelectMany(lambda evt: evt.Jets())
    .Select(lambda jet: {
        "jet_pt": jet.pt() / 1000.0,  # MeV to GeV
    })
)

# Fetch data (one file for speed/test)
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

# Get jet pt as awkward array
jet_pt = all_jet_pts["jet_pt_fetch"].jet_pt

# Plotting: Make a histogram. Binning from 0 to 500 GeV (50 bins)
h = Hist.new.Reg(50, 0, 500, name="pt", label=r"Jet $p_T$ [GeV]").Int64()
h.fill(pt=ak.to_numpy(jet_pt))

# Draw the plot
fig, ax = plt.subplots()
h.plot(ax=ax, histtype="step", label="all jets")
ax.set_xlabel(r"Jet $p_T$ [GeV]")
ax.set_ylabel("Jets")
ax.set_title("Jet $p_T$ Spectrum")
ax.legend()
plt.savefig("jet_pt.png")
plt.close(fig)
```


## Model: gpt-5

- Built a ServiceX FuncADL query on DAOD_PHYSLITE to select only jet pT (converted to GeV) to minimize data moved.
- Fetched data from the specified Rucio dataset and converted to an Awkward Array (no Python lists used).
- Filled a histogram with jet pT using Hist and saved the plot to a PNG file.
- Note: The code uses NFiles=1 for a quick test. Remove NFiles to run over the full dataset.

```python
#!/usr/bin/env python3

# Fetch jet pT from an ATLAS PHYSLITE dataset via ServiceX, histogram with Hist, and save a plot.

from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

def main():
    # Use an ATLAS-like plotting style
    plt.style.use(hep.style.ATLAS)

    # Rucio dataset name provided by the user
    ds_name = (
        "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
        "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
    )

    # Construct a minimal query: select only jet pt (in GeV)
    base_query = FuncADLQueryPHYSLITE()
    jet_pts_query = (
        base_query
        .SelectMany(lambda evt: evt.Jets())
        .Select(lambda jet: {"jet_pt": jet.pt() / 1000.0})  # MeV -> GeV
    )

    # Deliver via ServiceX
    # Start with NFiles=1 for a quick test. Remove NFiles to run the full dataset.
    sx_result = deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_pt_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Remove this line to process the full dataset
                    Query=jet_pts_query,
                )
            ]
        )
    )

    # Convert the ServiceX output to Awkward
    data = to_awk(sx_result)
    jet_pts = data["jet_pt_fetch"].jet_pt  # flat awkward array of jet pT [GeV]

    # Define a 1D histogram for jet pT
    # Choose a broad pT range; overflow is captured in overflow bins internally.
    h_pt = (
        Hist.new
        .Reg(200, 0.0, 1000.0, name="pt", label="Jet pT [GeV]")
        .Int64()
    )

    # Fill histogram using Awkward -> NumPy view
    h_pt.fill(pt=ak.to_numpy(jet_pts))

    # Plot and save
    fig, ax = plt.subplots(figsize=(8, 6))
    h_pt.plot(ax=ax, histtype="step", linewidth=1.5, color="C0", label="All jets")
    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("Entries")
    ax.legend()
    ax.set_title("Jet pT distribution (DAOD_PHYSLITE)")

    plt.tight_layout()
    plt.savefig("jet_pt.png", dpi=150)
    plt.close(fig)

    print("Saved plot to jet_pt.png")

if __name__ == "__main__":
    main()
```


## Model: gpt-5-mini

- Queried the PHYSLITE dataset on Rucio with ServiceX to retrieve every jet's pT (converted to GeV).
- Used NFiles=1 for a quick test pull from the dataset (see note in code — remove NFiles to run on the full dataset).
- Loaded the returned data into an Awkward Array, built a 1D histogram with hist (100 bins, range chosen from the data), plotted with mplhep/ATLAS style, and saved the plot to jet_pt.png.

```python
# Fetch jet pT from the specified PHYSLITE dataset with ServiceX, fill a histogram, and save to PNG.
# Note: This script uses NFiles=1 to run quickly for testing. Remove the NFiles argument (or set it to None)
# to run over the full dataset (this will be much slower and fetch more data).

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Build the FuncADL query: flatten jets and return pt in GeV
base_query = FuncADLQueryPHYSLITE()
jet_pt_query = (
    base_query
    .SelectMany(lambda ev: ev.Jets())          # flatten jets across events
    .Select(lambda j: {"pt": j.pt() / 1000.0}) # convert MeV -> GeV
)

# 2) Define the dataset (PHYSLITE on rucio)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# 3) Deliver the query via ServiceX. Use NFiles=1 for a quick test run.
#    To run on the full dataset, remove NFiles or set it to None.
spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="jets_fetch",
            Dataset=dataset.Rucio(ds_name),
            NFiles=1,
            Query=jet_pt_query,
        )
    ]
)

print("Submitting request to ServiceX (NFiles=1 for a quick test)...")
fetched = to_awk(deliver(spec))
print("ServiceX delivery complete.")

# 4) Extract the awkward array of jet pts
# The returned structure is a dict keyed by Sample.Name ("jets_fetch")
jets_result = fetched["jets_fetch"]
# We used the key 'pt' in the Select above
jet_pts = jets_result.pt  # This is an Awkward Array of floats (GeV)

# Basic info:
n_jets = int(ak.num(jet_pts)) if ak.ndim(jet_pts) == 0 else int(len(jet_pts))
print(f"Number of jet entries fetched: {n_jets}")

# 5) Build a histogram using hist. Choose binning from data:
#    use up to max(200, 1.1*max_pt) to get a sensible range for typical jets.
if len(jet_pts) == 0:
    raise RuntimeError("No jets were returned from the query.")
max_pt = float(ak.max(jet_pts))
upper = max(200.0, max_pt * 1.1)

h = (
    Hist.new
    .Reg(100, 0.0, upper, name="pt", label="Jet $p_T$ [GeV]")
    .Int64()
)

# Fill the histogram. Convert the Awkward 1D array to a NumPy array for filling.
# (We keep all internal manipulation with Awkward; conversion here is just for hist.fill)
h.fill(pt=ak.to_numpy(jet_pts))

# 6) Plot with mplhep ATLAS style and save as PNG
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8,6))
h.plot(ax=ax, histtype="stepfilled", edgecolor="black", label="Jets (from PHYSLITE sample)")
ax.set_xlabel("Jet $p_T$ [GeV]")
ax.set_ylabel("Counts")
ax.set_xlim(0, upper)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)

outname = "jet_pt.png"
plt.tight_layout()
plt.savefig(outname, dpi=150)
print(f"Saved jet pT plot to: {outname}")
```


## Model: gpt-5-nano

- Set up a ServiceX query against PHYSLITE to fetch all jets and their transverse momenta (pt), converting from MeV to GeV.
- Use a single-file test run (NFiles=1) against the provided rucio dataset to keep fetches small.
- Convert the ServiceX results to an Awkward Array with to_awk and extract the jet pt values in GeV.
- Create a 1D histogram of jet pT using Hist, then plot with matplotlib (ATLAS style) and save as a PNG.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk

# 1) Define the base query (PHYSLITE) and the jet pT extraction (in GeV)
base_query = FuncADLQueryPHYSLITE()

# Get all jets and fetch their pt in GeV
jet_pts_query = (
    base_query
    .SelectMany(lambda evt: evt.Jets())
    .Select(lambda jet: {"pt_GeV": jet.pt() / 1000.0})  # MeV -> GeV
)

# 2) Fetch the data from the provided dataset (rucio)
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# For testing, start with a single file
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

# 3) Extract the jet pT values (GeV) as an Awkward Array
pt_values = all_jet_pts["jet_pt_fetch"].pt_GeV

# 4) Build a histogram of jet pT ( GeV )
hist_jet_pt = Hist.new.Reg(120, 0, 600, name="pt_GeV", label="Jet pT [GeV]")

# Fill histogram with Awkward array directly
hist_jet_pt.fill(pt_GeV=pt_values)

# 5) Plot and save to PNG
plt.style.use(hep.style.ATLAS)

fig, ax = plt.subplots(figsize=(8, 6))
hist_jet_pt.plot(ax=ax, histtype="step", fill=False, label="Jets")

ax.set_xlabel("Jet pT [GeV]")
ax.set_ylabel("Entries")
ax.legend()

plt.tight_layout()
plt.savefig("jet_pt_hist.png")
plt.close()
```


## Model: gpt-4o

- Utilized ServiceX to fetch jet transverse momentum (\( p_T \)) data from the specified Rucio dataset.
- Converted the jet \( p_T \) values from MeV to GeV within the query.
- Used Awkward Array to process the data into a suitable format for plotting.
- Created and filled a histogram of jet \( p_T \) using the Hist library.
- Plotted the histogram and saved it as a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import hist
import matplotlib.pyplot as plt
import mplhep as hep

# The base query should run against PHYSLITE.
base_query = FuncADLQueryPHYSLITE()

# Query: get all jet pT
jet_pts_query = (base_query
    .SelectMany(lambda evt: evt.Jets())
    .Select(lambda jet: {
        "jet_pt": jet.pt() / 1000.0,  # Convert from MeV to GeV
    })
)

# Dataset name
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Fetch the data
all_jet_pts = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_pt_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1  # Start with a single file for testing
                )
            ],
            Query=jet_pts_query
        ),
    )
)

# Access the awkward array for jet pt
jet_pts = all_jet_pts["jet_pt_fetch"].jet_pt

# Create a histogram for pT
h = (
    hist.Hist.new
    .Reg(100, 0, 1000, name="jet_pt", label="Jet $p_T$ [GeV]")
    .Double()
)

# Fill the histogram
h.fill(jet_pt=jet_pts.to_numpy())

# Plot the histogram
plt.style.use(hep.style.ATLAS)
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="Jets")
plt.xlabel("Jet $p_T$ [GeV]")
plt.ylabel("Entries")
plt.legend()
plt.title("Histogram of Jet $p_T$")
plt.savefig("jet_pt_histogram.png")
plt.show()
```


## Model: o4-mini

- Constructed a FuncADLQueryPHYSLITE query to flatten all jets and select their transverse momentum (converted to GeV).  
- Used ServiceX to deliver one file from the specified Rucio dataset (for a quick test).  
- Converted the delivered data into an Awkward Array via `to_awk`.  
- Built a 1D histogram of the jet $p_T$ using `hist.Hist.new.Reg`.  
- Plotted the distribution with ATLAS styling (via `mplhep`) and saved it to `jet_pt.png`.  

```python
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# 1) Build the FuncADL query: flatten jets and extract pT in GeV
base_query = FuncADLQueryPHYSLITE()
jet_pt_query = (
    base_query
    .SelectMany(lambda evt: evt.Jets())
    .Select(lambda jet: {"jet_pt": jet.pt() / 1000.0})
)

# 2) Define the Rucio dataset and fetch (NFiles=1 for testing)
ds_name = (
    "mc23_13p6TeV:"
    "mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_pt_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=jet_pt_query,
                )
            ]
        )
    )
)

# 3) Extract the Awkward Array of jet pT
jet_pts = results["jet_pt_fetch"].jet_pt  # Awkward Array in GeV

# 4) Create a histogram: 100 bins from 0 to 500 GeV
h = Hist.new.Reg(100, 0, 500, name="pt", label="Jet $p_T$ [GeV]").Double()
h.fill(pt=jet_pts)

# 5) Plot and save to jet_pt.png
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(ax=ax, histtype="step", label="All jets")
ax.set_xlabel("Jet $p_T$ [GeV]")
ax.set_ylabel("Counts")
ax.legend()
plt.savefig("jet_pt.png")
plt.close(fig)
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 7.20 | 11640 | 632 | 12272 | 0.0283 |
| gpt-5 | 73.66 | 11639 | 2263 | 13902 | 0.0372 |
| gpt-5-mini | 26.13 | 11639 | 2552 | 14191 | 0.0080 |
| gpt-5-nano | 19.23 | 11639 | 3285 | 14924 | 0.0019 |
| gpt-4o | 10.80 | 11640 | 570 | 12210 | 0.0348 |
| o4-mini | 13.83 | 11639 | 1704 | 13343 | 0.0203 |
