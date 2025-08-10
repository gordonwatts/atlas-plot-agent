# Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.



## Model: gpt-4.1

- Accessed the given rucio dataset using ServiceX and func_adl with the PHYSLITE event format.
- Queried and extracted the Missing Transverse Energy (ETmiss, field `met`) from each event, converting it from MeV to GeV.
- Converted the resulting ServiceX output into an awkward array for manipulation.
- Created a 1D histogram (using `Hist`) of the ETmiss values for all events.
- Plotted the histogram using matplotlib and mplhep with ATLAS style, and wrote the plot to a PNG file (`etmiss.png`).

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

# 1. Access the dataset and query ETmiss
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)
base_query = FuncADLQueryPHYSLITE()
met_query = (
    base_query
    .Select(lambda e: e.MissingET().First())
    .Select(lambda met: {"etmiss": met.met() / 1000.0})  # Convert MeV to GeV
)

result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="met_query",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # REMOVE for full dataset; NFiles=1 for fast testing
                    Query=met_query,
                )
            ]
        )
    )
)

etmiss = result['met_query'].etmiss  # awkward array of ETmiss [GeV] per event

# Filter out None values (shouldn't exist, but just in case)
etmiss = etmiss[etmiss != None]

# 2. Make the histogram
# Define ETmiss bins: 0 to 800 GeV, in 40 bins (can increase if needed)
hist_etmiss = (
    Hist.new
    .Reg(40, 0, 800, name="etmiss", label="E$_{T}^{miss}$ [GeV]")
    .Int64()
)
hist_etmiss.fill(etmiss=ak.to_numpy(etmiss))

# 3. Plot the histogram
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
hist_etmiss.plot(histtype="step", ax=ax, linewidth=2)
ax.set_xlabel(r"$E_{T}^{miss}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("ATLAS Simulation\n$E_{T}^{miss}$ in {}\n(NFiles=1)".format(ds_name.split(":")[1]))
plt.tight_layout()
plt.savefig("etmiss.png")
plt.close()
```


## Model: gpt-5

- Used ServiceX (FuncADL against PHYSLITE) to skim only MissingET from the requested Rucio dataset.
- Converted ETmiss to GeV at the ServiceX level to minimize data volume.
- Brought results back as Awkward Arrays and filtered non-finite values.
- Filled a 1D histogram with Hist and saved the plot to a PNG file (etmiss.png).
- Ran on the full dataset (no NFiles limit) as requested.

```python
#!/usr/bin/env python3

# Fetch ETmiss from an ATLAS PHYSLITE dataset via ServiceX, then plot it.

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
from hist import Hist

plt.style.use(hep.style.ATLAS)

def main():
    # Dataset: mc23_13p6TeV JZ2 PHYSLITE
    ds_name = (
        "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
        "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
    )

    # Build query: select event MissingET (first element) and return met in GeV
    base = FuncADLQueryPHYSLITE()
    met_query = (
        base
        .Select(lambda e: e.MissingET().First())
        .Select(lambda m: {"met": m.met() / 1000.0})
    )

    # Deliver from ServiceX: run on full dataset (no NFiles specified)
    result = to_awk(
        deliver(
            ServiceXSpec(
                Sample=[
                    Sample(
                        Name="met_fetch",
                        Dataset=dataset.Rucio(ds_name),
                        Query=met_query,
                    )
                ]
            )
        )
    )

    # Extract Awkward Array of ETmiss (GeV)
    met = result["met_fetch"].met

    # Filter any non-finite values
    met = met[ak.is_finite(met)]

    # Build histogram (0–500 GeV, 100 bins)
    h = (
        Hist.new
        .Reg(100, 0.0, 500.0, name="met", label="E_T^miss [GeV]")
        .Int64()
    )

    # Fill histogram using Awkward -> NumPy view (no Python lists)
    h.fill(met=ak.to_numpy(met))

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    h.plot(ax=ax, histtype="step", linewidth=2, label="All events")
    ax.set_xlabel("E_T^miss [GeV]")
    ax.set_ylabel("Events")
    ax.set_title("ATLAS Open Data (mc23) — Missing Transverse Energy")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("etmiss.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
```


## Model: gpt-5-mini

- Fetched MissingET.met from the specified PHYSLITE rucio dataset via ServiceX (I used NFiles=1 for a quick/test run — see note in the code on how to change that to run on the full dataset).
- Converted MET from MeV to GeV (dividing by 1000 inside the query).
- Built a 1D histogram of ETmiss using hist, plotted it with mplhep/Matplotlib, and saved the plot to met_hist.png.
- All data manipulation of the ServiceX output uses Awkward arrays.

```python
# Fetch MissingET.met from the PHYSLITE dataset via ServiceX, histogram it, and save a PNG.
# Notes:
#  - This script uses NFiles=1 for a quick/test run. To run over the full dataset remove the NFiles argument
#    (or set NFiles to a larger value or leave it out entirely).
#  - MET is converted from MeV to GeV in the FuncADL query (divide by 1000).

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Build the FuncADL query: get the event MissingET (must First()) and extract .met() in GeV.
base_query = FuncADLQueryPHYSLITE()
met_query = (
    base_query
    .Select(lambda e: e.MissingET().First())                # MissingET is stored as a sequence -> take the first
    .Select(lambda m: {"met": m.met() / 1000.0})            # convert MeV -> GeV here
)

# 2) Define the dataset (the one you requested)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# 3) Deliver the query via ServiceX
#    Using NFiles=1 for a quick/test run. Remove NFiles (or set to a larger value) to run on the full dataset.
svc_spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="met_fetch",
            Dataset=dataset.Rucio(ds_name),
            NFiles=1,
            Query=met_query,
        )
    ]
)

result_dict = to_awk(deliver(svc_spec))

# result_dict is keyed by the Sample Name
met_awk = result_dict["met_fetch"].met  # awkward array of MET (GeV)

# 4) Basic checks & statistics (all using Awkward)
n_events = int(ak.num(met_awk)) if ak.num(met_awk).shape == () else int(ak.sum(ak.num(met_awk)))  # defensive
mean_met = float(ak.mean(met_awk)) if len(met_awk) > 0 else 0.0
max_met = float(ak.max(met_awk)) if len(met_awk) > 0 else 0.0
p99 = float(ak.quantile(met_awk, 0.99)) if len(met_awk) > 0 else 0.0

print(f"Fetched MET for {len(met_awk)} events (Awkward array).")
print(f"Mean MET = {mean_met:.2f} GeV, max MET = {max_met:.2f} GeV, 99th percentile = {p99:.2f} GeV")

# 5) Make a histogram and plot
#    Choose an upper edge based on the data to make a reasonable plot range.
upper_edge = max(100.0, max_met * 1.1)  # at least 100 GeV, or 10% above observed max
nbins = 100

h = (
    Hist.new.Reg(nbins, 0.0, upper_edge, name="met", label=r"$E_{T}^{miss}$ [GeV]")
        .Double()
)

# Convert awkward array to numpy for histogram filling. We avoid Python lists; conversion to numpy is standard.
met_np = ak.to_numpy(met_awk)

# Fill histogram
h.fill(met=met_np)

# Plot with mplhep ATLAS style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8, 6))
h.plot(ax=ax, histtype="stepfilled", facecolor="C0", edgecolor="black", label="Events")
ax.set_xlabel(r"$E_{T}^{miss}$ [GeV]")
ax.set_ylabel("Counts")
ax.legend()
ax.set_xlim(0, upper_edge)

plt.tight_layout()
outname = "met_hist.png"
plt.savefig(outname, dpi=150)
print(f"Saved MET histogram to {outname}")
```


## Model: gpt-5-nano

- I prepared a Python script that fetches MissingET (MET) per event from the specified PHYSLITE rucio dataset using ServiceX.
- I used an Awkward Array workflow to extract MET values in GeV (dividing by 1000 as needed) and then fill a Hist histogram with those values.
- The histogram is plotted and saved as met_hist.png using Hist for the data and Matplotlib for plotting.
- For testing, I fetch only 1 file (NFiles=1). You can increase NFiles to cover more events; I explain that in the code comments.

```python
# Script: fetch MET per event from PHYSLITE dataset, histogram, and save as PNG

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from hist import Hist
import matplotlib.pyplot as plt

# Dataset to analyze
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Base query: get MissingET and convert to GeV
base_query = FuncADLQueryPHYSLITE()

# MET per event (take the first MissingET object per event and read met in MeV, convert to GeV)
met_query = (
    base_query
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {"met": m.met() / 1000.0})
)

# Fetch data via ServiceX (test with 1 file)
met_fetch = deliver(
    ServiceXSpec(
        Sample=[
            Sample(
                Name="met_fetch",
                Dataset=dataset.Rucio(ds_name),
                NFiles=1,           # For testing; increase for full dataset if needed
                Query=met_query,
            )
        ]
    )
)

# Convert to Awkward Array
met_data = to_awk(met_fetch)

# Extract MET values (Awkward array of MET in GeV)
# The data structure is a dict: {"met_fetch": AwkwardArray with field "met"}
met_vals = met_data["met_fetch"].met

# Create a 1D histogram: MET [GeV] from 0 to 500 GeV
h_met = Hist.new.Reg(100, 0.0, 500.0, name="met", label="MET [GeV]").Weight()

# Fill histogram with MET values
h_met.fill(met=met_vals)

# Plot and save to PNG
plt.figure(figsize=(6, 4))
h_met.plot(histtype="step", linewidth=1.5, label="MET per event")
plt.xlabel("MET [GeV]")
plt.ylabel("Entries")
plt.title("Missing ET (MET) distribution")
plt.legend()
plt.tight_layout()
plt.savefig("met_hist.png")
plt.close()
```


## Model: gpt-4o

- Created a query to select the missing transverse energy (ETmiss) from the specified Rucio dataset.
- Used ServiceX to deliver the data, retrieving only the necessary missing ET information.
- Utilized Awkward Array to handle the resulting data.
- Used the Hist library to create a histogram of the ETmiss values.
- Plotted and saved the histogram as a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
from hist import Hist
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)  # Use ATLAS-like style for HEP plots

# Query to extract ETmiss from PHYSLITE xAOD data
base_query = FuncADLQueryPHYSLITE()
etmiss_query = (base_query
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {"etmiss": m.met() / 1000.0})  # Convert to GeV
)

# Define the dataset
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697")

# Fetch the data
etmiss_data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="etmiss_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Change to more if needed after verifying setup works
                    Query=etmiss_query,
                )
            ]
        ),
    )
)

# Extract the awkward array of ETmiss values
etmiss_values = etmiss_data["etmiss_fetch"].etmiss

# Create a histogram of ETmiss
h = (Hist.new
     .Reg(50, 0, 500, name="etmiss", label="Missing ET [GeV]")
     .Double()  # Use double precision for count storage
)

# Fill the histogram
h.fill(etmiss=etmiss_values)

# Plot the histogram
fig, ax = plt.subplots()
h.plot(histtype='step', linewidth=2, color='blue', ax=ax, label='ETmiss Distribution')
ax.set_xlabel("Missing ET [GeV]")
ax.set_ylabel("Events")
plt.legend()
plt.savefig("etmiss_distribution.png")
plt.show()
```


## Model: o4-mini

- Defined a FuncADL query to select the event‐level MissingET object, converted its `.met()` from MeV to GeV.
- Used ServiceX to deliver that query on one file of the specified Rucio PHYSLITE dataset and loaded into an Awkward Array.
- Constructed a 1D histogram of ETmiss with `hist`, filled it from the Awkward Array.
- Plotted the histogram with ATLAS styling via `mplhep` and saved the figure as `met.png`.

```python
# Fetch, histogram, and plot Missing ET (ETmiss) from DAOD_PHYSLITE dataset
import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# 1) Build the FuncADL query to get ETmiss (convert MeV to GeV)
base_query = FuncADLQueryPHYSLITE()
met_query = (
    base_query
    # MissingET() returns a list with one element
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {"met": m.met() / 1000.0})
)

# 2) Deliver the query via ServiceX (using 1 file for testing)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2."
    "deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)
spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="met_fetch",
            Dataset=dataset.Rucio(ds_name),
            NFiles=1,          # start with 1 file for testing
            Query=met_query,
        )
    ]
)
result = to_awk(deliver(spec))

# 3) Extract the Awkward Array of ETmiss
met_array = result["met_fetch"].met  # in GeV

# 4) Build and fill a histogram of ETmiss
h = (
    hist.Hist.new
    .Reg(100, 0, 500, name="met", label="Missing $E_T$ [GeV]")
    .Double()
)
h.fill(met=met_array)

# 5) Plot with ATLAS style and save to PNG
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(ax=ax, histtype="step", color="black")
ax.set_xlabel("Missing $E_T$ [GeV]")
ax.set_ylabel("Events")
ax.legend(["ETmiss"])
fig.tight_layout()
fig.savefig("met.png")
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 7.85 | 11640 | 672 | 12312 | 0.0287 |
| gpt-5 | 54.40 | 11639 | 2641 | 14280 | 0.0410 |
| gpt-5-mini | 26.15 | 11639 | 2843 | 14482 | 0.0086 |
| gpt-5-nano | 14.52 | 11639 | 3619 | 15258 | 0.0020 |
| gpt-4o | 7.01 | 11640 | 568 | 12208 | 0.0348 |
| o4-mini | 12.23 | 11639 | 1343 | 12982 | 0.0187 |
