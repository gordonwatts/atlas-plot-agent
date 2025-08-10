# Plot the scalar sum in each event of the pT of jets with pT > 30 GeV that are not within 0.4 in ΔR of any light lepton with pT > 10 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697.



## Model: gpt-4.1

- Fetched from the specified ttbar dataset (all-hadronic, PHYSLITE format) with ServiceX, retrieving:
  - For each event: jets with their pt/eta/phi, electrons and muons with pt/eta/phi (to define light leptons).
- In the ServiceX query, applied:
  - Jet pT > 30 GeV in query (reducing data volume early).
  - Electron/muon pT > 10 GeV in query.
- Constructed awkward arrays of selected jets and light leptons.
- Built vectors for jets and leptons using the `vector` library (enabled awkward behaviors).
- For each event, masked out jets within ΔR < 0.4 of any selected light lepton.
- Calculated, for each event, the sum of pT of the jets that survive the above cleaning.
- Filled and plotted a histogram of this scalar sum over all events, and saved the plot as `jetsum_scalar_pt.png`.

```python
import awkward as ak
import vector
import numpy as np
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Register vector behaviors for awkward array
vector.register_awkward()

# -- Step 1: Build ServiceX FuncADL query --

base_query = FuncADLQueryPHYSLITE()

# For light leptons: electrons and muons with pt > 10 GeV
def lepton_query(lepton_getter):
    return lepton_getter \
        .Where(lambda lep: lep.pt()/1000.0 > 10.0) \
        .Select(lambda lep: {
            "pt": lep.pt()/1000.0,
            "eta": lep.eta(),
            "phi": lep.phi(),
        })

query = (
    base_query
    .Select(lambda e: (
        # Jets with pt>30
        e.Jets()
          .Where(lambda jet: jet.pt()/1000.0 > 30.0)
          .Select(lambda jet: {
              "pt": jet.pt()/1000.0,
              "eta": jet.eta(),
              "phi": jet.phi(),
          }),
        # electrons
        e.Electrons(),
        # muons
        e.Muons(),
    ))
    .Select(lambda obj: {
        "jets": obj[0],
        "ele": obj[1]
            .Where(lambda ele: ele.pt()/1000.0 > 10.0)
            .Select(lambda ele: {
                "pt": ele.pt()/1000.0,
                "eta": ele.eta(),
                "phi": ele.phi(),
            }),
        "mu": obj[2]
            .Where(lambda mu: mu.pt()/1000.0 > 10.0)
            .Select(lambda mu: {
                "pt": mu.pt()/1000.0,
                "eta": mu.eta(),
                "phi": mu.phi(),
            }),
    })
)

# -- Step 2: Fetch data from ServiceX --
ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"
data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[Sample(
                Name="jetsum",
                Dataset=dataset.Rucio(ds_name),
                NFiles=1,  # TEST on 1 file; increase for full stats
                Query=query,
            )]
        )
    )
)
events = data["jetsum"]

# -- Step 3: Build Lorentz vectors, mask jets near light leptons in each event --
def jetsum_scalar_pt(jet_array, le_array, mu_array):
    # Make jagged arrays of jet/lepton (Momentum4D: pt, eta, phi, mass=0)
    jets = ak.zip(
        {
            "pt": jet_array["pt"],
            "eta": jet_array["eta"],
            "phi": jet_array["phi"],
            "mass": ak.zeros_like(jet_array["pt"]),
        },
        with_name="Momentum4D"
    )
    eles = ak.zip(
        {
            "pt": le_array["pt"],
            "eta": le_array["eta"],
            "phi": le_array["phi"],
            "mass": ak.zeros_like(le_array["pt"]),
        },
        with_name="Momentum4D"
    )
    mus = ak.zip(
        {
            "pt": mu_array["pt"],
            "eta": mu_array["eta"],
            "phi": mu_array["phi"],
            "mass": ak.zeros_like(mu_array["pt"]),
        },
        with_name="Momentum4D"
    )
    # All leptons: concatenate electrons and muons per event
    leptons = ak.concatenate([eles, mus], axis=1)
    # For each event: 
    if len(jets) == 0:
        return ak.Array([])  # empty
    # Calculate ΔR between each jet and each lepton in event
    # Shape: (nevents, njets, nleptons)
    if ak.max(ak.num(leptons, axis=1)) == 0:
        # No leptons at all: all jets pass
        mask = ak.ones_like(jets.pt, dtype=bool)
    else:
        # Outer loop: for each event
        jet_lep_pairs = ak.cartesian([jets, leptons], axis=1)  # shape (nevents, njets, nleptons)
        jets_rep, leps_rep = ak.unzip(jet_lep_pairs)
        dR = jets_rep.deltaR(leps_rep)
        # For each jet in each event, require dR >= 0.4 for all leptons
        mask = ak.all((dR >= 0.4) | (ak.num(leptons, axis=1) == 0)[:, None], axis=2)
        # (If no leptons, mask=True for jets)
    # For each event, sum scalar pt of surviving jets
    jets_pass = jets[mask]
    sumpt = ak.sum(jets_pass.pt, axis=1)
    return sumpt

jetsum = jetsum_scalar_pt(events["jets"], events["ele"], events["mu"])

# -- Step 4: Plot histogram --
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# Set style
plt.style.use(hep.style.ATLAS)

# Only select non-zero entries (no jets passing cut => sumpt=0)
valid_jetsum = jetsum[~ak.is_none(jetsum) & (jetsum > 0)]

h = (
    Hist.new
    .Reg(40, 0, 2000, name="jetsumpt", label=r"Scalar sum p$_T$ (GeV)")
    .Int64()
)
h.fill(jetsumpt=ak.flatten(valid_jetsum))
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="Events")
plt.xlabel(r"Scalar sum p$_T$ (GeV) of jets (see description)")
plt.ylabel("Events / bin")
plt.legend()
plt.tight_layout()
plt.savefig("jetsum_scalar_pt.png")
plt.close()
```


## Model: gpt-5

- Uses ServiceX (func_adl) to skim only what’s needed from the given DAOD_PHYSLITE dataset:
  - Jets with pT > 30 GeV (pt/eta/phi)
  - Light leptons (electrons and muons) with pT > 10 GeV (pt/eta/phi)
- Combines electrons and muons, then removes jets within ΔR < 0.4 of any light lepton using Awkward + Vector (no Python loops or lists)
- Computes the per-event scalar sum of the remaining jet pT’s
- Fills and plots a histogram with Hist and saves to PNG
- Note: The code runs on 1 file for a quick test. Remove NFiles to run on the full dataset

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

def main():
    # Build a ServiceX func_adl query against PHYSLITE:
    # - Only jets with pt > 30 GeV
    # - Only electrons and muons with pt > 10 GeV
    # Return only pt/eta/phi for each collection (units converted to GeV)
    base = FuncADLQueryPHYSLITE()
    query = (
        base
        .Select(lambda e: (e.Jets(), e.Electrons(), e.Muons()))
        .Select(lambda cols: {
            # Jets: apply cut at ServiceX level
            "jet_pt":  cols[0].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.pt() / 1000.0),
            "jet_eta": cols[0].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.eta()),
            "jet_phi": cols[0].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.phi()),
            # Electrons: pt > 10 GeV
            "ele_pt":  cols[1].Where(lambda el: el.pt() / 1000.0 > 10.0).Select(lambda el: el.pt() / 1000.0),
            "ele_eta": cols[1].Where(lambda el: el.pt() / 1000.0 > 10.0).Select(lambda el: el.eta()),
            "ele_phi": cols[1].Where(lambda el: el.pt() / 1000.0 > 10.0).Select(lambda el: el.phi()),
            # Muons: pt > 10 GeV
            "mu_pt":  cols[2].Where(lambda mu: mu.pt() / 1000.0 > 10.0).Select(lambda mu: mu.pt() / 1000.0),
            "mu_eta": cols[2].Where(lambda mu: mu.pt() / 1000.0 > 10.0).Select(lambda mu: mu.eta()),
            "mu_phi": cols[2].Where(lambda mu: mu.pt() / 1000.0 > 10.0).Select(lambda mu: mu.phi()),
        })
    )

    ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"

    # Quick test on a single file; remove NFiles to process the full dataset
    spec = ServiceXSpec(
        Sample=[
            Sample(
                Name="ttbar_allhad_sumpt",
                Dataset=dataset.Rucio(ds_name),
                NFiles=1,
                Query=query,
            )
        ]
    )

    result = to_awk(deliver(spec))
    events = result["ttbar_allhad_sumpt"]

    # Build Momentum4D vectors for jets and light leptons (mass set to 0)
    jets = ak.zip(
        {
            "pt": events["jet_pt"],
            "eta": events["jet_eta"],
            "phi": events["jet_phi"],
            "mass": 0.0 * events["jet_pt"],
        },
        with_name="Momentum4D",
    )

    electrons = ak.zip(
        {
            "pt": events["ele_pt"],
            "eta": events["ele_eta"],
            "phi": events["ele_phi"],
            "mass": 0.0 * events["ele_pt"],
        },
        with_name="Momentum4D",
    )

    muons = ak.zip(
        {
            "pt": events["mu_pt"],
            "eta": events["mu_eta"],
            "phi": events["mu_phi"],
            "mass": 0.0 * events["mu_pt"],
        },
        with_name="Momentum4D",
    )

    # Combine electrons and muons into a single light-lepton collection
    leptons = ak.concatenate([electrons, muons], axis=1)

    # Compute ΔR between each jet and each lepton, grouped per jet (nested=True)
    pairs = ak.cartesian({"jet": jets, "lep": leptons}, axis=1, nested=True)
    dR = pairs["jet"].deltaR(pairs["lep"])

    # Min ΔR to any lepton per jet; if no leptons in event, min will be None => fill with large number
    min_dR = ak.fill_none(ak.min(dR, axis=2), 999.0)

    # Keep jets with min ΔR >= 0.4 to any light lepton (effectively all jets pass if no leptons)
    keep_jets_mask = min_dR >= 0.4

    # Scalar sum of pT per event for the selected jets
    selected_jet_pts = jets.pt[keep_jets_mask]
    st_per_event = ak.sum(selected_jet_pts, axis=1)  # [GeV]

    # Make a histogram and plot
    plt.style.use(hep.style.ATLAS)
    h = (
        Hist.new
        .Reg(60, 0.0, 1200.0, name="st", label="Scalar sum pT of jets (ΔR>0.4 from light leptons) [GeV]")
        .Int64()
    )
    h.fill(st=ak.to_numpy(st_per_event))

    fig, ax = plt.subplots(figsize=(8, 6))
    h.plot(ax=ax, histtype="step", color="C0", linewidth=1.5, label="Events")
    ax.set_xlabel("Scalar sum pT [GeV]")
    ax.set_ylabel("Events / bin")
    ax.legend()
    ax.set_title("Sum pT of jets (pT>30 GeV) not within ΔR<0.4 of light leptons (pT>10 GeV)")

    out_name = "sumpt_jets_noLeptonDR04.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_name}")

if __name__ == "__main__":
    main()
```


## Model: gpt-5-mini

- I fetch jets, electrons and muons from the requested PHYSLITE rucio dataset using ServiceX, applying the pT cuts at the query level to reduce transferred data (jets with pT>30 GeV, leptons with pT>10 GeV).
- I build awkward/vector Lorentz-like objects for jets and leptons, compute ΔR per jet to all light leptons in the same event, reject jets that have any lepton within ΔR < 0.4, form the scalar sum of the remaining jets' pT per event (HT-like quantity), and histogram that per-event sum.
- I used NFiles=1 in the ServiceX deliver call to keep the example quick for testing. If you want a production-level result, remove NFiles or increase it to sample more files (explain if you want guidance on how many files to fetch).
- The script writes the histogram image to ht_jets_not_near_leptons.png.

```python
# Copy of the xAOD tool-injection helpers (required to be present in the script per the instructions).
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar
import ast
from func_adl import ObjectStream, func_adl_callable
T = TypeVar("T")

@dataclass
class ToolInfo:
    name: str


def make_a_tool(
    query: ObjectStream[T],
    tool_name: str,
    tool_type: str,
    include_files: Optional[List[str]],
    init_lines: List[str] = [],
) -> Tuple[ObjectStream[T], ToolInfo]:
    query_base = query.MetaData(
        {
            "metadata_type": "inject_code",
            "name": tool_name,
            "header_includes": include_files,
            "private_members": [f"{tool_type} *{tool_name};"],
            "instance_initialization": [
                f'{tool_name}(new {tool_type} ("{tool_name}"))'
            ],
            "initialize_lines": [l.format(tool_name=tool_name) for l in init_lines],
            "link_libraries": ["xAODBTaggingEfficiencyLib"],
        }
    )

    return query_base, ToolInfo(name=tool_name)


def make_tool_accessor(
    t_info: ToolInfo,
    function_name: str,
    source_code: List[str],
    arguments: Iterable[Tuple[str, type]],
    return_type_cpp: str,
    return_type_python: str
):
    def tool_callback(
        s: ObjectStream[T], a: ast.Call
    ) -> Tuple[ObjectStream[T], ast.Call]:
        new_s = s.MetaData(
            {
                "metadata_type": "add_cpp_function",
                "name": function_name,
                "code": [
                    "double result;",
                    *[l.format(tool_name=t_info.name) for l in source_code],
                ],
                "result": "result",
                "include_files": [],
                "arguments": [a[0] for a in arguments],
                "return_type": return_type_cpp,
            }
        )
        return new_s, a

    def tool_call(**arg_dict):
        ...
    tool_call.__name__ = function_name
    tool_call.__annotations__['return'] = eval(return_type_python)

    return func_adl_callable(tool_callback)(tool_call)


# The actual analysis code
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

# ServiceX / FuncADL imports
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Enable vector behaviors on awkward arrays
vector.register_awkward()

# Build the FuncADL query (PHYSLITE stream)
# - Jets filtered to pt>30 GeV at query time
# - Electrons and Muons filtered to pt>10 GeV at query time
base = FuncADLQueryPHYSLITE()

query = (
    base
    .Select(lambda e: {
        # jets: keep only kinematics we need and apply pt > 30 GeV (note: pt() is in MeV -> divide by 1000)
        "jets": e.Jets().Where(lambda j: j.pt() / 1000.0 > 30.0)
                    .Select(lambda j: {"pt": j.pt() / 1000.0, "eta": j.eta(), "phi": j.phi()}),

        # electrons: keep those with pt > 10 GeV (MeV -> GeV)
        "electrons": e.Electrons().Where(lambda el: el.pt() / 1000.0 > 10.0)
                           .Select(lambda el: {"pt": el.pt() / 1000.0, "eta": el.eta(), "phi": el.phi()}),

        # muons: keep those with pt > 10 GeV
        "muons": e.Muons().Where(lambda mu: mu.pt() / 1000.0 > 10.0)
                    .Select(lambda mu: {"pt": mu.pt() / 1000.0, "eta": mu.eta(), "phi": mu.phi()}),
    })
)

# Dataset (PHYSLITE rucio dataset specified by the user)
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697")

# Deliver the query. Start with NFiles=1 for a quick test (increase or omit NFiles for a full run).
# NOTE: This will run remotely and return the selected fields as awkward arrays.
svc_spec = ServiceXSpec(
    Sample=[
        Sample(Name="sample1", Dataset=dataset.Rucio(ds_name), NFiles=1, Query=query)
    ]
)

print("Submitting ServiceX job (NFiles=1 for a quick test). This may take a short while...")
delivered = deliver(svc_spec)
awk_data = to_awk(delivered)

# Extract the arrays (awk_data keyed by Sample Name)
sample = awk_data["sample1"]
jets = sample.jets      # jagged array: events -> list of jets (each jet has fields pt, eta, phi)
electrons = sample.electrons
muons = sample.muons

# Combine electrons and muons into a single "light lepton" collection per event
# ak.concatenate will concatenate along the inner (per-event) axis; axis=1 means concatenate lists per event
# If one collection is empty for some events, concatenate still behaves correctly.
leptons = ak.concatenate([electrons, muons], axis=1)

# Build vector-like objects recognized by vector: use fields pt, eta, phi, mass
# We set mass ~ 0.0 (mass is required by the Momentum4D behavior)
jets_vec = ak.zip(
    {
        "pt": jets.pt,
        "eta": jets.eta,
        "phi": jets.phi,
        "mass": ak.zeros_like(jets.pt)  # mass ~ 0
    },
    with_name="Momentum4D",
)

leptons_vec = ak.zip(
    {
        "pt": leptons.pt,
        "eta": leptons.eta,
        "phi": leptons.phi,
        "mass": ak.zeros_like(leptons.pt)
    },
    with_name="Momentum4D",
)

# For each jet in each event, compute whether any lepton in the same event is within ΔR < 0.4.
# Strategy: use ak.cartesian with nested=True to get, per-jet, the list of dR values to each lepton.
# If an event has zero leptons, the per-jet list will be empty -> ak.any(empty) is False (no lepton close).
pairs = ak.cartesian({"j": jets_vec, "l": leptons_vec}, axis=1, nested=True)
# pairs.j and pairs.l are nested lists per event: shape = events -> jets -> leptons
if len(pairs) == 0:
    # Defensive: if no events (empty result), make an empty array of sums and produce an empty histogram.
    ht_per_event = ak.Array([])
else:
    dR = pairs.j.deltaR(pairs.l)  # per event: list (over jets) of lists (over leptons) of ΔR
    # For each jet, True if any lepton within 0.4
    is_close = ak.any(dR < 0.4, axis=1)  # per event: list of booleans for each jet

    # Select jets that are NOT close to any lepton
    jets_pt = jets.pt  # per-event lists of jet pT in GeV
    jets_keep_mask = ak.logical_not(is_close)
    # Keep only jet pts where mask True (mask aligns with jets)
    jets_pt_keep = jets_pt[jets_keep_mask]

    # Scalar sum of kept jets' pT per event
    ht_per_event = ak.sum(jets_pt_keep, axis=1)  # shape: (nEvents,)
    # If some events have no jets or no kept jets, ak.sum returns None; replace None with 0.0
    ht_per_event = ak.fill_none(ht_per_event, 0.0)

# Now make a histogram of HT (scalar sum per event)
# Choose a reasonable binning, e.g. 0 - 1000 GeV in 50 bins (adjust as desired)
nbins = 50
ht_max = 1000.0
hist_ht = Hist.new.Reg(nbins, 0.0, ht_max, name="HT", label="Scalar sum of kept jet p_{T} [GeV]").Double()

# Convert the awkward 1D array into a NumPy array for filling the hist (this conversion is only for histogramming/plotting)
ht_np = np.asarray(ak.to_numpy(ht_per_event)) if hasattr(ak, "to_numpy") else np.asarray(ak.to_list(ht_per_event))
# Fill the histogram
hist_ht.fill(HT=ht_np)

# Plot with mplhep/ATLAS style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8,6))
hist_ht.plot(ax=ax, histtype="stepfilled", label="jets (pT>30 GeV, not within ΔR<0.4 of any lepton pT>10 GeV)", color="C0", edgecolor="black", alpha=0.7)
ax.set_xlim(0, ht_max)
ax.set_xlabel("Scalar sum of jet p$_{T}$ per event [GeV]")
ax.set_ylabel("Events")
ax.legend()
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

outname = "ht_jets_not_near_leptons.png"
plt.savefig(outname, dpi=150)
print(f"Saved histogram to {outname}")
```


## Model: gpt-5-nano

- I will fetch the PHYSLITE DAOD from the requested rucio dataset using ServiceX, select jets with pT > 30 GeV, select light leptons (electrons and muons) with pT > 10 GeV, compute per-event the sum of jet pT for jets that are not within ΔR < 0.4 of any light lepton, and then plot a histogram of these per-event sums to a PNG file using Hist.
- I will implement the computation in Awkward Array form (vectorized), and I will output the final histogram as a PNG via Matplotlib.
- The code below uses:
  - ServiceX with a PHYSLITE base query (FuncADLQueryPHYSLITE)
  - Awkward Array operations for jet/lepton selection and the ΔR logic
  - Hist for the histogram and Matplotlib to save to PNG

```python
# pip-install hints (not executed here):
# pip install servicex func_adl_servicex_xaodr25 awkward-plotting-helpers hist
# pip install vector

import ak  # awkward
import vector  # for Lorentz vectors (Momentum4D)
import numpy as np
import hist
import matplotlib.pyplot as plt

# ServiceX imports
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset

# Initialize vector behaviors (needed for .pt, .eta, .phi, and .deltaR)
vector.register_awkward()

# The dataset (PHYSLITE) you specified
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE."
    "e8514_s4369_r16083_p6697"
)

# Build the base PHYSLITE query
base_query = FuncADLQueryPHYSLITE()

# 1) Jets with pT > 30 GeV (in GeV; input is MeV, so divide by 1000)
# We'll keep the event structure, so each event has its list of jets passing the cut.
jets_pass = (
    base_query
    .Select(lambda e: e.Jets())  # per-event jets
    .Select(lambda jets: jets.Where(lambda j: j.pt() / 1000.0 > 30.0))  # GeV threshold
)

# 2) Light leptons with pT > 10 GeV
# Electrons
elec_pass = (
    base_query
    .Select(lambda e: e.Electrons())
    .Select(lambda eles: eles.Where(lambda ele: ele.pt() / 1000.0 > 10.0))
)
# Muons
muon_pass = (
    base_query
    .Select(lambda e: e.Muons())
    .Select(lambda mus: mus.Where(lambda mu: mu.pt() / 1000.0 > 10.0))
)

# 3) Combine leptons per event: electrons + muons
# Note: We rely on Awkward's per-event concatenation
# leptons_all has the same event structure as jets_pass (a jagged array per event)
leptons_all = (
    # concatenate per-event lists: electrons first, then muons
    # We perform a per-event concatenate in a single Select to avoid materializing Python lists
    base_query.Select(
        lambda e: ak.concatenate(
            [e.Electrons().Where(lambda ele: ele.pt() / 1000.0 > 10.0),
             e.Muons().Where(lambda mu: mu.pt() / 1000.0 > 10.0)],
            axis=0,  # per-event concatenation
        )
    )
)

# 4) Compute per-event the scalar sum of jet pT for jets that are not ΔR < 0.4 away from any lepton
# Approach (vectorized):
# - Build a cartesian product per event: (jet, lepton)
# - Compute deltaR for each pair
# - Mark pairs as "close" if deltaR <= 0.4
# - For each jet, determine if there exists any "close" lepton
# - If no lepton is close, count the jet's pT once toward the event sum
# - Finally, divide by the number of leptons to correct for double-counting (because each jet is paired with all leptons)
# Note: This relies on the cartesian behavior producing per-event [jet, lepton] pairs with a structure enabling per-jet grouping.
pairs = ak.cartesian({"jet": jets_pass, "lep": leptons_all}, axis=1)

# deltaR for each pair
# jets are Momentum4D-like objects with .deltaR available via Vector behaviors
dR_pairs = pairs.jet.deltaR(pairs.lep)  # shape: [event, jet_index, lepton_index]

# Close pairs: within 0.4
close = dR_pairs <= 0.4

# For each jet, determine if any close lepton exists
has_close_per_jet = ak.any(close, axis=2)  # shape: [event, jet]

# Mask for jets to keep: no close lepton for that jet
keep_jet_mask = ~has_close_per_jet  # shape: [event, jet]

# Sum of pT for kept jets, counted once per jet.
# We exploit the structure: pairs.jet.pt gives the jet pT for each (jet, lepton) pair.
# If a jet is kept, all its lepton-pair entries will be kept; sum across leptons gives pT * n_leptons
# We then divide by n_leptons (L) per event to get pT per kept jet.
jet_pt_per_pair = pairs.jet.pt()  # shape: [event, jet, lepton]
# Apply the per-jet keep mask; broadcasting keeps alignment
pt_sum_per_pair = ak.where(keep_jet_mask, jet_pt_per_pair, 0.0)

# Sum over the lepton axis (axis=2) to accumulate per jet
sum_pt_per_jet = ak.sum(pt_sum_per_pair, axis=2)  # shape: [event, jet]

# Total per-event sum across all jets (only kept jets contribute)
total_sum_per_event = ak.sum(sum_pt_per_jet, axis=1)  # shape: [event]

# Number of leptons per event (for division by L)
L_per_event = ak.num(leptons_all)  # shape: [event]

# Per-event scalar sum of jet pT for jets not close to any lepton
# Handle cases with L=0 gracefully (set sum to 0 in that case)
def safe_divide(numer, denom):
    # numer and denom are Awkward Arrays
    return ak.where(denom > 0, numer / denom, 0.0)

per_event_sum = safe_divide(total_sum_per_event, L_per_event)

# 5) Build a histogram of per_event_sum
# Define a simple 1D histogram: 100 bins from 0 to some max (choose 0-2000 GeV as a rough range)
import hist as hist

h = (
    Hist.new.Reg(100, 0, 2000, name="sum_pt", label="Scalar sum of jet pT (GeV)")
)

# Fill the histogram with the per-event sums
# The ServiceX/Awkward data is in GeV because we divided by 1000 earlier for jet pT, and leptons were just used for ΔR.
# Note: per_event_sum is in GeV by construction (since jet.pt()/1000.0 is GeV)
h.fill(sum_pt=ak.to_list(per_event_sum))

# Plotting the histogram and saving to PNG
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

# Extract a 1D numpy array and the bin edges
edges = h.axes["sum_pt"].edges
counts = h.view().flatten()

# Simple matplotlib plot for a quick PNG
plt.figure(figsize=(8, 6))
# Use a quick bar-like plot using bin centers
bin_centers = 0.5 * (edges[:-1] + edges[1:])
plt.bar(bin_centers, counts, width=(edges[1]-edges[0]), align="center", edgecolor="black")
plt.xlabel("Scalar sum of jet pT (GeV)")
plt.ylabel("Entries")
plt.title("Per-event scalar sum of jet pT for jets with pT>30 GeV not within ΔR<0.4 of any lepton (pT>10 GeV)")
plt.tight_layout()
plt.savefig("per_event_scalar_sum_jets_not_near_leptons.png")
plt.close()

# End of script
```

Notes and caveats:
- The deltaR-based separation logic is implemented using a per-event cross-product (jet, lepton) approach with ak.cartesian, then computing per-pair deltaR. We then determine per-jet if any near-lepton exists. Jets without any near-lepton contribute their pT once to the per-event sum.
- We divide the per-jet pT sum by the number of leptons per event to get a per-jet pT contribution of 1 per kept jet, since each kept jet contributes once for each lepton in the cross-product. This cancels the L-fold repetition.
- If an event has zero leptons above 10 GeV, per-event sum is defined as 0.
- This code emphasizes keeping data in Awkward arrays and uses the Vector-based methods for momentum-related quantities (pt, eta, phi) and deltaR when available.
- The final plot is saved as per_event_scalar_sum_jets_not_near_leptons.png.

If you’d like me to tailor the code further (e.g., choose a different histogram range, add axis labels, or write multiple plots for sanity checks), I can adjust accordingly.


## Model: gpt-4o

- This code fetches data from the specified dataset using ServiceX and filters jets and light leptons based on the given pT criteria.
- It calculates the scalar sum of the pT of jets in each event that are not within 0.4 in ΔR of any light lepton.
- The result is then plotted as a histogram and saved as a PNG file.
  
```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import vector
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# Register vector behaviors
vector.register_awkward()

# Define the dataset name
ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"

# Base query for PHYSLITE
base_query = FuncADLQueryPHYSLITE()

# Query to get jets and leptons with specified pT
query = (
    base_query
    .Select(lambda e: {
        "jets": e.Jets().Select(lambda j: {
            "pt": j.pt() / 1000.0,
            "eta": j.eta(),
            "phi": j.phi()
        }).Where(lambda j: j["pt"] > 30.0),
        "leptons": e.Electrons().Select(lambda ele: {
            "pt": ele.pt() / 1000.0,
            "eta": ele.eta(),
            "phi": ele.phi()
        }).Where(lambda ele: ele["pt"] > 10.0)
        .Concat(
            e.Muons().Select(lambda mu: {
                "pt": mu.pt() / 1000.0,
                "eta": mu.eta(),
                "phi": mu.phi()
            }).Where(lambda mu: mu["pt"] > 10.0)
        )
    })
)

# Fetch the data
data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_lepton_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        ),
    )
)

# Extract jets and leptons
jets = data["jet_lepton_fetch"].jets
leptons = data["jet_lepton_fetch"].leptons

# Calculate DeltaR between jets and leptons
pairs = ak.cartesian({"jet": jets, "lepton": leptons}, nested=True)
delta_r = vector.Array(pairs["jet"]).deltaR(vector.Array(pairs["lepton"]))

# Mask jets that are not within 0.4 ΔR of any light lepton
dr_mask = ak.all(delta_r > 0.4, axis=-1)
filtered_jets = jets[dr_mask]

# Scalar sum of pT of filtered jets
jet_pts_sum = ak.sum(filtered_jets.pt, axis=-1)

# Plot histogram
plt.style.use(hep.style.ATLAS)  # Use ATLAS style for the plot
h = Hist.new.Reg(50, 0, 500, name="sum_jet_pt", label="Scalar Sum of Jet pT [GeV]").Double()
h.fill(sum_jet_pt=ak.to_numpy(ak.flatten(jet_pts_sum)))

h.plot(histtype='fill', label="Events")

plt.xlabel("Scalar Sum of Jet $p_T$ [GeV]")
plt.ylabel("Number of Events")
plt.legend()
plt.title("Scalar Sum of Jet $p_T$ Distribution")
plt.grid(True)

# Save the plot
plt.savefig("jet_scalar_sum_pt.png")
plt.show()
```


## Model: o4-mini

- Built a FuncADL query on the ATLAS PHYSLITE DAOD to retrieve per-event lists of jets, electrons, and muons (pt, η, φ).
- Delivered the query on 1 file of the specified Rucio dataset via ServiceX, converting the result to Awkward arrays.
- Registered Vector’s Awkward behaviors and built Lorentz‐vector objects for jets and leptons; applied pT cuts (>30 GeV for jets, >10 GeV for leptons).
- Computed ΔR between each jet and all light leptons in the same event; vetoed jets with any ΔR<0.4.
- Calculated the scalar sum of the surviving jet pT per event.
- Filled a 50-bin histogram from 0–2000 GeV with these per-event sums, styled it in ATLAS format, and saved to `sum_pt.png`.

```python
import awkward as ak
import vector
import matplotlib.pyplot as plt
import mplhep as hep
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import ServiceXSpec, Sample, dataset, deliver
from servicex_analysis_utils import to_awk
from hist import Hist

# 1) Enable Vector behaviors for Akward
vector.register_awkward()  

# 2) Build the FuncADL query to get pt, eta, phi of electrons, muons, jets (in GeV/radians)
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: (e.Electrons(), e.Muons(), e.Jets()))
    .Select(lambda t: {
        'ele_pt' : t[0].Select(lambda l: l.pt()/1000.0),
        'ele_eta': t[0].Select(lambda l: l.eta()),
        'ele_phi': t[0].Select(lambda l: l.phi()),
        'mu_pt'  : t[1].Select(lambda l: l.pt()/1000.0),
        'mu_eta' : t[1].Select(lambda l: l.eta()),
        'mu_phi' : t[1].Select(lambda l: l.phi()),
        'jet_pt' : t[2].Select(lambda j: j.pt()/1000.0),
        'jet_eta': t[2].Select(lambda j: j.eta()),
        'jet_phi': t[2].Select(lambda j: j.phi()),
    })
)

# 3) Fetch data (start with NFiles=1 for testing)
rucio_name = (
    "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad."
    "deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"
)
awk_dict = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="events",
                    Dataset=dataset.Rucio(rucio_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        )
    )
)
evt = awk_dict["events"]

# 4) Build vector arrays
jets = ak.zip(
    {"pt": evt.jet_pt, "eta": evt.jet_eta, "phi": evt.jet_phi},
    with_name="Momentum4D",
)
eles = ak.zip(
    {"pt": evt.ele_pt, "eta": evt.ele_eta, "phi": evt.ele_phi},
    with_name="Momentum4D",
)
mus = ak.zip(
    {"pt": evt.mu_pt, "eta": evt.mu_eta, "phi": evt.mu_phi},
    with_name="Momentum4D",
)
leps = ak.concatenate([eles, mus], axis=1)

# 5) Apply pT cuts
good_jets = jets[jets.pt > 30.0]
good_leps = leps[leps.pt > 10.0]

# 6) Compute ΔR matrix [n_events, n_jets, n_leps] and veto jets near any lepton
dR = good_jets.deltaR(good_leps) 
# keep jets with all ΔR > 0.4
jet_mask = ak.all(dR > 0.4, axis=2)

# 7) Sum surviving jet pT per event
selected_jets = good_jets[jet_mask]
sum_pt_per_event = ak.sum(selected_jets.pt, axis=1)

# 8) Fill histogram
hist = (
    Hist.new
    .Reg(50, 0, 2000, name="sum_pt", label="Scalar sum of jet $p_T$ [GeV]")
    .Int64()
)
# convert to numpy for filling
import numpy as np
vals = ak.to_numpy(sum_pt_per_event)
hist.fill(sum_pt=vals)

# 9) Plot and save
plt.style.use(hep.style.ATLAS)
ax = hist.plot(histtype="stepfilled", color="C0", label="ttbar all‐had")
ax.legend()
ax.set_ylabel("Entries")
ax.set_xlabel("Scalar sum of surviving jets $p_T$ [GeV]")
plt.savefig("sum_pt.png")
```


## Model: openai/gpt-oss-120b

- Construct a **FuncADL** query that extracts per‑event jet kinematics (`pt`, `eta`, `phi`) with `pt > 30 GeV` and electron & muon kinematics (`pt`, `eta`, `phi`) with `pt > 10 GeV`.
- Use **ServiceX** to run the query on the requested Rucio dataset and obtain an Awkward‑array dictionary.
- Build Vector‑compatible Lorentz‑vector records for jets and “light leptons” (electrons + muons) and compute the ΔR between every jet and every lepton in the same event.
- Keep only jets that are **not** within ΔR < 0.4 of any selected lepton, then compute the scalar sum of the remaining jet pₜ per event.
- Fill a one‑dimensional histogram of this scalar sum with **hist**, plot it with **matplotlib** & **mplhep**, and save the figure as `jet_ht.png`.

```python
# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

# Register vector behaviours for Awkward
vector.register_awkward()

# ------------------------------------------------------------
# 1. Define the FuncADL query
# ------------------------------------------------------------
# Base query for PHYSLITE (calibrated, lightweight)
base = FuncADLQueryPHYSLITE()

# Build a flat dictionary per event with the needed collections.
# Jet kinematics are pre‑filtered to pt > 30 GeV.
# Electrons & muons are pre‑filtered to pt > 10 GeV.
query = (
    base.Select(
        lambda e: {
            # Jets (pt > 30 GeV)
            "jet_pt":  e.Jets().Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.pt() / 1000.0),
            "jet_eta": e.Jets().Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.eta()),
            "jet_phi": e.Jets().Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.phi()),

            # Electrons (pt > 10 GeV)
            "ele_pt":  e.Electrons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.pt() / 1000.0),
            "ele_eta": e.Electrons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.eta()),
            "ele_phi": e.Electrons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.phi()),

            # Muons (pt > 10 GeV)
            "mu_pt":   e.Muons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.pt() / 1000.0),
            "mu_eta":  e.Muons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.eta()),
            "mu_phi":  e.Muons().Where(lambda l: l.pt() / 1000.0 > 10.0).Select(lambda l: l.phi()),
        }
    )
)

# ------------------------------------------------------------
# 2. Run the query via ServiceX
# ------------------------------------------------------------
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad."
    "deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"
)

# For a quick test you can start with a few files; remove NFiles for the full dataset.
samples = ServiceXSpec(
    Sample=[
        Sample(
            Name="jet_ht",
            Dataset=dataset.Rucio(ds_name),
            NFiles=5,               # <-- adjust or remove for full analysis
            Query=query,
        )
    ]
)

# Deliver and convert to Awkward arrays
result = to_awk(deliver(samples))

# Extract the event‑wise dictionary
evts = result["jet_ht"]

# ------------------------------------------------------------
# 3. Build vector objects & apply ΔR cleaning
# ------------------------------------------------------------
# 3a. Combine electrons & muons into a single “light lepton” collection
lep_pt   = ak.concatenate([evts.ele_pt, evts.mu_pt],   axis=1)
lep_eta  = ak.concatenate([evts.ele_eta, evts.mu_eta], axis=1)
lep_phi  = ak.concatenate([evts.ele_phi, evts.mu_phi], axis=1)

# 3b. Build vector records (mass = 0 for both jets and leptons)
jets_vec = ak.zip(
    {
        "pt":   evts.jet_pt,
        "eta":  evts.jet_eta,
        "phi":  evts.jet_phi,
        "mass": ak.zeros_like(evts.jet_pt),
    },
    with_name="Momentum4D",
)

leps_vec = ak.zip(
    {
        "pt":   lep_pt,
        "eta":  lep_eta,
        "phi":  lep_phi,
        "mass": ak.zeros_like(lep_pt),
    },
    with_name="Momentum4D",
)

# 3c. Compute ΔR between each jet and each lepton in the same event
#    Result: for each jet an array of ΔR values to all leptons
dr = jets_vec.deltaR(leps_vec)   # shape: (event, jet, lepton)

# 3d. Determine which jets are “close” to any lepton (ΔR < 0.4)
close_mask = ak.any(dr < 0.4, axis=2)   # boolean per jet (event, jet)

# 3e. Keep only jets that are NOT close
good_jet_pt = evts.jet_pt[~close_mask]

# ------------------------------------------------------------
# 4. Compute scalar sum of selected jet pT per event (H_T)
# ------------------------------------------------------------
ht = ak.sum(good_jet_pt, axis=1)   # H_T in GeV, one value per event

# ------------------------------------------------------------
# 5. Histogram & Plot
# ------------------------------------------------------------
# Define a 1‑D histogram: 0–500 GeV, 50 uniform bins
h_ht = (
    Hist.new.Reg(50, 0, 500, name="ht", label=r"$H_T$ [GeV]")
        .Weight()
)

# Fill: each event contributes weight = 1
h_ht.fill(ht=ak.to_numpy(ht), weight=1.0)

# Plot with ATLAS style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(6, 5))
h_ht.plot(ax=ax, histtype="fill", edgecolor="black", linewidth=1, label=r"$H_T$ (jets $p_T>30$ GeV, $\Delta R>0.4$ from leptons)")
ax.set_ylabel("Events")
ax.legend()
plt.tight_layout()
plt.savefig("jet_ht.png")
plt.close()

print("Finished! Plot saved as 'jet_ht.png'.")
```


## Model: openai/gpt-oss-20b

**What the code does**

- Builds a ServiceX query that pulls jets, electrons and muons from the `mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697` dataset.  
- Uses the vector library to compute ΔR between each jet and every light lepton (e⁺/e⁻ and μ⁺/μ⁻).  
- Keeps jets with `pT > 30 GeV` that are farther than 0.4 in ΔR from *any* light lepton with `pT > 10 GeV`.  
- Calculates the scalar sum of the surviving jet pT for each event.  
- Fills a `scikit‑HEP hist` histogram with that sum, plots it with the ATLAS style and writes the figure to `jet_scalar_sum.png`.

```python
# --------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------
import awkward as ak
import vector
import hist
import matplotlib.pyplot as plt
import mplhep as hep

from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.electron_v1 import Electron_v1
from func_adl_servicex_xaodr25.xAOD.muon_v1 import Muon_v1

# --------------------------------------------------------------
# 2. Register Vector behaviours – required for .deltaR, .mass etc.
# --------------------------------------------------------------
vector.register_awkward()

# --------------------------------------------------------------
# 3


## Model: Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8

- I'll create a plot showing the scalar sum of jet pT in each event, where jets have pT > 30 GeV and are not within ΔR < 0.4 of any light lepton with pT > 10 GeV
- I'm using the PHYSLITE dataset for ttbar all-hadronic events
- The analysis will:
  1. Select jets with pT > 30 GeV
  2. Select electrons and muons with pT > 10 GeV
  3. Remove jets that are within ΔR < 0.4 of any light lepton
  4. Calculate the scalar sum of the remaining jet pT per event
  5. Create a histogram of this scalar sum

```python
import awkward as ak
import vector
import numpy as np
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import hist
import matplotlib.pyplot as plt
import mplhep as hep

# Register vector behaviors for ΔR calculations
vector.register_awkward()

# Create the base query
base_query = FuncADLQueryPHYSLITE()

# Query to get jets, electrons, and muons with required selections
query = base_query.Select(lambda e: {
    'jets': e.Jets('AntiKt4EMTopoJets'),
    'electrons': e.Electrons('Electrons'),
    'muons': e.Muons('Muons')
}).Select(lambda event: {
    # Select jets with pT > 30 GeV
    'selected_jets_pt': event['jets'].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.pt() / 1000.0),
    'selected_jets_eta': event['jets'].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.eta()),
    'selected_jets_phi': event['jets'].Where(lambda j: j.pt() / 1000.0 > 30.0).Select(lambda j: j.phi()),
    
    # Select electrons with pT > 10 GeV
    'selected_electrons_pt': event['electrons'].Where(lambda e: e.pt() / 1000.0 > 10.0).Select(lambda e: e.pt() / 1000.0),
    'selected_electrons_eta': event['electrons'].Where(lambda e: e.pt() / 1000.0 > 10.0).Select(lambda e: e.eta()),
    'selected_electrons_phi': event['electrons'].Where(lambda e: e.pt() / 1000.0 > 10.0).Select(lambda e: e.phi()),
    
    # Select muons with pT > 10 GeV
    'selected_muons_pt': event['muons'].Where(lambda m: m.pt() / 1000.0 > 10.0).Select(lambda m: m.pt() / 1000.0),
    'selected_muons_eta': event['muons'].Where(lambda m: m.pt() / 1000.0 > 10.0).Select(lambda m: m.eta()),
    'selected_muons_phi': event['muons'].Where(lambda m: m.pt() / 1000.0 > 10.0).Select(lambda m: m.phi()),
})

# Dataset name
ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"

# Execute the query
result_dict = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_lepton_analysis",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Starting with 1 file for testing
                    Query=query,
                )
            ]
        ),
    )
)

# Get the data
data = result_dict["jet_lepton_analysis"]

# Convert to awkward arrays with vector behaviors for physics calculations
jets = ak.zip({
    "pt": data.selected_jets_pt,
    "eta": data.selected_jets_eta,
    "phi": data.selected_jets_phi
}, with_name="Momentum2D")

electrons = ak.zip({
    "pt": data.selected_electrons_pt,
    "eta": data.selected_electrons_eta,
    "phi": data.selected_electrons_phi
}, with_name="Momentum2D")

muons = ak.zip({
    "pt": data.selected_muons_pt,
    "eta": data.selected_muons_eta,
    "phi": data.selected_muons_phi
}, with_name="Momentum2D")

# Combine leptons into a single collection
leptons = ak.concatenate([electrons, muons], axis=1)

# Calculate deltaR between jets and leptons
# Create cartesian product to get all jet-lepton pairs
jet_lepton_pairs = ak.cartesian([jets, leptons], axis=1)
jet_pairs, lepton_pairs = ak.unzip(jet_lepton_pairs)

# Calculate deltaR for all jet-lepton pairs
delta_r = jet_pairs.deltaR(lepton_pairs)

# Find jets that are NOT within 0.4 of any lepton
# For each jet, check if all deltaR values are > 0.4
is_isolated = ak.all(delta_r > 0.4, axis=2)

# Apply isolation cut to jet pT
isolated_jet_pt = data.selected_jets_pt[is_isolated]

# Calculate scalar sum of jet pT for each event
scalar_sum_pt = ak.sum(isolated_jet_pt, axis=1)

# Remove None values (events with no jets)
scalar_sum_pt = scalar_sum_pt[~ak.is_none(scalar_sum_pt)]

# Create histogram
h = hist.Hist.new.Reg(50, 0, 1000, name="scalar_sum", label="Scalar sum of jet pT [GeV]").Double()
h.fill(scalar_sum=scalar_sum_pt)

# Plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(10, 6))
h.plot(ax=ax, histtype="fill", linewidth=1, edgecolor="gray")
ax.set_xlabel("Scalar sum of jet pT [GeV]")
ax.set_ylabel("Events")
ax.set_title("Scalar sum of jet pT (jets with pT > 30 GeV, isolated from leptons with ΔR > 0.4)")
plt.savefig("scalar_sum_pt.png")
plt.close()
```


## Model: arcee-ai/coder-large

- Constructed a query to fetch jets and leptons from the specified dataset.
- Filtered jets to include only those with pT > 30 GeV.
- Filtered leptons to include only those with pT > 10 GeV.
- Calculated ΔR between each jet and each lepton.
- Excluded jets that are within 0.4 in ΔR of any lepton.
- Calculated the scalar sum of pT for the remaining jets in each event.
- Created a histogram of the scalar sum of pT.
- Saved the histogram to a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import vector
import hist
import matplotlib.pyplot as plt
import mplhep as hep

# Register vector behaviors
vector.register_awkward()

# The base query should run against PHYSLITE.
base_query = FuncADLQueryPHYSLITE()

# Query: get jets and leptons
query = (base_query
    .Select(lambda e: (e.Jets(), e.Electrons(), e.Muons()))
    .Select(lambda collections: {
        "jets": collections[0].Select(lambda j: {
            "pt": j.pt() / 1000.0,
            "eta": j.eta(),
            "phi": j.phi()
        }),
        "electrons": collections[1].Select(lambda el: {
            "pt": el.pt() / 1000.0,
            "eta": el.eta(),
            "phi": el.phi()
        }),
        "muons": collections[2].Select(lambda mu: {
            "pt": mu.pt() / 1000.0,
            "eta": mu.eta(),
            "phi": mu.phi()
        })
    })
)

# Do the fetch
# Define the dataset
ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"

data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_lepton_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        ),
    )
)

# Extract data
jets = data["jet_lepton_fetch"].jets
electrons = data["jet_lepton_fetch"].electrons
muons = data["jet_lepton_fetch"].muons

# Combine electrons and muons into a single lepton array
leptons = ak.concatenate([electrons, muons], axis=1)

# Filter jets with pt > 30 GeV
jets = jets[jets.pt > 30]

# Filter leptons with pt > 10 GeV
leptons = leptons[leptons.pt > 10]

# Create Lorentz vectors for jets and leptons
jet_vecs = ak.zip({
    "pt": jets.pt,
    "eta": jets.eta,
    "phi": jets.phi,
    "mass": 0.0
}, with_name="Momentum4D")

lepton_vecs = ak.zip({
    "pt": leptons.pt,
    "eta": leptons.eta,
    "phi": leptons.phi,
    "mass": 0.0
}, with_name="Momentum4D")

# Calculate ΔR between each jet and each lepton
dR = jet_vecs.cross(lepton_vecs, nested=True).deltaR()

# Find jets that are not within 0.4 in ΔR of any lepton
mask = ak.all(dR > 0.4, axis=-1)

# Apply mask to jets
filtered_jets = jets[mask]

# Calculate the scalar sum of pT for the remaining jets in each event
scalar_sum_pt = ak.sum(filtered_jets.pt, axis=1)

# Create a histogram of the scalar sum of pT
h = (
    hist.Hist.new.Reg(50, 0, 200, name="scalar_sum_pt", label="Scalar Sum of Jet pT [GeV]")
    .Int64()
)

# Fill the histogram
h.fill(scalar_sum_pt=scalar_sum_pt)

# Plot the histogram
plt.style.use(hep.style.ATLAS)
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="Scalar Sum of Jet pT")
plt.legend()
plt.xlabel("Scalar Sum of Jet pT [GeV]")
plt.ylabel("Events")
plt.savefig("scalar_sum_pt.png")
```
>>end-reply<
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 17.17 | 11670 | 1589 | 13259 | 0.0361 |
| gpt-5 | 91.66 | 11669 | 4929 | 16598 | 0.0639 |
| gpt-5-mini | 43.83 | 11669 | 5208 | 16877 | 0.0133 |
| gpt-5-nano | 65.36 | 11669 | 16627 | 28296 | 0.0072 |
| gpt-4o | 11.30 | 11670 | 836 | 12506 | 0.0375 |
| o4-mini | 31.99 | 11669 | 3964 | 15633 | 0.0303 |
| openai/gpt-oss-120b | 50.26 | 11734 | 3398 | 15132 | 0.0038 |
| openai/gpt-oss-20b | 11.32 | 11734 | 2048 | 13782 | 0.0010 |
| Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 | 21.70 | 11931 | 1522 | 13453 | 0.0269 |
| arcee-ai/coder-large | 13.49 | 11952 | 1057 | 13009 | 0.0068 |
