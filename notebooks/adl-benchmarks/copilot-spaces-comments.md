# Some further comments and notes

## Question 6

This is the first question that generated real trouble.

### GPT-4.1

This could never get it even close - we had to move to a thinking model.

```python
# This script fetches events from the specified rucio dataset with at least three jets,
# finds the trijet (combination of three jets) with invariant mass closest to 172.5 GeV,
# and for each event plots:
#   (1) the pT of the trijet four-momentum
#   (2) the maximum b-tagging discriminant value among the jets in the trijet
#
# Output: Two 1D histograms saved as PNG files.

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import vector
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

vector.register_awkward()
plt.style.use(hep.style.ATLAS)

# Dataset name (from user prompt)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"
)

# Query jets: get jets' kinematics and btag discriminant per event
base_query = FuncADLQueryPHYSLITE()
query = (
    base_query
    .Where(lambda e: e.Jets().Count() >= 3)
    .Select(lambda e: {
        "jet_pt":  e.Jets().Select(lambda j: j.pt()/1000.0),
        "jet_eta": e.Jets().Select(lambda j: j.eta()),
        "jet_phi": e.Jets().Select(lambda j: j.phi()),
        "jet_mass":e.Jets().Select(lambda j: j.m()/1000.0),
        "jet_btag":e.Jets().Select(lambda j: j.btagNN()),
    })
)

# Fetch the data (use NFiles=1 for speed; increase for more statistics)
awk_data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="trijet_selection",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        ),
    )
)

jets = awk_data["trijet_selection"]
n_events = len(jets.jet_pt)

# Build jets as Lorentz vectors per event
jet_vecs = ak.zip({
    "pt":   jets.jet_pt,
    "eta":  jets.jet_eta,
    "phi":  jets.jet_phi,
    "mass": jets.jet_mass,
}, with_name="Momentum4D")

jet_btag = jets.jet_btag

# For each event, find all trijet combinations
trijet_combos = ak.combinations(jet_vecs, 3, fields=["j1", "j2", "j3"], axis=1)
trijet_btags = ak.combinations(jet_btag, 3, fields=["j1", "j2", "j3"], axis=1)

# Combine vectors and b-tag values in parallel
trijet_vecs = trijet_combos.j1 + trijet_combos.j2 + trijet_combos.j3
trijet_masses = trijet_vecs.mass
trijet_pts = trijet_vecs.pt

# Find index of trijet whose invariant mass is closest to 172.5 GeV in each event
target_mass = 172.5
mass_diffs = np.abs(trijet_masses - target_mass)
min_idx = ak.argmin(mass_diffs, axis=1, keepdims=True)

# Select in each event the best trijet (closest mass to top quark mass)
best_trijet_pt = ak.flatten(trijet_pts[min_idx])
best_trijet_btag = ak.flatten(
    ak.max(
        ak.zip({
            "b1": trijet_btags.j1,
            "b2": trijet_btags.j2,
            "b3": trijet_btags.j3,
        })[min_idx], axis=2  # get max over three jets
    )
)

# Remove events with no trijet found (should not occur, but just in case)
mask_valid = ~ak.is_none(best_trijet_pt)
pt_values = best_trijet_pt[mask_valid]
btag_values = best_trijet_btag[mask_valid]

# Plot histogram for trijet pT
h_pt = (
    Hist.new
    .Reg(40, 0, 500, name="trijet_pt", label="Trijet $p_T$ [GeV]")
    .Double()
)
h_pt.fill(trijet_pt=ak.to_numpy(pt_values))

plt.figure(figsize=(8,6))
h_pt.plot(histtype="stepfilled", linewidth=2, alpha=0.7, color="C0")
plt.xlabel("Trijet $p_T$ [GeV]")
plt.ylabel("Events")
plt.title("Trijet $p_T$ (mass closest to 172.5 GeV)")
plt.tight_layout()
plt.savefig("trijet_pt.png")

# Plot histogram for max b-tag discriminant in trijet
h_btag = (
    Hist.new
    .Reg(40, -10, 15, name="trijet_max_btag", label="Max trijet b-tag discriminant")
    .Double()
)
h_btag.fill(trijet_max_btag=ak.to_numpy(btag_values))

plt.figure(figsize=(8,6))
h_btag.plot(histtype="stepfilled", linewidth=2, alpha=0.7, color="C1")
plt.xlabel("Max b-tag discriminant in trijet")
plt.ylabel("Events")
plt.title("Max b-tag discriminant (trijet closest to 172.5 GeV)")
plt.tight_layout()
plt.savefig("trijet_max_btag.png")
```

Its main failure is that it didn't copy in any of the tool code - totally ignoring those instructions. There may be other errors further down, but I didn't track them.

### OpenAI's `o4-mini`

This is what I did the most work with. Here is what I ended up with - with some minor editing by me.

```python
#!/usr/bin/env python3
"""
Fetch jets from a PHYSLITE ttbar all-had dataset via ServiceX,
build all trijet combinations per event, select the one whose
invariant mass is closest to 172.5 GeV, and then plot:

  - the trijet pT
  - the maximum b-tagging discriminant among the three jets

as a 2D histogram saved to a PNG file.
"""

# ---- ServiceX and xAOD tool imports ----
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# xAOD tool injection boilerplate (must be copied exactly)
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar
from func_adl import ObjectStream, func_adl_callable
import ast

import numpy as np

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

# For b-tagging we need the jet type
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1

# ---- Build the ServiceX query ----
# Base on PHYSLITE, filter events with >=3 jets, then retrieve jet kinematics and b-tag weight
base = FuncADLQueryPHYSLITE()

# Inject a BTaggingSelectionTool (FixedCutBEff_77) to compute the tag weight (discriminant)
base, btag_tool = make_a_tool(
    base,
    tool_name="btag_tool",
    tool_type="BTaggingSelectionTool",
    include_files=["xAODBTaggingEfficiency/BTaggingSelectionTool.h"],
    init_lines=[
        'ANA_CHECK(asg::setProperty({tool_name}, "OperatingPoint", "FixedCutBEff_77"));',
        "ANA_CHECK({tool_name}->initialize());",
    ],
)

# Accessor to get the b-tagging discriminant (GNN weight between -10 and 15)
tag_weight = make_tool_accessor(
    btag_tool,
    function_name="tag_weight",
    source_code=["ANA_CHECK({tool_name}->getTaggerWeight(*jet, result, false));"],
    arguments=[("jet", Jet_v1)],
    return_type_cpp="double",
    return_type_python="float",
)

# Name of the Rucio dataset
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV."
    "601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv."
    "DAOD_PHYSLITE.e8514_s4369_r16083_p6697"
)

# Build the final query: filter events, then pull out lists of jet pt, eta, phi, mass, and b-tag
query = (
    base
    .Where(lambda e: e.Jets().Count() >= 3)
    .Select(lambda e: {
        "pt":   e.Jets().Select(lambda j: j.pt() / 1000.0),
        "eta":  e.Jets().Select(lambda j: j.eta()),
        "phi":  e.Jets().Select(lambda j: j.phi()),
        "mass": e.Jets().Select(lambda j: j.m() / 1000.0),
        "btag": e.Jets().Select(lambda j: tag_weight(j)),
    })
)

# Fetch just 1 file for testing; remove NFiles for full dataset
samples = [
    Sample(
        Name="jets",
        Dataset=dataset.Rucio(ds_name),
        NFiles=1,
        Query=query,
    )
]

# Execute and convert to Awkward arrays
out = to_awk(deliver(ServiceXSpec(Sample=samples)))
jets = out["jets"]

# ---- Physics & array manipulation ----
import awkward as ak
import vector

# enable Vector behaviors
vector.register_awkward()

# Build an Awkward array of Lorentz vectors for each jet
jet_vecs = ak.zip(
    {
        "pt":  jets["pt"],
        "eta": jets["eta"],
        "phi": jets["phi"],
        "mass": jets["mass"],
    },
    with_name="Momentum4D",
)

# Combine kinematics and btag into one record array
jets_struct = ak.zip({"vec": jet_vecs, "btag": jets["btag"]})

# Build all 3-jet combinations per event
trijets = ak.combinations(jets_struct, 3, fields=["j1", "j2", "j3"], axis=1)

# Sum the 4-vectors to get each trijet's 4-momentum
p4 = trijets.j1.vec + trijets.j2.vec + trijets.j3.vec

# Compute invariant mass and pT for each trijet
m = p4.mass
pt = p4.pt

# Find the trijet whose mass is closest to 172.5 GeV
delta = abs(m - 172.5)
# get index of the best combination per event
idx = ak.argmin(delta, axis=1, keepdims=True)

# Select the best trijet pT and b-tag value, then flatten to 1D
best_pt = ak.flatten(pt[idx])
best_trijets = ak.flatten(trijets[idx])

# Compute max b-tag discriminant per trijet
bvals = np.stack(
    [best_trijets.j1.btag, best_trijets.j2.btag, best_trijets.j3.btag], axis=-1
)
max_b = ak.max(bvals, axis=-1)

best_btag = max_b

# ---- Histogramming and plotting ----
import numpy as np
import hist
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

# 2D histogram: trijet pT vs max b-tag discriminant
h_pt = (
    hist.Hist.new
       .Reg(100, 0, 1500, name="trijet_pt", label="Trijet $p_T$ [GeV]")
       .Int64()
)
h_b = hist.Hist.new.Reg(
    100, -10, 15, name="max_btag", label="Max b-tag discriminant"
).Int64()

h_pt.fill(trijet_pt=best_pt)
h_b.fill(max_btag=best_btag)

plt.style.use(hep.style.ATLAS)

plt.figure()
h_pt.plot(histtype="step", linewidth=1, label="Closest-mass trijet")
plt.xlabel("Trijet $p_T$ [GeV]")
plt.ylabel("Events")
plt.legend()
plt.savefig("trijet_pt.png")

plt.figure()
h_b.plot(histtype="step", linewidth=1, label="Max b-tag score")
plt.xlabel("Max b-tag discriminant")
plt.ylabel("Events")
plt.legend()
plt.savefig("max_btag_discriminant.png")
```

### Claude Sonnet 4

```python
import numpy as np
import awkward as ak
import vector
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar
from func_adl import ObjectStream
from func_adl import func_adl_callable
import ast

# Set up vector library for 4-momentum calculations
vector.register_awkward()

# Tool setup code for b-tagging
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
    """
    Injects C++ code into the query to initialize a tool of the specified type.
    """
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
    """
    Creates a Python-callable accessor for a C++ tool in the func_adl query.
    """
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
        """
        NOTE: This is a dummy function that injects C++ into the object stream to do the
        actual work.
        """
        ...
    tool_call.__name__ = function_name
    tool_call.__annotations__['return'] = eval(return_type_python)
    
    return func_adl_callable(tool_callback)(tool_call)

# Import for jet type
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1

# Create base query
base_query = FuncADLQueryPHYSLITE()

# Set up b-tagging tool (using FixedCutBEff_77 working point)
query_with_tool, btag_tool_info = make_a_tool(
    base_query,
    "btag_discriminator", 
    "BTaggingSelectionTool",
    include_files=["xAODBTaggingEfficiency/BTaggingSelectionTool.h"],
    init_lines=[
        'ANA_CHECK(asg::setProperty({tool_name}, "OperatingPoint", "FixedCutBEff_77"));',
        "ANA_CHECK({tool_name}->initialize());",
    ],
)

# Create accessor for b-tagging weight
tag_weight = make_tool_accessor(
    btag_tool_info,
    function_name="tag_weight",
    source_code=["ANA_CHECK({tool_name}->getTaggerWeight(*jet, result, false));"],
    arguments=[("jet", Jet_v1)],
    return_type_cpp="double", 
    return_type_python="float",
)

# Build the query to get jet data with b-tagging info
jet_query = (query_with_tool
    .Where(lambda e: e.Jets().Count() >= 3)  # Events with at least 3 jets
    .Select(lambda e: {
        "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),      # Convert to GeV
        "jet_eta": e.Jets().Select(lambda j: j.eta()),
        "jet_phi": e.Jets().Select(lambda j: j.phi()),
        "jet_m": e.Jets().Select(lambda j: j.m() / 1000.0),        # Convert to GeV
        "jet_btag": e.Jets().Select(lambda j: tag_weight(j)),      # B-tagging discriminant
    })
)

# Execute the query
print("Fetching data from ServiceX...")
ds_name = "mc23_13p6TeV:mc23_13p6TeV.601237.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697"

data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="trijet_analysis",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Starting with 1 file for testing
                    Query=jet_query,
                )
            ]
        ),
    )
)

jets_data = data["trijet_analysis"]
print(f"Fetched {len(jets_data)} events")

# Create Lorentz vectors for jets
jets = ak.zip({
    "pt": jets_data.jet_pt,
    "eta": jets_data.jet_eta, 
    "phi": jets_data.jet_phi,
    "mass": jets_data.jet_m
}, with_name="Momentum4D")

# Get b-tagging discriminants
btag_scores = jets_data.jet_btag

print("Finding trijet combinations closest to 172.5 GeV...")

# Generate all 3-jet combinations per event
trijet_combinations = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
btag_combinations = ak.combinations(btag_scores, 3, fields=["b1", "b2", "b3"])

# Calculate trijet invariant masses
trijet_4momentum = trijet_combinations.j1 + trijet_combinations.j2 + trijet_combinations.j3
trijet_masses = trijet_4momentum.mass

# Find the combination closest to 172.5 GeV (top quark mass) in each event
target_mass = 172.5
mass_differences = ak.abs(trijet_masses - target_mass)
closest_indices = ak.argmin(mass_differences, axis=1, keepdims=True)

# Extract the closest trijet properties
closest_trijet_pt = ak.flatten(trijet_4momentum.pt[closest_indices])
closest_trijet_mass = ak.flatten(trijet_masses[closest_indices]) 

# Get the maximum b-tagging score among the three jets in the closest trijet
closest_btag_b1 = ak.flatten(btag_combinations.b1[closest_indices])
closest_btag_b2 = ak.flatten(btag_combinations.b2[closest_indices])
closest_btag_b3 = ak.flatten(btag_combinations.b3[closest_indices])

# Find maximum b-tag score among the three jets
max_btag_scores = ak.max(ak.concatenate([
    closest_btag_b1[:, np.newaxis],
    closest_btag_b2[:, np.newaxis], 
    closest_btag_b3[:, np.newaxis]
], axis=1), axis=1)

print(f"Found {len(closest_trijet_pt)} events with valid trijet combinations")
print(f"Mean trijet pT: {ak.mean(closest_trijet_pt):.1f} GeV")
print(f"Mean trijet mass: {ak.mean(closest_trijet_mass):.1f} GeV (target: 172.5 GeV)")
print(f"Mean max b-tag score: {ak.mean(max_btag_scores):.3f}")

# Create histograms
print("Creating histograms...")

# Histogram for trijet pT
pt_hist = (
    Hist.new.Reg(50, 0, 800, name="pt", label="Trijet pT [GeV]")
    .Weight()
)
pt_hist.fill(pt=ak.to_numpy(closest_trijet_pt), weight=1.0)

# Histogram for maximum b-tagging discriminant
btag_hist = (
    Hist.new.Reg(50, -10, 15, name="btag", label="Max B-tagging Discriminant")
    .Weight()
)
btag_hist.fill(btag=ak.to_numpy(max_btag_scores), weight=1.0)

# Plot the results
plt.style.use(hep.style.ATLAS)

# Plot 1: Trijet pT
fig, ax = plt.subplots(figsize=(10, 8))
pt_hist.plot(histtype="fill", linewidth=1, edgecolor="black", alpha=0.7, color="skyblue")
plt.xlabel("Trijet $p_T$ [GeV]")
plt.ylabel("Events")
plt.title("$p_T$ of Trijet System Closest to 172.5 GeV\n(ttbar all-hadronic events)")
plt.grid(True, alpha=0.3)
hep.atlas.label(loc=0, data=False)
plt.tight_layout()
plt.savefig("trijet_pt.png", dpi=300, bbox_inches='tight')
print("Saved trijet pT plot to: trijet_pt.png")
plt.close()

# Plot 2: Maximum b-tagging discriminant
fig, ax = plt.subplots(figsize=(10, 8))
btag_hist.plot(histtype="fill", linewidth=1, edgecolor="black", alpha=0.7, color="lightcoral")
plt.xlabel("Maximum B-tagging Discriminant (FixedCutBEff_77)")
plt.ylabel("Events")
plt.title("Maximum B-tagging Score Among Jets in Trijet System\n(Closest to 172.5 GeV, ttbar all-hadronic events)")
plt.grid(True, alpha=0.3)
hep.atlas.label(loc=0, data=False)
plt.tight_layout()
plt.savefig("max_btag_discriminant.png", dpi=300, bbox_inches='tight')
print("Saved maximum b-tagging discriminant plot to: max_btag_discriminant.png")
plt.close()

print("\nAnalysis complete!")
print(f"Processed {len(closest_trijet_pt)} events with ≥3 jets")
print("Generated plots:")
print("- trijet_pt.png: pT distribution of trijet systems closest to 172.5 GeV")
print("- max_btag_discriminant.png: Maximum b-tagging discriminant among jets in selected trijets")
```

* `ak.abs` should be `abs`.

Wow - that worked!!

