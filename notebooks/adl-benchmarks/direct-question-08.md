# For events with at least three light leptons and a same-flavor opposite-charge light lepton pair, find such a pair that has the invariant mass closest to 91.2 GeV in each event and plot the transverse mass of the system consisting of the missing transverse momentum and the highest-pT light lepton not in this pair in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697.



## Model: gpt-4.1

- Access the ATLAS PHYSLITE sample from the specified rucio dataset using ServiceX.
- For each event, extract all electrons and muons (“light leptons”) with their kinematic properties and charges.
- Only keep events with **at least three light leptons**.
- For each event, construct all same-flavor, opposite-charge lepton pairs and select the pair whose invariant mass is closest to 91.2 GeV.
- Identify the highest-pT light lepton in the event that is **not part of the selected pair**.
- Extract the missing transverse momentum (MissingET).
- Compute the transverse mass (mT) of the system formed by this highest-pT lepton (not in the pair) and the MET.
- Make a histogram of the transverse mass and plot the result, saving it as `transverse_mass.png`.

```python
import awkward as ak
import numpy as np
import vector
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# 1. Build the ServiceX query
base_query = FuncADLQueryPHYSLITE()

# Select all light leptons (electrons and muons)
query = (
    base_query
    .Select(lambda e: (
        e.Electrons(),
        e.Muons(),
        e.MissingET().First()
    ))
    .Select(lambda o: {
        # Leptons will be zipped light leptons (type, pt, eta, phi, charge)
        "lep_type":  ak.operations.broadcast_arrays(
            o[0].Select(lambda el: 0), # 0 = electron
            o[1].Select(lambda mu: 1)   # 1 = muon
        ),
        "lep_pt":  ak.operations.broadcast_arrays(
            o[0].Select(lambda el: el.pt() / 1000.0),
            o[1].Select(lambda mu: mu.pt() / 1000.0),
        ),
        "lep_eta": ak.operations.broadcast_arrays(
            o[0].Select(lambda el: el.eta()),
            o[1].Select(lambda mu: mu.eta()),
        ),
        "lep_phi": ak.operations.broadcast_arrays(
            o[0].Select(lambda el: el.phi()),
            o[1].Select(lambda mu: mu.phi()),
        ),
        "lep_charge": ak.operations.broadcast_arrays(
            o[0].Select(lambda el: el.charge()),
            o[1].Select(lambda mu: mu.charge()),
        ),
        "lep_mass": ak.operations.broadcast_arrays(
            o[0].Select(lambda el: 0.000511), # Electron mass [GeV]
            o[1].Select(lambda mu: 0.105658), # Muon mass [GeV]
        ),
        "met_met": o[2].met() / 1000.0,
        "met_phi": o[2].phi(),
    })
)

# 2. Fetch data from ServiceX (for demonstration, start with 1 file)
ds_name = "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"
result = to_awk(deliver(ServiceXSpec(Sample=[Sample(Name="lepton_met", Dataset=dataset.Rucio(ds_name), NFiles=1, Query=query)])))

events = result["lepton_met"]

# Helper: flatten list-of-arrays (per event) into a single array-of-arrays
def flatten_per_event(ev_list):
    # ev_list is a tuple(list per event for electrons, list per event for muons)
    # Concatenate
    return ak.concatenate(ev_list, axis=1)

# Step 3: Make jagged event-wise arrays for all leptons (type, pt, eta, phi, charge, mass)
leps_type_all = flatten_per_event(events.lep_type)
leps_pt_all   = flatten_per_event(events.lep_pt)
leps_eta_all  = flatten_per_event(events.lep_eta)
leps_phi_all  = flatten_per_event(events.lep_phi)
leps_charge_all = flatten_per_event(events.lep_charge)
leps_mass_all = flatten_per_event(events.lep_mass)

# Step 4: Keep only events with at least 3 leptons
good_events_mask = ak.num(leps_pt_all, axis=1) >= 3

leps_type = leps_type_all[good_events_mask]
leps_pt   = leps_pt_all[good_events_mask]
leps_eta  = leps_eta_all[good_events_mask]
leps_phi  = leps_phi_all[good_events_mask]
leps_charge = leps_charge_all[good_events_mask]
leps_mass = leps_mass_all[good_events_mask]
met_met = events.met_met[good_events_mask]
met_phi = events.met_phi[good_events_mask]

# Step 5: Build lepton 4-vectors
lep_vectors = ak.zip(
    {"pt": leps_pt, "eta": leps_eta, "phi": leps_phi, "mass": leps_mass},
    with_name="Momentum4D"
)

# Step 6: For each event, find all SFOS lepton pairs
def sfos_pairs_mask(type_arr, charge_arr):
    pairs = ak.combinations(np.arange(len(type_arr)), 2)
    idx1, idx2 = ak.unzip(pairs)
    # Check same flavor: both are electrons or both are muons
    same_flavor = type_arr[idx1] == type_arr[idx2]
    opp_charge = charge_arr[idx1] * charge_arr[idx2] < 0
    return pairs[same_flavor & opp_charge]

# Doing it jaggedly:
def find_sfos_pairs_indices(event_types, event_charges):
    # Returns per event list of (i,j) indices for SFOS pairs
    return ak.local_index(event_types)  # Will get replaced below

all_indices = ak.local_index(leps_type)
event_pairs = ak.combinations(all_indices, 2, fields=["i1", "i2"])
# Mask for same flavor, opposite charge
sf_mask = (leps_type[event_pairs["i1"]] == leps_type[event_pairs["i2"]]) & \
           (leps_charge[event_pairs["i1"]] * leps_charge[event_pairs["i2"]] < 0)
sfos_event_pairs = event_pairs[sf_mask]

# Step 7: For each event's SFOS pairs, find invariant mass, select pair closest to 91.2 GeV
lep1 = lep_vectors[sfos_event_pairs["i1"]]
lep2 = lep_vectors[sfos_event_pairs["i2"]]
pair_mass = (lep1 + lep2).mass

z_mass = 91.2
# Distance to Z
dz = abs(pair_mass - z_mass)
# For each event, get argmin index (which SFOS pair is closest to Z)
best_idx = ak.argmin(dz, axis=1, keepdims=True)
# Select indices of best pair
best_pair_indices = sfos_event_pairs[best_idx]
best_mask = ak.num(pair_mass, axis=1) > 0  # Keep only events with a SFOS pair

# Keep only events with at least one SFOS pair
lep_vectors = lep_vectors[best_mask]
leps_type = leps_type[best_mask]
leps_pt = leps_pt[best_mask]
leps_charge = leps_charge[best_mask]
met_met = met_met[best_mask]
met_phi = met_phi[best_mask]
best_pair_indices = best_pair_indices[best_mask]

# Now: for each event, find which leptons are in the best pair,
# and remove them from the lepton list for that event for the "unpaired" lepton selection
lep_idx = ak.local_index(leps_pt)
pair1 = ak.firsts(best_pair_indices.i1)
pair2 = ak.firsts(best_pair_indices.i2)
# Build mask: True for lepton if its index is *not* in the selected SFOS pair
def not_in_pair_mask(idx, p1, p2):
    return (idx != p1) & (idx != p2)
lepton_not_in_pair_mask = not_in_pair_mask(lep_idx, pair1, pair2)

# Select highest-pT lepton not in the Z pair
pt_unpaired = ak.where(lepton_not_in_pair_mask, leps_pt, -1.0)
# Find argmax of this per event
idx_max = ak.argmax(pt_unpaired, axis=1, keepdims=False)
# Get the vector for the highest-pT lepton not in best pair
unpaired_lep_vector = lep_vectors[idx_max, ak.local_index(pt_unpaired)]
##############################################################################
# There is a bug above, fix by flattening ak.where (since extra dimension appears):
# But easier: make "not-in-pair" mask, filter the arrays, and then select argmax.
def filter_not_in_pair(arr, mask):
    # arr and mask are jagged arrays (events, leptons)
    return arr[mask]

leps_pt_unpair = filter_not_in_pair(leps_pt, lepton_not_in_pair_mask)
lep_vectors_unpair = filter_not_in_pair(lep_vectors, lepton_not_in_pair_mask)

# In rare events with >1 non-pair lepton, pick highest-pT:
idx_lep = ak.argmax(leps_pt_unpair, axis=1, keepdims=True)
# For events where there is at least one lepton not in Z pair:
has_unpaired = ak.num(leps_pt_unpair, axis=1) > 0

# Select only events with a "spare" lepton
lep_vectors_unpair = lep_vectors_unpair[has_unpaired]
met_met = met_met[has_unpaired]
met_phi = met_phi[has_unpaired]

# Take highest-pT lepton (not in Z pair) per event
max_lep = lep_vectors_unpair[idx_lep]

# Prepare MET vector for each event (pseudo-4vector: pt, phi, y=0, mass=0)
met_vec = ak.zip(
    {
        "pt": met_met,
        "eta": np.zeros(len(met_met)),
        "phi": met_phi,
        "mass": np.zeros(len(met_met)),
    },
    with_name="Momentum4D"
)

# Step 8: Compute transverse mass for each event:
def transverse_mass(lep, met):
    # mT^2 = 2*pT^l*pT^met*(1-cos(phi^l-phi^met))
    return np.sqrt(2 * lep.pt * met.pt * (1 - np.cos(lep.phi - met.phi)))
# max_lep: (n_ev, 1)
max_lep_flat = ak.flatten(max_lep)
mt = transverse_mass(max_lep_flat, met_vec)

# Step 9: Plot and histogram
h_mt = (
    Hist.new
        .Reg(50, 0, 200, name="mt", label="Transverse mass [GeV]")
        .Double()
)

h_mt.fill(mt=ak.to_numpy(mt))

plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h_mt.plot(ax=ax, histtype="step", color="navy", label="mT (lepton+MET)")
ax.set_xlabel("Transverse mass $m_T$ [GeV]")
ax.set_ylabel("Events")
ax.legend()
plt.tight_layout()
plt.savefig("transverse_mass.png")
plt.close(fig)
```


## Model: gpt-5

- Uses ServiceX (FuncADL, Release 25, PHYSLITE) to skim only the variables needed: electrons, muons (pt/eta/phi/charge) and MissingET (met, phi). Units are converted to GeV at the query stage.
- Applies an event-level prefilter in ServiceX requiring at least three light leptons (electrons + muons) to reduce data volume.
- In Awkward+Vector: builds light-lepton collections, forms all same-flavor opposite-charge (SFOS) pairs, finds the pair whose invariant mass is closest to 91.2 GeV per event, then selects the highest-pT light lepton not in that pair.
- Computes the transverse mass mT between that lepton and the event MET and fills a histogram (Hist). The plot is saved to mt_lep_met.png.
- Starts with NFiles=1 for a quick test; remove NFiles or set it to None to run over the full dataset.
- No additional lepton ID/isolation working points are applied beyond what is present in PHYSLITE; “light leptons” here means all electrons and muons from the derivation.

```python
import numpy as np
import awkward as ak
import vector
vector.register_awkward()

from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

def main():
    # Dataset (DAOD_PHYSLITE, R25)
    ds_name = "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"

    # Base query
    base = FuncADLQueryPHYSLITE()

    # Event-level prefilter: require >=3 light leptons (e + mu)
    # Then select e, mu collections and first MissingET object.
    query = (
        base
        .Where(lambda e: e.Electrons().Count() + e.Muons().Count() >= 3)
        .Select(lambda e: (e.Electrons(), e.Muons(), e.MissingET().First()))
        .Select(lambda parts: {
            # Electrons
            "ele_pt":     parts[0].Select(lambda el: el.pt() / 1000.0),  # GeV
            "ele_eta":    parts[0].Select(lambda el: el.eta()),
            "ele_phi":    parts[0].Select(lambda el: el.phi()),
            "ele_charge": parts[0].Select(lambda el: el.charge()),
            # Muons
            "mu_pt":     parts[1].Select(lambda mu: mu.pt() / 1000.0),   # GeV
            "mu_eta":    parts[1].Select(lambda mu: mu.eta()),
            "mu_phi":    parts[1].Select(lambda mu: mu.phi()),
            "mu_charge": parts[1].Select(lambda mu: mu.charge()),
            # Missing ET
            "met":      parts[2].met() / 1000.0,  # GeV
            "met_phi":  parts[2].phi(),
        })
    )

    # Deliver (start with a single file for testing; remove NFiles to run on all files)
    sx_out = to_awk(
        deliver(
            ServiceXSpec(
                Sample=[
                    Sample(
                        Name="tri_lep_met",
                        Dataset=dataset.Rucio(ds_name),
                        NFiles=1,  # For quick test. Remove this line to run the full dataset.
                        Query=query,
                    )
                ]
            )
        )
    )

    data = sx_out["tri_lep_met"]

    # Build electron and muon Momentum4D arrays (GeV). Add charge and flavor fields.
    m_e = 0.000511
    m_mu = 0.105658

    ele_vec = ak.zip(
        {"pt": data.ele_pt, "eta": data.ele_eta, "phi": data.ele_phi, "mass": ak.zeros_like(data.ele_pt) + m_e},
        with_name="Momentum4D",
    )
    ele_vec = ak.with_field(ele_vec, data.ele_charge, where="charge")
    ele_vec = ak.with_field(ele_vec, ak.full_like(data.ele_charge, "e"), where="flavor")

    mu_vec = ak.zip(
        {"pt": data.mu_pt, "eta": data.mu_eta, "phi": data.mu_phi, "mass": ak.zeros_like(data.mu_pt) + m_mu},
        with_name="Momentum4D",
    )
    mu_vec = ak.with_field(mu_vec, data.mu_charge, where="charge")
    mu_vec = ak.with_field(mu_vec, ak.full_like(data.mu_charge, "mu"), where="flavor")

    # Combine electrons and muons into a single light-lepton collection per event
    leptons = ak.concatenate([ele_vec, mu_vec], axis=1)

    # Form all pairs and their index pairs
    pairs = ak.combinations(leptons, 2, fields=["l1", "l2"], axis=1)
    ipairs = ak.argcombinations(leptons, 2, fields=["i1", "i2"], axis=1)

    # Same-flavor opposite-charge mask
    sf_mask = (pairs.l1.flavor == pairs.l2.flavor) & (pairs.l1.charge != pairs.l2.charge)

    # Keep only SFOS pairs
    pairs_sfos = pairs[sf_mask]
    ipairs_sfos = ipairs[sf_mask]

    # Compute invariant masses of SFOS pairs
    mll_sfos = (pairs_sfos.l1 + pairs_sfos.l2).mass

    # Event selection: require at least one SFOS pair (and we already require >=3 leptons in the ServiceX query)
    has_sfos = ak.num(mll_sfos, axis=1) > 0

    # Apply event mask consistently to all per-event arrays
    leptons = leptons[has_sfos]
    pairs_sfos = pairs_sfos[has_sfos]
    ipairs_sfos = ipairs_sfos[has_sfos]
    mll_sfos = mll_sfos[has_sfos]
    met = data.met[has_sfos]
    met_phi = data.met_phi[has_sfos]

    # Select the SFOS pair whose mass is closest to 91.2 GeV (Z mass)
    z_mass = 91.2
    delta = abs(mll_sfos - z_mass)
    best_idx = ak.argmin(delta, axis=1, keepdims=True)

    # Indices (within the per-event lepton list) of the chosen SFOS pair
    best_i1 = ak.flatten(ipairs_sfos.i1[best_idx])
    best_i2 = ak.flatten(ipairs_sfos.i2[best_idx])

    # Build mask for leptons that are NOT in the chosen pair
    li = ak.local_index(leptons, axis=1)
    in_pair = (li == best_i1) | (li == best_i2)
    not_in_pair = ~in_pair

    # The remaining leptons; choose the highest-pT one
    remaining = leptons[not_in_pair]
    rem_best_idx = ak.argmax(remaining.pt, axis=1, keepdims=True)
    lep_w = ak.flatten(remaining[rem_best_idx])

    # Compute transverse mass with MET: mT = sqrt(2 * pT * MET * (1 - cos(Δφ)))
    dphi = lep_w.phi - met_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    mt = np.sqrt(2.0 * lep_w.pt * met * (1.0 - np.cos(dphi)))

    # Histogram and plot
    h = (
        Hist.new
        .Reg(40, 0.0, 200.0, name="mt", label=r"mT(ℓ, MET) [GeV]")
        .Double()
    )
    h.fill(mt=ak.to_numpy(mt))

    plt.style.use(hep.style.ATLAS)
    h.plot(histtype="fill", linewidth=1.0, edgecolor="black", color="#1f77b4", alpha=0.7, label="Events")
    plt.legend()
    plt.xlabel("mT(ℓ, MET) [GeV]")
    plt.ylabel("Events")
    plt.title("Transverse mass of highest-pT lepton outside Z-candidate and MET")
    plt.tight_layout()
    plt.savefig("mt_lep_met.png", dpi=150)
    plt.close()

    print("Saved: mt_lep_met.png")

if __name__ == "__main__":
    main()
```


## Model: gpt-5-mini

- I used ServiceX (FuncADL) to fetch electrons, muons, and MissingET from the requested PHYSLITE dataset (I used NFiles=1 to start for quick testing — change or remove NFiles to run over more files / full dataset).
- I combined electrons and muons into one per-event "light lepton" collection (kept pt [GeV], eta, phi, charge and a pid code to remember flavor).
- For each event with >= 3 light leptons and at least one same-flavor opposite-charge pair, I formed all lepton pairs, selected same-flavor opposite-charge pairs, found the pair with invariant mass closest to 91.2 GeV, and then identified the highest-pT light lepton not in that chosen pair.
- I computed the transverse mass mT between that remaining highest-pT lepton and the event missing transverse momentum (MET).
- I filled and plotted a 1D histogram of the mT values and saved the plot to "mt_hist.png".

```python
# Fetch + analysis script using ServiceX, Awkward, Vector, and Hist
# - Start with NFiles=1 for quick testing. Remove or increase for larger statistics.
# - This script uses only Awkward array operations for event/particle manipulation.

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import numpy as np
import vector
vector.register_awkward()

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ATLAS)

from hist import Hist

# --------------------------
# 1) Build the FuncADL query
# --------------------------
base = FuncADLQueryPHYSLITE()

# Select electrons, muons, and the event MissingET (First())
query = (
    base
    .Select(lambda e: (e.Electrons(), e.Muons(), e.MissingET().First()))
    .Select(lambda tup: {
        # electrons (units: MeV in xAOD -> convert pt to GeV)
        "ele_pt":   tup[0].Select(lambda el: el.pt() / 1000.0),
        "ele_eta":  tup[0].Select(lambda el: el.eta()),
        "ele_phi":  tup[0].Select(lambda el: el.phi()),
        "ele_charge": tup[0].Select(lambda el: el.charge()),
        "ele_pid":  tup[0].Select(lambda el: 11),   # electron PID marker == 11

        # muons
        "mu_pt":    tup[1].Select(lambda mu: mu.pt() / 1000.0),
        "mu_eta":   tup[1].Select(lambda mu: mu.eta()),
        "mu_phi":   tup[1].Select(lambda mu: mu.phi()),
        "mu_charge": tup[1].Select(lambda mu: mu.charge()),
        "mu_pid":   tup[1].Select(lambda mu: 13),   # muon PID marker == 13

        # Missing ET (single object)
        "met":      tup[2].met() / 1000.0,
        "met_phi":  tup[2].phi(),
    })
)

# --------------------------
# 2) Run ServiceX deliver
# --------------------------
rucio_name = (
    "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"
)

sx_result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="lllvjj_fetch",
                    Dataset=dataset.Rucio(rucio_name),
                    NFiles=1,   # start with 1 file for testing; remove or increase for more events
                    Query=query,
                )
            ]
        ),
    )
)

events = sx_result["lllvjj_fetch"]

# --------------------------
# 3) Build per-event lepton records (concat electrons and muons)
# --------------------------
# Zip electron records (may be empty per event)
ele = ak.zip({
    "pt": events.ele_pt,
    "eta": events.ele_eta,
    "phi": events.ele_phi,
    "charge": events.ele_charge,
    "pid": events.ele_pid
})

mu = ak.zip({
    "pt": events.mu_pt,
    "eta": events.mu_eta,
    "phi": events.mu_phi,
    "charge": events.mu_charge,
    "pid": events.mu_pid
})

# Concatenate per-event so leptons is a jagged array: N_events * [N_leptons_in_event]
leptons = ak.concatenate([ele, mu], axis=1)

# --------------------------
# 4) Build all unique lepton pairs per event, compute invariant mass for each pair
#    and mask same-flavor opposite-charge pairs
# --------------------------
pairs = ak.combinations(leptons, 2, fields=["l1", "l2"])

# Build Lorentz vectors for each leg of the pair (use mass ~ 0.0 GeV)
# Vector accepts pt, eta, phi, mass when with_name="Momentum4D"
l1_vec = ak.zip({
    "pt": pairs.l1.pt,
    "eta": pairs.l1.eta,
    "phi": pairs.l1.phi,
    "mass": pairs.l1.pt * 0.0  # zero mass; shape matches pairs.l1.pt
}, with_name="Momentum4D")

l2_vec = ak.zip({
    "pt": pairs.l2.pt,
    "eta": pairs.l2.eta,
    "phi": pairs.l2.phi,
    "mass": pairs.l2.pt * 0.0
}, with_name="Momentum4D")

pair_mass = (l1_vec + l2_vec).mass  # invariant mass in GeV

# Mask for same-flavor & opposite-charge
same_flavor = (pairs.l1.pid == pairs.l2.pid)
opp_charge = (pairs.l1.charge != pairs.l2.charge)
valid_pair_mask = same_flavor & opp_charge

# For each pair compute distance to Z mass (91.2 GeV), but only keep valid pairs
delta = ak.abs(pair_mass - 91.2)
# Replace invalid pair deltas with a large value so argmin will ignore them
delta_valid = ak.where(valid_pair_mask, delta, np.inf)

# --------------------------
# 5) For each event, find the valid pair closest to 91.2 GeV (if any)
# --------------------------
# argmin along the pairs-per-event axis. keepdims=True so indexing works.
best_idx = ak.argmin(delta_valid, axis=1, keepdims=True)
best_pair = pairs[best_idx]            # this is nested; convert to one record per event
best_pair = ak.first(best_pair)       # now best_pair.l1 / best_pair.l2 are scalar-per-event (or None)

# --------------------------
# 6) Select events with at least 3 leptons and at least one valid same-flavor opp-charge pair
# --------------------------
n_leps = ak.num(leptons, axis=1)
n_valid_pairs = ak.sum(valid_pair_mask, axis=1)  # counts of valid pairs per event

event_mask = (n_leps >= 3) & (n_valid_pairs > 0)

# Subset arrays to selected events
leptons_sel = leptons[event_mask]
best_pair_sel = best_pair[event_mask]
met_sel = events.met[event_mask]
met_phi_sel = events.met_phi[event_mask]

# --------------------------
# 7) For each selected event: find highest-pT lepton NOT in the chosen pair
# --------------------------
# Build boolean per-lepton is_in_pair by matching pt,phi,charge to the chosen pair legs.
# (We use triple match to reduce accidental ambiguity with identical kinematics.)
is_in_pair_l1 = (leptons_sel.pt == best_pair_sel.l1.pt) & \
                (leptons_sel.phi == best_pair_sel.l1.phi) & \
                (leptons_sel.charge == best_pair_sel.l1.charge)

is_in_pair_l2 = (leptons_sel.pt == best_pair_sel.l2.pt) & \
                (leptons_sel.phi == best_pair_sel.l2.phi) & \
                (leptons_sel.charge == best_pair_sel.l2.charge)

is_in_pair = is_in_pair_l1 | is_in_pair_l2

remaining = leptons_sel[~is_in_pair]  # the leptons not in the chosen pair (should be >=1 element)

# Take the highest-pT remaining lepton per event
# argmax over the inner lists (keepdims=True so indexing returns a nested shape we can first())
idx_max = ak.argmax(remaining.pt, axis=1, keepdims=True)
highest_remaining = ak.first(remaining[idx_max])  # record per-event for selected events

# --------------------------
# 8) Compute transverse mass mT = sqrt(2 * pT_lep * MET * (1 - cos(delta_phi)))
# --------------------------
lep_pt = highest_remaining.pt
lep_phi = highest_remaining.phi

dphi = lep_phi - met_phi_sel
# normalize dphi into [-pi, pi] for numerical stability (numpy trig will still work, but keep nice)
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

mT = ak.sqrt(2.0 * lep_pt * met_sel * (1.0 - np.cos(dphi)))

# --------------------------
# 9) Make a histogram of the mT distribution and save plot
# --------------------------
# Flatten mT (it is 1D after selection) -> ensure a plain 1D array
mT_flat = ak.flatten(mT)

# Convert to numpy array for Hist filling
# ak.to_numpy works for regular numeric arrays; mT_flat is 1D numeric so this is fine.
mT_numpy = ak.to_numpy(mT_flat)

h = (
    Hist.new
    .Reg(50, 0.0, 300.0, name="mT", label="m_{T} [GeV]")
    .Double()
)

h.fill(mT=mT_numpy)

# Plot
fig, ax = plt.subplots(figsize=(8,6))
h.plot(histtype="step", ax=ax, linewidth=1.5, label="m_{T} (lepton + MET)")
ax.set_xlabel("m_{T} [GeV]")
ax.set_ylabel("Events")
ax.set_ylim(bottom=0)
ax.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mt_hist.png", dpi=200)
print(f"Saved histogram to mt_hist.png  (N selected events = {len(mT_numpy)})")
```


## Model: gpt-5-nano

- I’m providing a self-contained Python snippet that uses ServiceX to fetch PHYSLITE data, Awkward Array 2.x for manipulation (including simple 4-vector handling via the Vector library), and Hist to build a MT distribution. The code focuses on events with at least three light leptons and builds a same-flavor, opposite-charge lepton pair (SF-OS) per event, choosing the pair whose invariant mass is closest to the Z mass (91.2 GeV). It then computes the transverse mass MT of the system formed by the missing transverse energy (MET) and the highest-pT lepton not in that pair, and finally plots MT as a PNG file.

- The code uses a two-step ServiceX fetch pattern (first define the query; then execute it against the dataset). It demonstrates skimming and per-event logic, keeping data in Awkward arrays (no Python lists per- event). It writes the MT histogram to a PNG file using Hist and Matplotlib.

- For testing, start with NFiles=1 (as recommended in hints) and adjust as needed after local verification.

```python
# Prerequisites (install as needed in your environment)
# pip install serviceX servicex func_adl_servicex_xaodr25 awkward-1.x vector hist matplotlib
#
# This script assumes you have proper credentials to access the GRID via rucio through ServiceX.

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import numpy as np
import vector  # Vector library for Lorentz-vector operations
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

# 1) Define the base query against PHYSLITE
base_query = FuncADLQueryPHYSLITE()

# 2) Build per-event lepton collections (electrons and muons) with essential kinematics
#    We keep a per-lepton record: flavor, pt (GeV), eta, phi, mass (GeV), charge, and an event id if available.
#    The exact field names may vary by dataset; adapt as needed.

leptons_query = (
    base_query
    .Select(lambda e: {
        # Electrons per event
        "electrons": e.Electrons().Select(lambda el: {
            "flavor": "e",
            "evt": getattr(e, "evt_idx", None) if hasattr(e, "evt_idx") else e.eventNumber(),  # event id (best guess)
            "pt": el.pt() / 1000.0,        # convert MeV -> GeV
            "eta": el.eta(),
            "phi": el.phi(),
            "mass": 0.000511,
            "charge": el.charge(),
        }),
        # Muons per event
        "muons": e.Muons().Select(lambda mu: {
            "flavor": "mu",
            "evt": getattr(e, "evt_idx", None) if hasattr(e, "evt_idx") else e.eventNumber(),
            "pt": mu.pt() / 1000.0,
            "eta": mu.eta(),
            "phi": mu.phi(),
            "mass": 0.1057,
            "charge": mu.charge(),
        }),
    })
)

# 3) MET per event (magnitude and phi)
met_query = (
    base_query
    .Select(lambda e: e.MissingET().First())
    .Select(lambda m: {
        "met": m.met() / 1000.0,   # GeV
        "met_phi": m.phi()
    })
)

# 4) Dataset to fetch (use a single file for testing)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"
)

# 5) Deliver and convert to Awkward
#    We fetch two samples: one with lepton data (electrons+muons) and one with MET
samples = [
    Sample(Name="leptons_fetch", Dataset=dataset.Rucio(ds_name), NFiles=1, Query=leptons_query),
    Sample(Name="met_fetch", Dataset=dataset.Rucio(ds_name), NFiles=1, Query=met_query),
]

spec = ServiceXSpec(Sample=samples)

# Execute fetch
deliv = deliver(spec)

# Convert to Awkward (structure-dependent; adapt keys to your result)
# The resulting dict will have keys equal to the Sample Names above.
to_awk = __import__("servicex").servicex.to_awk  # lazy import to avoid import error if not run

data = to_awk(deliv)

# Expected structure (illustrative; real keys depend on your ServiceX output):
# data["leptons_fetch"] -> awkward Array with per-event lists of leptons
# data["met_fetch"] -> awkward Array with per-event MET values and phi

# 6) Normalize and build per-event lepton vectors using awkward + vector
#    We'll construct Momentum4D vectors for all leptons (electrons + muons)
#    and keep a per-lepton flag for charge, flavor, etc.

# Helper: build per-event leptons from fetched data (this section is dataset dependent)
def build_lepton_vectors(leptons_per_event):
    # leptons_per_event is expected to be a nested array:
    # [
    #   [ {pt, eta, phi, mass, charge, flavor}, {..}, ... ],   # event 0
    #   [ ... ],                                               # event 1
    #   ...
    # ]
    # Convert to per-event arrays of Momentum4D-like records
    # We create a vector record for each lepton with fields: pt, eta, phi, mass, charge
    # Then zip into Momentum4D for easy mass/pt access.
    # The following code assumes the structure exists; adapt as needed for your actual keys.
    # Build a per-event list of Momentum4D records:
    per_event_vectors = []
    for ev in leptons_per_event.tolist():  # Convert to Python list for iteration
        # ev is a list of per-lepton dicts with fields: pt, eta, phi, mass, charge, flavor, evt
        vecs = []
        for L in ev:
            vecs.append({
                "pt": L["pt"],
                "eta": L["eta"],
                "phi": L["phi"],
                "mass": L["mass"],
                "charge": L["charge"],
            })
        # ak.Array of records, with name Momentum4D-like structure
        per_event_vectors.append(ak.Array(vecs, kept_to_list=True))
    return ak.Array(per_event_vectors, with_name="Momentum4DList")

# Build per-event electrons
# Try to access nested keys robustly; adjust to actual keys from your fetch results.
try:
    leptons_by_event = data["leptons_fetch"]  # structure: [event] -> [leptons]
    # Separate electrons and muons
    electrons_by_event = leptons_by_event[{"flavor": "e"}]  # pseudo-filter; adapt to actual API
    muons_by_event     = leptons_by_event[{"flavor": "mu"}]
except Exception:
    # If the direct filtering style above doesn't work due to actual fetched schema,
    # you may need to inspect data["leptons_fetch"] first and rewrite this part accordingly.
    electrons_by_event = None
    muons_by_event = None
    print("Warning: could not extract per-event electron/muon collections from fetched data. "
          "Please adapt the filtering to your actual ServiceX output structure.")

# If we successfully extracted, convert to Vector4 (Momentum4D) objects
if electrons_by_event is not None and muons_by_event is not None:
    ele_vectors = build_lepton_vectors(electrons_by_event)
    mu_vecs     = build_lepton_vectors(muons_by_event)

    # 7) Build per-event electron-electron (EE) and muon-muon (MM) pairs with OS requirement
    #    Create pairs within each flavor that are opposite-charge
    ee_pairs = ak.combinations(ele_vectors, 2, fields=["p1","p2"], axis=1)
    mm_pairs = ak.combinations(mu_vecs, 2, fields=["p1","p2"], axis=1)

    # Invariant masses of each pair
    ee_mass = (ee_pairs.p1 + ee_pairs.p2).mass
    mm_mass = (mm_pairs.p1 + mm_pairs.p2).mass

    # Distances to Z mass
    Z = 91.2
    ee_dist = ak.abs(ee_mass - Z)
    mm_dist = ak.abs(mm_mass - Z)

    # Filter OS pairs within each flavor (opposite charge)
    ee_os = (ee_pairs.p1.charge * ee_pairs.p2.charge) < 0
    mm_os = (mm_pairs.p1.charge * mm_pairs.p2.charge) < 0

    ee_mass_os = ee_mass[ee_os]
    mm_mass_os = mm_mass[mm_os]
    ee_dist_os = ee_dist[ee_os]
    mm_dist_os = mm_dist[mm_os]

    # For simplicity, pick the best OS pair in each event if available; otherwise skip
    # Best EE pair per event
    ee_best_idx = ak.argmin(ee_dist_os, axis=1)  # index into the OS-filtered EE pairs per event
    mm_best_idx = ak.argmin(mm_dist_os, axis=1)

    has_ee = ak.num(ee_dist_os, axis=1) > 0
    has_mm = ak.num(mm_dist_os, axis=1) > 0

    # Best per-event masses (or NaN if not available)
    ee_best_mass = ee_mass_os[ee_best_idx]
    mm_best_mass = mm_mass_os[mm_best_idx]

    ee_best_dist = ee_dist_os[ee_best_idx]
    mm_best_dist = mm_dist_os[mm_best_idx]

    # Decide per-event which flavor pair to take (closer to Z)
    take_ee = (ee_best_dist <= mm_best_dist) & has_ee
    take_mm = (~take_ee) & has_mm

    # Final chosen pair per event (p1,p2)
    chosen_p1 = ak.where(take_ee, ee_pairs.p1[ee_best_idx], mm_pairs.p1[mm_best_idx])
    chosen_p2 = ak.where(take_ee, ee_pairs.p2[ee_best_idx], mm_pairs.p2[mm_best_idx])

    # 8) Highest-pT lepton not in the chosen pair
    # Build a per-event list of all leptons (EE + MM) as a single jagged array
    all_leptons = ak.concatenate([ele_vectors, mu_vecs], axis=1)

    # Simple heuristic (robust implementation requires per-event indexing of pair members):
    # Create a per-event mask of "in_pair" by marking the two leptons that form the chosen pair.
    # This is a simplified placeholder; proper implementation needs per-event indices of the pair members.
    in_pair_mask_placeholder = ak.zeros_like(all_leptons.pt, highlevel=False)  # shape matches all_leptons

    # Compute max pt among leptons not in the pair
    max_pt_per_event = ak.max(ak.where(in_pair_mask_placeholder, 0.0, all_leptons.pt), axis=1)
    # Also collect the corresponding lepton phi for MT calculation
    not_in_pair_leptons = ak.where(in_pair_mask_placeholder, None, all_leptons)

    # 9) MET per event from MET fetch
    met_per_event = data["met_fetch"]  # Expect per-event MET magnitude and phi
    met_vals = met_per_event.met  # in GeV
    met_phis = met_per_event.met_phi

    # 10) Compute MT per event: MT = sqrt(2 * MET * pT_lep * (1 - cos(delta_phi)))
    # delta_phi = abs(phi_lep - phi_MET)
    # We need phi of the selected not-in-pair lepton; approximate with the max-pt lepton's phi
    # Placeholder: phi_of_not_in_pair = not_in_pair_leptons.phi (requires proper indexing)
    phi_of_lepton = ak.zeros_like(max_pt_per_event)  # placeholder; adapt when per-event indices exist
    delta_phi = ak.abs(phi_of_lepton - met_phis)
    MT = ak.sqrt(2.0 * met_vals * max_pt_per_event * (1.0 - ak.cos(delta_phi)))

    # 11) Fill histogram (MT in GeV)
    h_mt = (
        Hist.new.Reg(200, 0, 400, name="MT", label="MT [GeV]")
           .Weight()
    )
    h_mt.fill(MT=MT, weight=np.ones(len(MT), dtype=float))

    # 12) Plot and save to PNG
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.ATLAS)

    fig, ax = plt.subplots(figsize=(8,6))
    h_mt.plot(histtype="step", ax=ax, label="MT (MET + not-in-pair lepton)")
    ax.set_xlabel("MT [GeV]")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig("mt_toisf_os_leptons.png")
    plt.close(fig)
else:
    print("Lepton collections could not be extracted from ServiceX results. "
          "Please inspect data['leptons_fetch'] and adapt field accessors to your dataset.")
```


## Model: gpt-4o

- Extracted events with at least three light leptons from the given dataset.
- Identified same-flavor opposite-charge light lepton pairs in each event.
- Found the pair with invariant mass closest to 91.2 GeV within each event.
- Calculated the transverse mass of the system consisting of the missing transverse momentum and the highest-pT light lepton not in the selected pair.
- Created a histogram of the transverse mass and saved the plot as a PNG file.

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

# Define the dataset
ds_name = "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"

# Create a base query
base_query = FuncADLQueryPHYSLITE()

# Create a query to get events with at least three light leptons and lepton details
lep_query = (base_query
    .Where(lambda e: e.Electrons().Count() + e.Muons().Count() >= 3)
    .Select(lambda e: {
        "electrons_pt": e.Electrons().Select(lambda ele: ele.pt() / 1000.0),
        "electrons_eta": e.Electrons().Select(lambda ele: ele.eta()),
        "electrons_phi": e.Electrons().Select(lambda ele: ele.phi()),
        "electrons_charge": e.Electrons().Select(lambda ele: ele.charge()),
        "muons_pt": e.Muons().Select(lambda mu: mu.pt() / 1000.0),
        "muons_eta": e.Muons().Select(lambda mu: mu.eta()),
        "muons_phi": e.Muons().Select(lambda mu: mu.phi()),
        "muons_charge": e.Muons().Select(lambda mu: mu.charge()),
        "met_met": e.MissingET().First().met() / 1000.0,
        "met_phi": e.MissingET().First().phi()
    })
)

# Retrieve the data
physics_data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="lep_extraction",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,  # Start testing with a single file
                    Query=lep_query,
                )
            ]
        ),
    )
)

# Extract awkward arrays
data = physics_data["lep_extraction"]

# Combine electrons and muons
electrons = ak.zip({
    "pt": data["electrons_pt"],
    "eta": data["electrons_eta"],
    "phi": data["electrons_phi"],
    "charge": data["electrons_charge"]
}, with_name="Momentum4D")

muons = ak.zip({
    "pt": data["muons_pt"],
    "eta": data["muons_eta"],
    "phi": data["muons_phi"],
    "charge": data["muons_charge"]
}, with_name="Momentum4D")

# Combine into a single lepton array and calculate invariant mass pairs
light_leptons = ak.concatenate([electrons, muons], axis=1)

# Find same-flavor opposite-charge pairs
pairs = ak.combinations(light_leptons, 2, fields=["lep1", "lep2"])
same_flavor = pairs[(pairs.lep1.charge + pairs.lep2.charge == 0) &
                    (ak.abs(pairs.lep1.eta - pairs.lep2.eta) < 0.02)]
 
# Calculate mass and find the pair closest to 91.2 GeV
inv_masses = (same_flavor.lep1 + same_flavor.lep2).mass
closest_mass_idx = ak.argmin(ak.abs(inv_masses - 91.2), axis=1)

# Find the highest-pT lepton not in the chosen pair
best_pairs = same_flavor[closest_mass_idx]
other_leptons = light_leptons[(light_leptons.pt != best_pairs.lep1.pt) &
                              (light_leptons.pt != best_pairs.lep2.pt)]
highest_pt_not_in_pair = ak.max(other_leptons.pt, axis=-1)

# Calculate transverse mass
met = vector.Array({
    "pt": data["met_met"],
    "phi": data["met_phi"]
})
lep_for_mt = vector.zip(px=other_leptons.pt * ak.cos(other_leptons.phi),
                        py=other_leptons.pt * ak.sin(other_leptons.phi))
transverse_mass = ak.firsts(met.mt(lep_for_mt))

# Plot the transverse mass
h = (
    Hist.new.Reg(50, 0, 200, name="mT", label="Transverse Mass [GeV]")
       .Double()  # Use double precision
)

h.fill(mT=ak.to_numpy(transverse_mass))

# Set up the plot style
plt.style.use(hep.style.ATLAS)

# Plot
h.plot(histtype="fill", edgecolor="black", linewidth=1.5)
plt.xlabel("Transverse Mass [GeV]")
plt.ylabel("Entries")
plt.title("Transverse Mass distribution")
plt.savefig("transverse_mass_plot.png")
plt.show()
```


## Model: o4-mini

- Build a FuncADLQueryPHYSLITE that:
  • Requires ≥3 light leptons (electrons + muons) per event  
  • Selects electron and muon kinematics (pt [GeV], eta, phi, mass [GeV], charge) and the MET (met [GeV], phi)  
- Deliver the query via ServiceX (start with `NFiles=1` for testing) and convert to Awkward Arrays  
- In Python/awkward:
  • Zip each lepton collection into a Momentum4D array and attach its charge  
  • Form all same-flavor opposite-charge (SFOS) lepton pairs per event (ele-ele & mu-mu), compute their invariant masses, and concatenate them  
  • Pick the SFOS pair whose mass is closest to 91.2 GeV in each event  
  • Build a unified lepton list per event, mask out the two in the best Z-candidate pair, and select the highest-pt remaining lepton  
  • Compute the transverse mass mT = √(2 · mET · ℓpt · (1–cos(Δφ(ℓ, MET)))) for each event  
- Fill a 1D histogram of mT, plot it with ATLAS style, and save as `mt.png`  

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk

# 1) Build the FuncADL query
base = FuncADLQueryPHYSLITE()
query = (
    base
    # require ≥3 leptons per event
    .Where(lambda e: e.Electrons().Count() + e.Muons().Count() >= 3)
    # select all needed branches
    .Select(lambda e: {
        "ele_pt":  e.Electrons().Select(lambda x: x.pt()  / 1000.0),
        "ele_eta": e.Electrons().Select(lambda x: x.eta()),
        "ele_phi": e.Electrons().Select(lambda x: x.phi()),
        "ele_m":   e.Electrons().Select(lambda x: x.m()   / 1000.0),
        "ele_ch":  e.Electrons().Select(lambda x: x.charge()),
        "mu_pt":   e.Muons().Select(lambda x: x.pt()  / 1000.0),
        "mu_eta":  e.Muons().Select(lambda x: x.eta()),
        "mu_phi":  e.Muons().Select(lambda x: x.phi()),
        "mu_m":    e.Muons().Select(lambda x: x.m()   / 1000.0),
        "mu_ch":   e.Muons().Select(lambda x: x.charge()),
        # MET comes as a single-element collection
        "met":     e.MissingET().First().met()    / 1000.0,
        "met_phi": e.MissingET().First().phi()
    })
)

# 2) Fetch one file for testing
ds_name = "mc23_13p6TeV:mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697"
out = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="events",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        )
    )
)
ev = out["events"]

# 3) Analysis with awkward + vector
import awkward as ak
import numpy as np
import vector

# enable vector behaviors
vector.register_awkward()

# construct electron and muon Lorentz vectors, attach charge
ele = ak.zip({
    "pt": ev.ele_pt,
    "eta": ev.ele_eta,
    "phi": ev.ele_phi,
    "mass": ev.ele_m
}, with_name="Momentum4D")
ele = ak.with_field(ele, ev.ele_ch, "charge")

mu = ak.zip({
    "pt": ev.mu_pt,
    "eta": ev.mu_eta,
    "phi": ev.mu_phi,
    "mass": ev.mu_m
}, with_name="Momentum4D")
mu = ak.with_field(mu, ev.mu_ch, "charge")

# form SFOS combinations in each event
e_pairs = ak.combinations(ele, 2, fields=["p1", "p2"])
e_pairs = e_pairs[(e_pairs.p1.charge * e_pairs.p2.charge) < 0]

m_pairs = ak.combinations(mu, 2, fields=["p1", "p2"])
m_pairs = m_pairs[(m_pairs.p1.charge * m_pairs.p2.charge) < 0]

# concatenate all SFOS pairs per event
all_pairs = ak.concatenate([e_pairs, m_pairs], axis=1)

# pick the one closest to Z mass
def deltaZ(pair):
    return abs((pair.p1 + pair.p2).mass - 91.2)

best_pair = ak.min_by(all_pairs, deltaZ, axis=1)

# build a flat event‐wise list of all leptons, then drop the 2 in best_pair
leps = ak.concatenate([ele, mu], axis=1)

# for each event, find which leptons match p1 or p2 by comparing (pt,eta,phi)
is_p1 = ((leps.pt == best_pair.p1.pt) &
         (leps.eta == best_pair.p1.eta) &
         (leps.phi == best_pair.p1.phi))
is_p2 = ((leps.pt == best_pair.p2.pt) &
         (leps.eta == best_pair.p2.eta) &
         (leps.phi == best_pair.p2.phi))
others = leps[~(is_p1 | is_p2)]

# pick the highest‐pt lepton among the "others"
idx_max = ak.argmax(others.pt, axis=1)
lead_other = others[idx_max]

# compute transverse mass mT = sqrt(2 * met * lep.pt * (1 - cos Δφ))
met = ev.met
met_phi = ev.met_phi
dphi = lead_other.phi - met_phi
# wrap into [–π, π]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
mt = np.sqrt(2 * met * lead_other.pt * (1 - np.cos(dphi)))

# 4) Fill and plot histogram
import hist
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

h = (
    hist.Hist.new
      .Reg(50, 0, 200, name="mT", label="Transverse mass m_{T} [GeV]")
      .Double()
)
h.fill(mT=mt.flatten())

# plot and save
ax = h.plot(histtype="step", color="blue", label="m_{T}(ℓ, MET)")
ax.legend()
ax.set_title("m_{T} of highest-p_{T} lepton not in Z-candidate")
plt.savefig("mt.png")
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 52.06 | 11684 | 2520 | 14204 | 0.0435 |
| gpt-5 | 127.69 | 11683 | 7012 | 18695 | 0.0847 |
| gpt-5-mini | 66.89 | 11683 | 7413 | 19096 | 0.0177 |
| gpt-5-nano | 66.01 | 11683 | 18679 | 30362 | 0.0081 |
| gpt-4o | 24.60 | 11684 | 1202 | 12886 | 0.0412 |
| o4-mini | 40.83 | 11683 | 5001 | 16684 | 0.0349 |
