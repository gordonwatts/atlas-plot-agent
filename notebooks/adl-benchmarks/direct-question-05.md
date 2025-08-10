# Plot the ETmiss of events that have an opposite-charge muon pair with an invariant mass between 60 and 120 GeV in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697.



## Model: gpt-4.1

- Fetched data from the user-specified rucio dataset of Z→μμ events in DAOD_PHYSLITE format using ServiceX.
- For each event, selected the muon four-momenta and charges. For all unique pairs of muons, computed the pair invariant mass and checked their charges for being opposite.
- Selected events with at least one opposite-charge muon pair with invariant mass in [60, 120] GeV.
- For those events, extracted the ETmiss (missing transverse energy).
- Plotted a histogram of ETmiss and saved it to a `png` file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import numpy as np
import vector

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

vector.register_awkward()

# Step 1: Build the query to extract muon kinematics, charge, and ETmiss
base_query = FuncADLQueryPHYSLITE()

# We'll get muon pt, eta, phi, mass, charge, and the first MissingET per event
query = (
    base_query
    .Select(lambda e: {
        "muon_pt":   e.Muons().Select(lambda mu: mu.pt() / 1000.0),     # GeV
        "muon_eta":  e.Muons().Select(lambda mu: mu.eta()),
        "muon_phi":  e.Muons().Select(lambda mu: mu.phi()),
        "muon_mass": e.Muons().Select(lambda mu: mu.m()),               # mass is in MeV
        "muon_charge": e.Muons().Select(lambda mu: mu.charge()),
        "etmiss": e.MissingET().First().met() / 1000.0,
    })
)

# Step 2: Run the query on the dataset
ds_name = "mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697"

result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="muon_pair_etmiss",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,    # Start with 1 file for testing. You can increase NFiles for more statistics.
                    Query=query,
                )
            ]
        ),
    )
)

events = result["muon_pair_etmiss"]

# Step 3: Build muon 4-vectors (convert mass to GeV)
muon_vectors = ak.zip(
    {
        "pt": events.muon_pt,
        "eta": events.muon_eta,
        "phi": events.muon_phi,
        "mass": events.muon_mass / 1000.0,  # Convert MeV to GeV
    },
    with_name="Momentum4D"
)
muon_charge = events.muon_charge
etmiss = events.etmiss

# Step 4: For each event, find all unique muon pairs, check for opposite charge and mass in [60,120] GeV
pair_idx = ak.combinations(ak.local_index(muon_charge), 2, fields=["i1", "i2"], axis=1)
# For safety, handle events with fewer than 2 muons (ak.combinations returns [] in those cases)

# Select the muon pairs themselves
muon_pairs = ak.combinations(muon_vectors, 2, fields=["mu1", "mu2"], axis=1)
charge_pairs = ak.combinations(muon_charge, 2, fields=["q1", "q2"], axis=1)

# Invariant mass of the pair
dimu_mass = (muon_pairs.mu1 + muon_pairs.mu2).mass

# Check for opposite charge
is_opposite_charge = charge_pairs.q1 != charge_pairs.q2

# Only consider pairs with opposite charge and 60 < mass < 120 GeV
mass_in_window = (dimu_mass > 60) & (dimu_mass < 120)
good_pair = is_opposite_charge & mass_in_window

# For each event, does there exist at least one such pair?
event_has_good_pair = ak.any(good_pair, axis=1)

# Select ETmiss for events passing the selection
etmiss_selected = etmiss[event_has_good_pair]

# Step 5: Plot the ETmiss distribution

hist_etmiss = (
    Hist.new
    .Reg(40, 0, 200, name="etmiss", label="E$_T^{miss}$ [GeV]")
    .Weight()
)
hist_etmiss.fill(etmiss=ak.flatten(etmiss_selected))

plt.style.use(hep.style.ATLAS)
hist_etmiss.plot(histtype="step", linewidth=2, label=r"$Z\to\mu\mu$ events")
plt.xlabel(r"$E_T^{miss}$ [GeV]")
plt.ylabel("Events")
plt.legend()
plt.title(r"$E_T^{miss}$ for $Z\to\mu^+\mu^-$ events,\ $60 < m_{\mu\mu} < 120$ GeV")
plt.tight_layout()
plt.savefig("etmiss_zmm_oppositecharge_pair.png")
print("Saved plot to etmiss_zmm_oppositecharge_pair.png")
```


## Model: gpt-5

- Uses ServiceX (FuncADL, PHYSLITE) to fetch only what’s needed: muon kinematics (pt/eta/phi/charge) and event MET, with an event-level pre-filter of at least two muons with pt > 10 GeV to reduce I/O.
- Constructs muon Momentum4D vectors with Awkward+Vector (keeping everything in GeV).
- Forms all opposite-charge muon pairs per event, computes invariant masses, and selects events with any OS pair in 60–120 GeV.
- Fills and plots the E_T^miss distribution with Hist/mplhep and saves it to a PNG file.
- For quick testing, the code processes only 1 file (NFiles=1). Remove NFiles (or set it to None) to run over the full dataset.

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# 1) Build the ServiceX/func_adl query on PHYSLITE:
#    - Pre-filter events with at least two muons above 10 GeV (to reduce I/O).
#    - Select per-event muon pt/eta/phi/charge and event MET (converted to GeV).
base = FuncADLQueryPHYSLITE()

mu_pt_min = 10.0  # GeV
query = (
    base
    .Where(lambda e:
           e.Muons()
            .Where(lambda mu: mu.pt() / 1000.0 > mu_pt_min)
            .Count() >= 2)
    .Select(lambda e: {
        "mu_pt": e.Muons()
                 .Where(lambda mu: mu.pt() / 1000.0 > mu_pt_min)
                 .Select(lambda mu: mu.pt() / 1000.0),
        "mu_eta": e.Muons()
                  .Where(lambda mu: mu.pt() / 1000.0 > mu_pt_min)
                  .Select(lambda mu: mu.eta()),
        "mu_phi": e.Muons()
                  .Where(lambda mu: mu.pt() / 1000.0 > mu_pt_min)
                  .Select(lambda mu: mu.phi()),
        "mu_charge": e.Muons()
                     .Where(lambda mu: mu.pt() / 1000.0 > mu_pt_min)
                     .Select(lambda mu: mu.charge()),
        "met": e.MissingET().First().met() / 1000.0,  # GeV
    })
)

# 2) Define the dataset (Rucio) and deliver via ServiceX.
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv."
    "DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697"
)

result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="zmumu_met",
                    Dataset=dataset.Rucio(ds_name),
                    Query=query,
                    NFiles=1,  # for a quick test; remove or set to None for full dataset
                )
            ]
        )
    )
)

events = result["zmumu_met"]

# 3) Build muon 4-vectors (Momentum4D) with Awkward+Vector.
#    Use a per-muon mass of 0.10566 GeV (muon rest mass).
mu_mass = ak.ones_like(events.mu_pt) * 0.10566  # broadcast to the jagged muon structure
mu_p4 = ak.zip(
    {
        "pt": events.mu_pt,
        "eta": events.mu_eta,
        "phi": events.mu_phi,
        "mass": mu_mass,
    },
    with_name="Momentum4D",
)

# 4) Pair muons within each event, require opposite charge and 60 < m_mumu < 120 GeV.
#    Zip charge together with p4 to keep them aligned through combinations.
muons = ak.zip({"p4": mu_p4, "charge": events.mu_charge})

pairs = ak.combinations(muons, 2, fields=["m1", "m2"], axis=1)
os_mask = (pairs.m1.charge * pairs.m2.charge) == -1

pair_mass = (pairs.m1.p4 + pairs.m2.p4).mass
z_window = (pair_mass > 60.0) & (pair_mass < 120.0)

# Any OS pair in the window per event?
select_mask = ak.any(os_mask & z_window, axis=1)

# 5) Select MET for those events and make a histogram.
met_selected = events.met[select_mask]

# Histogram: E_T^miss in [0, 300] GeV with 60 bins
h_met = Hist.new.Reg(60, 0.0, 300.0, name="met", label=r"E_T^{miss} [GeV]").Int64()
h_met.fill(met=ak.to_numpy(met_selected))

# 6) Plot and save
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(7, 5))
h_met.plot(ax=ax, histtype="fill", linewidth=1.0, edgecolor="gray", label=r"OS $\mu\mu$ (60<m<120)")

ax.set_xlabel(r"$E_T^{\mathrm{miss}}$ [GeV]")
ax.set_ylabel("Events")
ax.legend()
ax.set_title("MET in events with OS dimuon in Z window")
plt.tight_layout()
plt.savefig("met_os_dimuon.png", dpi=150)
plt.close(fig)

print(f"Selected events: {len(met_selected)}")
print("Saved plot to met_os_dimuon.png")
```


## Model: gpt-5-mini

- I fetched muon four-vectors and event ETmiss from the specified PHYSLITE rucio dataset using ServiceX (I limited the fetch to NFiles=1 for a quick test — remove NFiles or set it larger to run on more files / full dataset, which will be more expensive).
- I required events to have at least two muons at the query stage to reduce transferred data.
- Using Awkward + vector, I built per-event muon four-vectors, formed all unique muon pairs per event, selected opposite-charge pairs whose invariant mass is between 60 and 120 GeV, and kept events that have at least one such pair.
- I filled a 1D histogram of ETmiss (GeV) for the selected events with Hist and saved the plot to met_zmm.png.

```python
# Fetch Z->mumu candidate events and plot ETmiss
# - Uses ServiceX + FuncADLQueryPHYSLITE to fetch only necessary variables
# - Uses Awkward + vector for all physics manipulations (no Python lists)
# - Uses hist to build a histogram and matplotlib/mplhep to style/save the figure

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import vector
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# Register vector behaviors so that Momentum4D properties are available on awkward records
vector.register_awkward()

# Build the FuncADL query:
# - require at least two muons at the ServiceX level to reduce transferred data
# - fetch muon px,py,pz,E (convert MeV -> GeV) and charge for each muon in the event
# - fetch the event MET (MissingET.First().met()) in GeV
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Where(lambda e: e.Muons().Count() >= 2)
    .Select(lambda e: {
        "mu_px":     e.Muons().Select(lambda m: m.px() / 1000.0),
        "mu_py":     e.Muons().Select(lambda m: m.py() / 1000.0),
        "mu_pz":     e.Muons().Select(lambda m: m.pz() / 1000.0),
        "mu_E":      e.Muons().Select(lambda m: m.e()  / 1000.0),
        "mu_charge": e.Muons().Select(lambda m: m.charge()),
        "met":       e.MissingET().First().met() / 1000.0
    })
)

# Define the dataset (as requested)
ds_name = "mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697"

# Deliver the query. Start with NFiles=1 for testing (remove or increase for more statistics)
sx_spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="zmm_fetch",
            Dataset=dataset.Rucio(ds_name),
            NFiles=1,
            Query=query,
        )
    ]
)

# This will run the ServiceX job and convert results to awkward arrays.
# Note: running on the whole dataset is expensive; keep NFiles small for testing.
out = to_awk(deliver(sx_spec))
data = out["zmm_fetch"]

# Data fields: data.mu_px, data.mu_py, data.mu_pz, data.mu_E, data.mu_charge, data.met
# Build an awkward array of muon 4-vectors per event, including the charge as an extra field.
mu = ak.zip(
    {
        "px":    data.mu_px,
        "py":    data.mu_py,
        "pz":    data.mu_pz,
        "E":     data.mu_E,
        "charge": data.mu_charge,  # extra field — Vector behaviors will ignore it but it's kept in the record
    },
    with_name="Momentum4D",
)

# Form all unique 2-combinations of muons in each event
pairs = ak.combinations(mu, 2, fields=["p1", "p2"], axis=1)

# Compute pair invariant mass and require opposite charge
pair_mass = (pairs.p1 + pairs.p2).mass  # in GeV
pair_charge_sum = pairs.p1.charge + pairs.p2.charge

# Mask for opposite-charge pairs with 60 <= m_inv <= 120 GeV
pair_mask = (pair_mass >= 60.0) & (pair_mass <= 120.0) & (pair_charge_sum == 0)

# Keep events that have at least one pair passing the mask
event_has_zmm = ak.any(pair_mask, axis=1)

# Extract ETmiss for selected events
mets = data.met[event_has_zmm]  # awkward 1D array (one met per surviving event)

# Create a histogram for ETmiss (GeV)
h = Hist.new.Reg(60, 0, 300, name="met", label="ETmiss [GeV]").Int64()

# Fill the histogram. Convert the awkward array of MET values to numpy for filling.
# (We used only awkward arrays for the physics computations; conversion to numpy here
# is only for the histogram-filling API.)
mets_np = ak.to_numpy(mets) if len(mets) > 0 else np.array([], dtype=float)
h.fill(met=mets_np)

# Plot with mplhep/ATLAS style and save to PNG
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(8,6))
h.plot(histtype="stepfilled", ax=ax, color="C0", label="Z→μμ (60–120 GeV)")
ax.set_xlabel("ETmiss [GeV]")
ax.set_ylabel("Entries")
ax.set_title("ETmiss for events with an opposite-charge μμ pair, 60–120 GeV")
ax.legend()
plt.tight_layout()
fig.savefig("met_zmm.png", dpi=150)
print("Saved histogram to met_zmm.png")
```


## Model: gpt-5-nano

- I will fetch PHYSLITE DAOD data for the given rucio dataset using ServiceX.
- I will build per-event muon collections, construct 4-momentum vectors from muon kinematics, and form all same-event opposite-charge muon pairs.
- I will compute the invariant mass of each muon pair, select pairs in the 60–120 GeV window, and then mark events that have at least one such pair.
- I will plot the missing ET (ETmiss) distribution for events that pass the muon-pair mass requirement, using Awkward Array for data manipulation, Vector for 4-vectors, and Hist to create a PNG plot.
- The code uses only Awkward arrays for data manipulation after fetch and outputs a PNG file via Matplotlib.

```python
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

# ServiceX imports (these are the standard entry points for ServiceX XAOD_r25)
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import deliver, ServiceXSpec, Sample, dataset

# Ensure Vector's Awkward behaviors are available
vector.register_awkward()

# 1) Define the dataset (provided in the user prompt)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697"
)

# 2) Build a PHYSLITE-based query:
# We fetch per-event muons with essential kinematic fields and the event MET.
# We convert muon pT, eta, phi to a 4-vector (px, py, pz, E) in GeV and keep the charge.
# Note: μ_mass is the muon mass (GeV). We use 0.105658 GeV as a good approximation.
base_query = FuncADLQueryPHYSLITE()

muon_mass = 0.105658  # GeV

muon_met_query = (
    base_query
    .Select(lambda e: {
        # Muon collection for the event
        "muons": e.Muons().Select(lambda m: {
            "pt": m.pt() / 1000.0,      # convert MeV -> GeV
            "eta": m.eta(),
            "phi": m.phi(),
            "charge": m.charge(),
        }),
        # Missing ET (ETmiss) for the event (GeV)
        "met": e.MissingET().First().met() / 1000.0,
    })
)

# 3) Deliver the data via ServiceX
# Use NFiles=1 for testing; you can remove NFiles to process all available files.
sample = Sample(
    Name="zmumu_met_fetch",
    Dataset=dataset.Rucio(ds_name),
    NFiles=1,
    Query=muon_met_query,
)

spec = ServiceXSpec(Sample=[sample])

# 4) Convert the delivered data to Awkward Array
# The result is a dict with the sample name as key, containing muons and MET per event.
awkd = deliver(spec)
# Convert to Awkward for analysis
# The helper utility to_awk is commonly used in this workflow; here we emulate it:
try:
    from servicex_analysis_utils import to_awk
    # The to_awk utility converts the delivered dict into an Awkward array per sample
    all_events = to_awk(awkd)
    events = all_events["zmumu_met_fetch"]
except Exception:
    # If to_awk isn't available in your environment, fall back to the raw dict
    # and operate with the direct dictionary; this is a best-effort placeholder.
    events = awkd.get("zmumu_met_fetch", {})

# 5) Build per-event muon 4-vectors from pt, eta, phi, mass
# We expect events["muons"] to be a list per event of muon dicts with keys: pt, eta, phi, charge
# We'll create per-muon 4-vectors using pt, eta, phi, and muon mass; we will also carry the charge.
def build_muons_with_vectors(event_muons):
    # event_muons: 1D array (jagged) of muon dicts with fields: pt, eta, phi, charge
    if event_muons is ak.Nothing:
        return ak.Array([])

    # Convert to per-muon Momentum4D objects using pt/eta/phi/mass
    # We create a per-event list of Momentum4D vectors with fields px, py, pz, E
    # px = pt * cos(phi), py = pt * sin(phi), pz = pt * sinh(eta)
    pt = ak.to_numpy(event_muons["pt"])
    eta = ak.to_numpy(event_muons["eta"])
    phi = ak.to_numpy(event_muons["phi"])
    charge = ak.to_numpy(event_muons["charge"])
    # Vectorized geometry operations (Ak.asarray + vector operations)
    # Use awkward-friendly operations (ak.*) for jagged structures
    # Compute px, py, pz, E (GeV)
    px = event_muons["pt"] * ak.cos(event_muons["phi"])  # pt is already in GeV
    py = event_muons["pt"] * ak.sin(event_muons["phi"])
    pz = event_muons["pt"] * ak.sinh(event_muons["eta"])
    # Energy assuming muon mass
    E = ak.sqrt(px*px + py*py + pz*pz + muon_mass*muon_mass)

    # Create Momentum4D-like objects with fields px, py, pz, E
    p4 = ak.zip({"px": px, "py": py, "pz": pz, "E": E}, with_name="Momentum4D")
    # Return a per-event array of dicts with fields "p4" and "q"
    return ak.zip({"p4": p4, "q": charge}, with_name="MuonWithCharge")

# 6) Compute all opposite-charge muon pairs per event
# We assume events is an Awkward Array with a field "muons" (list per event)
# For demonstration purposes, we attempt to map each event's muons into pairs.
muons_per_event = events.muons  # per-event list of muon dicts
muons_with_vectors = muons_per_event.Select(lambda mu_list: build_muons_with_vectors(mu_list))

# Create all 2-element combinations per event: m1 and m2
pairs = ak.combinations(muons_with_vectors, 2, axis=1, fields=["m1", "m2"])

# Opposite-charge condition
opposite_charge = (pairs.m1.q * pairs.m2.q) < 0

# Invariant mass for each pair: (m1 + m2).mass
# Here m1 and m2 are Momentum4D-like objects under field m1.p4 and m2.p4
inv_mass = (pairs.m1.p4 + pairs.m2.p4).mass

# 60 GeV < m_ll < 120 GeV window
mass_window = (inv_mass >= 60.0) & (inv_mass <= 120.0)

# Pairs that satisfy both opposite charge and mass window
passing_pairs = pairs[mass_window & opposite_charge]

# 7) Event-level MET for events that have at least one such pair
# If there are any passing pairs in an event, we consider that event “selected”
# We need to determine per-event if there exists any passing pair.
# Create a per-event mask indicating presence of at least one passing pair
# Since passing_pairs is per-event (axis=1), we can reduce with any(...)
event_has_pass = ak.any((mass_window & opposite_charge), axis=1)

# MET per event
met_per_event = events.met if "met" in events else events.MissingET().First().met()/1000.0

# Apply the event-level mask to MET to get MET values for selected events only
selected_met = met_per_event[event_has_pass]

# 8) Build and fill a 1D histogram of ETmiss in GeV for the selected events
# Histogram: 100 bins from 0 to 200 GeV
h_met = (
    Hist.new.Reg(100, 0, 200, name="met", label="ETmiss [GeV]")
)

# Fill: selected_met is an Awkward Array of GeV values for selected events
h_met.fill(met=selected_met)

# 9) Plot the histogram to a PNG file
plt.style.use(hep.style.ATLAS)
ax = h_met.plot(histtype="step", linewidth=1.5)
plt.xlabel("ETmiss [GeV]")
plt.ylabel("Counts")
plt.title("ETmiss for events with opposite-charge muon pair in 60-120 GeV window")
plt.tight_layout()
plt.savefig("etmiss_pass_muon_pair_60_120GeV.png")
plt.close()
```


## Model: gpt-4o

- I will query the specified dataset to fetch muons for each event and calculate the invariant mass of opposite-charge muon pairs.
- I will filter events to select those containing an opposite-charge muon pair with an invariant mass between 60 and 120 GeV.
- I will then extract the missing transverse energy (ETmiss) from these filtered events.
- Finally, I will plot a histogram of ETmiss for the selected events and save it to a PNG file.

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset
import awkward as ak
import hist
import vector
import matplotlib.pyplot as plt
import mplhep as hep

# Register vector behavior for awkward arrays
vector.register_awkward()

# Define the base query for PHYSLITE dataset
base_query = FuncADLQueryPHYSLITE()

# Query to select muon pairs and ETmiss
query = (base_query
    .Select(lambda e: {
        'muons': e.Muons(),
        'met': e.MissingET().First().met() / 1000.0
    })
)

# ServiceX delivery for the dataset
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697")

result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="muon_etmiss",
                    Dataset=dataset.Rucio(ds_name),
                    # Set NFiles=1 for testing; increase as needed
                    NFiles=1,
                    Query=query,
                )
            ]
        ),
    )
)

# Extract data from the result
muons = result["muon_etmiss"]["muons"]
met = result["muon_etmiss"]["met"]

# Calculate invariant mass of opposite-charge muon pairs
muon_pairs = ak.combinations(muons, 2, fields=["mu1", "mu2"], with_name="Momentum4D")
opposite_charge_pairs = muon_pairs[(muon_pairs.mu1.charge + muon_pairs.mu2.charge) == 0]
muon1_vec = ak.zip({
    "pt": opposite_charge_pairs.mu1.pt() / 1000.0,
    "eta": opposite_charge_pairs.mu1.eta(),
    "phi": opposite_charge_pairs.mu1.phi(),
    "mass": opposite_charge_pairs.mu1.m() / 1000.0
}, with_name="Momentum4D")
muon2_vec = ak.zip({
    "pt": opposite_charge_pairs.mu2.pt() / 1000.0,
    "eta": opposite_charge_pairs.mu2.eta(),
    "phi": opposite_charge_pairs.mu2.phi(),
    "mass": opposite_charge_pairs.mu2.m() / 1000.0
}, with_name="Momentum4D")
invariant_mass = (muon1_vec + muon2_vec).mass

# Filter events with invariant mass between 60 and 120 GeV
selected_events = ak.any((invariant_mass > 60) & (invariant_mass < 120), axis=1)
selected_met = met[selected_events]

# Plotting the histogram of ETmiss
plt.style.use(hep.style.ATLAS)
hist_etmiss = hist.Hist(
    hist.axis.Regular(50, 0, 200, name="ETmiss", label=r"$E_T^{miss}$ [GeV]")
)
hist_etmiss.fill(ETmiss=ak.to_numpy(selected_met))
hist_etmiss.plot(histtype='step')
plt.xlabel(r"$E_T^{miss}$ [GeV]")
plt.title("ETmiss of Events with Opposite-Charge Muon Pair (60 < M < 120 GeV)")
plt.savefig("etmiss_histogram.png")
```


## Model: o4-mini

- Filtered for events with at least two muons at the ServiceX level and fetched per-muon four-momentum components (px, py, pz, E), muon charges, and the event missing transverse energy (MET).  
- In Python/Awkward: built Lorentz vectors, formed all muon–muon pairs per event, computed their invariant masses and checked for opposite charges in the window 60–120 GeV, then selected the MET of those events.  
- Filled a 1D histogram of the selected MET values using `hist`, plotted it with the ATLAS style via `mplhep`, and saved the plot to `etmiss.png`.  

```python
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import vector
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# Enable vector behaviors on Awkward arrays
vector.register_awkward()

# 1) Build and run the ServiceX query
base = FuncADLQueryPHYSLITE()
query = (
    base
    # Only events with at least two muons
    .Where(lambda e: e.Muons().Count() >= 2)
    .Select(lambda e: {
        # muon four-momentum components in GeV
        "mu_px":     e.Muons().Select(lambda mu: mu.px() / 1000.0),
        "mu_py":     e.Muons().Select(lambda mu: mu.py() / 1000.0),
        "mu_pz":     e.Muons().Select(lambda mu: mu.pz() / 1000.0),
        "mu_E":      e.Muons().Select(lambda mu: mu.E()  / 1000.0),
        # muon charges (+1 or –1)
        "mu_charge": e.Muons().Select(lambda mu: mu.charge()),
        # event missing ET in GeV
        "met":       e.MissingET().First().met() / 1000.0,
    })
)

dataset_name = (
    "mc23_13p6TeV:"
    "mc23_13p6TeV.513109.MGPy8EG_Zmumu_FxFx3jHT2bias_SW_CFilterBVeto."
    "deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r14622_r14663_p6697"
)

spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="muon_met_fetch",
            Dataset=dataset.Rucio(dataset_name),
            NFiles=1,          # start with 1 file for testing
            Query=query,
        )
    ]
)
results = deliver(spec)
awk_data = to_awk(results)["muon_met_fetch"]

# 2) In Python/Awkward: form Lorentz vectors and select events
muons = ak.zip({
    "px": awk_data["mu_px"],
    "py": awk_data["mu_py"],
    "pz": awk_data["mu_pz"],
    "E":  awk_data["mu_E"],
}, with_name="Momentum4D")

charges = awk_data["mu_charge"]
mets     = awk_data["met"]

# build all unique muon pairs per event
pairs      = ak.combinations(muons, 2, fields=["m1", "m2"])
charge_pairs = ak.combinations(charges, 2, fields=["c1", "c2"])

# compute invariant mass of each pair
inv_masses = (pairs.m1 + pairs.m2).mass

# require opposite charges and 60 < m_ll < 120 GeV
pair_mask = (charge_pairs.c1 != charge_pairs.c2) & (inv_masses >= 60.0) & (inv_masses <= 120.0)

# select events with at least one such pair
event_mask = ak.any(pair_mask, axis=1)

selected_mets = mets[event_mask]

# 3) Make and plot the histogram of MET
h_met = (
    Hist.new
    .Reg(50, 0, 200, name="met", label="Missing $E_T$ [GeV]")
    .Double()
)
h_met.fill(met=selected_mets)

plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots(figsize=(6,4))
h_met.plot(ax=ax, histtype="step", linewidth=1.5, label="60<m_{μμ}<120 GeV")
ax.set_title("Missing $E_T$ for Opposite-Charge μμ Pairs")
ax.legend()
fig.savefig("etmiss.png")
```
## Summary
| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost ($) |
|-------|----------|--------------|------------------|--------------|--------------------|
| gpt-4.1 | 13.95 | 11666 | 1198 | 12864 | 0.0329 |
| gpt-5 | 84.75 | 11665 | 4667 | 16332 | 0.0613 |
| gpt-5-mini | 34.18 | 11665 | 3883 | 15548 | 0.0107 |
| gpt-5-nano | 53.31 | 11665 | 14360 | 26025 | 0.0063 |
| gpt-4o | 19.30 | 11666 | 871 | 12537 | 0.0379 |
| o4-mini | 22.34 | 11665 | 2840 | 14505 | 0.0253 |
