from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from atlas_plot_agent.datamodel import conversation_context


@dataclass
class PlotSpec:
    "Specification for a plot."

    # Code to produce the plot.
    code: str

    # PNG as a bytes IO thing.
    plot: BytesIO


def make_plot(what: conversation_context) -> PlotSpec:
    "Create a plot from the conversation context."
    dataset = what.ds.name
    variable = what.what_to_plot
    print(f"Creating plot for dataset: {dataset}, variable: {variable}")

    plot_source = r"""
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

import awkward as ak
import matplotlib.pyplot as plt
import hist
import os
from pathlib import Path

# Define the dataset
ds_name = ("mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE"
    ".e8514_e8528_a911_s4114_r15224_r15225_p6697")

# Create ServiceX dataset and source running on PHYSLITE.
phys_lite_base = FuncADLQueryPHYSLITE()

# Query: get all jet pT
jet_pts_query = phys_lite_base.SelectMany(lambda evt: evt.Jets()).Select(
    lambda jet: {"jet_pt": jet.pt() / 1000.0}  # Default to plotting GeV not MeV
)

# Do the fetch
all_jet_pts = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="jet_data",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=jet_pts_query,
                )
            ]
        ),
        servicex_name="servicex-release-prod",
    )
)

# Create a histogram
h = hist.Hist.new.Reg(100, 0, 500, name="jet_pt", label="Jet $p_T$ [GeV]").Double()
h.fill(jet_pt=all_jet_pts["jet_data"].jet_pt)

# Plot
fig, ax = plt.subplots()
h.plot(ax=ax)
ax.set_title("Jet $p_T$ for all jets")
plt.show()
"""

    plot_path = Path(__file__).parent / "test.png"
    with open(plot_path, "rb") as f:
        plot_bytes = BytesIO(f.read())

    return PlotSpec(code=plot_source, plot=plot_bytes)
