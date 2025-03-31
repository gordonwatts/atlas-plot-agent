# Design Thoughts for v0.1

For the initial plan, lets make this pretty directed:

1. Get the user to ask something about plotting something from some data file
1. Check the data file makes sense
1. Work with the user to build the code to generate the plot
1. Return the plot to the user.

Lets, for now, say the workflow is fixed. Later we can let it be more flexible once we have this working. Design data structures and hand offs with this in mind.

## Initial Question

Lets see if we can build enough infrastructure to get this to work:

| `Plot the jet pT for all jets in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697`
