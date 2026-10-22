# Modelling decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10-10 | Treat the RN datasheet IL curves as CM (asymmetrical) although labelled "sym". | Level at 10 kHz matches CM inductance (about 29 dB for 47 mH in 100 Ω), while the leakage inductance would give about 1 dB. Peak frequencies imply about 10 pF across LN, typical of CM parasitics. |
| 2026-10-10 | Neglect YG (Cg = 0). | Follows the paper. YG depends on mounting and is conservative to ignore. |
| 2026-10-10 | Fit magnitudes only. | The paper fits magnitudes only and recovers phase correctly because the topology is physics based. Datasheets provide magnitudes only. |
| 2026-10-10 | Add jωCg to all four modal admittances, not only to YG. | Cg from every port to ground adds a multiple of the identity to the port Y matrix and shifts every eigenvalue. Table II lists it under G only. Irrelevant while Cg = 0, as in the paper, but needed for consistency if Cg is used later. |
| 2026-10-10 | Interpret CISPR 17 "50 Ω/50 Ω sym" as a 50 Ω differential source and load. | Eq. (2) matches a floating 50 Ω differential termination exactly and does not match SDD21 at 50 Ω per port. Matters when comparing with standard mixed-mode S parameters from a 4-port VNA. |
| 2026-10-10 | Keep the circuit topology in one element netlist (`circuit.py`). | The same netlist is verified against the modal formulas and later written to LTspice and Simscape, so the exported models are the verified circuit. |
| 2026-10-10 | Swap the L and C columns of the LCR_W network in Table III (EPCOS L = 0.3 µH, C = 9.1 pF; WE L = 0.3 µH, C = 8.2 pF; R unchanged). | The printed values (8.2 µH, 0.3 pF) shift the first WD peak to 11 MHz and remove the sym peak near 57 MHz, which contradicts Figs. 11 to 15, in particular Fig. 15 where the curves with and without LCR_W coincide below 40 MHz. A free fit of the tank to Fig. 15 gave L = 0.31 µH, C = 8.1 pF, R = 745 Ω, which is the swapped set. With the swap all paper-style plots match the figures. Printed values are kept in the JSON as `lcr_w_as_printed`. |
| 2026-10-11 | Fit in Python on the closed-form expressions, use LTspice only for validation and application. | The formulas are exact for the circuit and orders of magnitude faster. LTspice in the loop adds no accuracy. |
| 2026-10-11 | One data-agnostic estimator with priors instead of separate procedures. | CM IL constrains only YC. DM parameters must come from priors until measurements exist, and new data types should plug in without changing the pipeline. |
| 2026-10-11 | Model must work in `.ac` and `.tran`, with DCR, saturation, temperature and uncertainty. | Two use cases, filter insertion loss and converter conducted emissions. Restricting the netlist to R, L, C and coupling keeps it passive and stable in transient. |
| 2026-10-11 | Model variation through physical latent variables. | Independent tolerances on each element produce unphysical combinations, since one permeability change moves LC, the RL branches and core losses together. |
| 2026-10-11 | Layout-dependent effects out of scope for this project for now. | To be discussed separately. |
