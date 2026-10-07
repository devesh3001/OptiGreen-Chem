# Dataset selection for OptiGreen-Chem (Stage 2)

## What the architecture needs

The Stage-1 design (forecast → risk → MILP → cost/CO₂ → Monte Carlo) needs, for one
coherent supply chain:

| # | Requirement | Used by |
|---|---|---|
| R1 | Real demand by product × region, long enough to hold out years | forecasting |
| R2 | Chemical / process-industry products in mass or volume units | whole project |
| R3 | Real production sites with capacity and throughput | MILP, risk |
| R4 | Real storage / terminal nodes with inventory | MILP |
| R5 | Real disruption events to learn from | risk models, evaluation |
| R6 | Real routes / modes (pipeline, ship, barge, truck) | MILP, CO₂ |
| R7 | Open licence, scriptable download | reproducibility |

## Candidates examined

| Dataset | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Verdict |
|---|---|---|---|---|---|---|---|---|
| Synthetic generator (Stage 1) | ✗ invented | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | test fixture only |
| **DataCo Smart Supply Chain** (Constante et al. 2019; proposed earlier) | partial: retail orders, ~2.5 items per region-category-day after filtering | ✗ clothing, sports, electronics | ✗ none (plants invented) | ✗ | partial: late-delivery flag | partial: shipping class only | ✓ CC BY 4.0 | rejected as primary |
| SupplyGraph (Wasi et al. 2024, FMCG Bangladesh) | partial: 40 products, national only, 221 days | partial: edible oils/flour in tonnes | partial: 25 SAP plant codes, no capacity/location | partial: storage codes | partial: unmet orders | ✗ | ✓ | good GNN benchmark, too short and no geography |
| USAID SCMS delivery history | ✗ lumpy procurement | ✓ pharmaceuticals (kg) | partial: manufacturing sites | partial | ✓ late deliveries | partial: mode + freight | ✓ public domain | risk-only candidate |
| **EIA petroleum bulk file (PET.zip)** | ✓ state × product monthly sales 1983–2022 | ✓ refined products (gasoline, distillate, jet, residual, propane) | ✓ 12 refining districts: capacity, utilisation, crude runs, yields 1985–2026 | ✓ sub-PADD stocks | ✓ outages visible in utilisation (Katrina, Harvey, Uri, Ida, refinery fires) | ✓ inter-PADD flows by pipeline / tanker / barge | ✓ public domain, 55 MB, no key | **selected** |

## Why EIA is the optimum for this project

* It is the only open dataset where demand, production plants, terminals, inter-regional
  routes by mode, trade, prices **and** disruption events all describe the *same* physical
  supply chain.
* Refined petroleum products are core chemical-engineering products; refinery yields,
  densities, process-fuel use and CO₂ intensity can be derived from the data itself.
* Every layer of the Stage-1 architecture maps 1:1:

| Architecture element | Stage 1 (synthetic) | Stage 2 (EIA, real) |
|---|---|---|
| Plants | 4 invented | 12 EIA refining districts (operable capacity, monthly) |
| Warehouses | 8 invented | 10 bulk-terminal hubs on EIA stock areas (1A, 1B, 1C, P2 … P5) |
| Regions | 20 invented | 50 states + DC + 7 withheld-volume pools |
| Products | 5 invented | gasoline, distillate, jet fuel, residual fuel oil, propane |
| Routes | random | inter-PADD lanes with observed flows, Colonial tariff, sea/river distances |
| Disruptions | demand > P90 proxy | district outages detected from utilisation |
| Emissions | invented factors | refinery fuel use (EIA) × EPA factors; Cefic/ECTA transport factors |

## What is still assumed (stated in the report)

Unit costs not published by EIA: refining opex ($5/bbl), tanker day rate, barge rate,
terminal storage fee, shortage and contract penalties. Coordinates are approximate
population / refinery-cluster centres. Ethanol is assumed available at terminals (E10),
with the blend uplift estimated from EIA ethanol and natural-gasoline inputs.

## Access

```
https://www.eia.gov/opendata/bulk/PET.zip      # 55.5 MB, public domain
python -m optigreen.data.eia PET.zip --out data/real/eia
```
The extracted tables (≈2.6 MB, gzip CSV) are committed under `data/real/eia/`, with
`series_used.csv.gz` listing every EIA series id that was read (890 series).
