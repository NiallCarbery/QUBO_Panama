# Source Data
## Crossings Data
- Total Accumulated transits by Market Segment and Lock Type FY 2024 
- https://pancanal.com/en/statistics/

## Length Data
- Synthetic Data is created based on possible lengths for each ship.
- Data is generated based on the market segement each ship belongs too. 
- Ships repersent underlying proportions of crossings data.
- Max Panamax Canal Size 294 meters and NeoPanamax 366 meters.
- Each market segment is assigned a range of possible lengths based on these factors:
  - Maximum length of ship passing through each canal type.
  - Sizes have increased over time to improve economies of scale. (Increase in variance)
  - Average Ships sizes based on their trade good.
- https://api.pageplace.de/preview/DT0400.9781000831177_A44431776/preview-9781000831177_A44431776.pdf

## Benefit Factor
Assigning Factor Scores to Ship Types
Based on industry knowledge and the Panama Canal's priorities, we assign factor scores to each ship type.

1. Container Ships
SI: 8 (High importance in global trade)

EV: 9 (High value cargo)

CP: 5 (Mixed cargo perishability)

EI: 5 (Moderate environmental impact)

2. Chemical Tankers
SI: 7 (Transport of essential chemicals)

EV: 8 (Valuable cargo)

CP: 5 (Some chemicals are time-sensitive)

EI: 4 (Potential environmental risk)

3. Liquefied Petroleum Gas (LPG) Carriers
SI: 8 (Energy supply importance)

EV: 7 (Valuable but less than crude oil)

CP: 2 (Low perishability)

EI: 7 (Cleaner fuel transport)

4. Dry Bulk Carriers
SI: 6 (Transport of raw materials)

EV: 4 (Lower value per ton)

CP: 1 (Non-perishable)

EI: 5 (Moderate impact)

5. Vehicle Carriers/RoRo
SI: 7 (Automotive industry support)

EV: 8 (High value cargo)

CP: 3 (Low perishability)

EI: 6 (Efficient operations)

6. Refrigerated Ships
SI: 9 (Food supply chain importance)

EV: 8 (High value goods)

CP: 10 (Highly perishable)

EI: 5 (Additional energy use for refrigeration)

7. Crude Product Tankers
SI: 9 (Critical energy resource)

EV: 7 (High value cargo)

CP: 1 (Non-perishable)

EI: 3 (High environmental risk)

8. General Cargo Ships
SI: 6 (Varied importance)

EV: 6 (Moderate value)

CP: 5 (Mixed perishability)

EI: 5 (Moderate impact)

9. Passenger Ships
SI: 5 (Tourism importance)

EV: 5 (Revenue from passengers)

CP: 7 (Time-sensitive schedules)

EI: 4 (High emissions per passenger)

10. Liquefied Natural Gas (LNG) Carriers
SI: 10 (High strategic importance)

EV: 8 (High value cargo)

CP: 2 (Low perishability)

EI: 8 (Transport of clean fuel)