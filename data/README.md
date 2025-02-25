# Source Data
## Crossings Data
- Total Accumulated transits by Market Segment and Lock Type FY 2024 
- https://pancanal.com/en/statistics/

## Length Data
- Synthetic Data is created based on possible lengths for each ship.
- Data is generated based on the market segement each ship belongs too. 
- Ships repersent underlying proportions of crossings data.
- Max Panamax Canal Size 294 meters and NeoPanamax 366.
- Each market segment is assigned a range of possible lengths based on these factors:
  - Maximum length of ship passing through each canal type.
  - Sizes have increased over time to improve economies of scale. (Increase in variance)
  - Average Ships sizes based on their trade good.
- https://api.pageplace.de/preview/DT0400.9781000831177_A44431776/preview-9781000831177_A44431776.pdf

# Proccessing
- **Data Loading**: Loads ship transit data and length ranges from CSV files.
- **Proportional Sampling**: Samples ship lengths based on the proportions of different ship types and their canal categories.
- **Length Generation**: Generates random ship lengths within specified ranges for each ship type and canal category.
- **DataFrame Output**: Compiles the generated ship lengths into a pandas DataFrame.
- **Run the Script**: Execute the script to generate ship lengths and print the resulting DataFrame.