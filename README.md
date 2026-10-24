# MMA-3001
MMA 3001 project - MSI occupancy and environment 
# Smart Building Occupancy Detection (MMA3001)

## Project Overview
This repository contains a computational engineering pipeline for predicting room occupancy using non-intrusive environmental sensors (Temperature, Humidity, CO2, etc.). It addresses the physical limitations of gas decay by mapping multi-class occupancy states into a binary (Occupied/Empty) target to optimize HVAC energy consumption.

## Repository Structure
* `data/`: Contains the raw environmental and occupancy CSV datasets (Excluded via .gitignore).
* `src/model.py`: The primary machine learning pipeline (Data wrangling, JSON extraction, Dummy Baseline, and Random Forest classification).
* `src/test_model.py`: Pytest unit tests for validating the JSON parsing logic.
* `docs/`: Auto-generated HTML documentation of the source code.

## Dependencies and Environment
* Python 3.12+
* Pandas
* Scikit-Learn
* Pytest
* Pdoc

## Reproducing the Workflow
1. Place the raw `5EnvSensor_MayToDec2024_180kRows.csv`, `5occupancySensor_MayToDec2024_9MRows.csv`, and the mapping Excel file into the `data/` directory.
2. Run `pytest src/test_model.py` to verify the JSON parser integrity.
3. Run `python src/model.py` to execute the data wrangling proximity merge and output the binary Random Forest classification report.

## License
MIT License