# Share of women in local government - Data package

This data package contains the data that powers the chart ["Share of women in local government"](https://ourworldindata.org/grapher/share-of-women-in-local-government?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on December 12, 2024.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## 5.5.1 - Proportion of elected seats held by women in deliberative bodies of local government (%) - SG_GEN_LOCGELS
Proportion of elected seats held by women in deliberative bodies of local government (%)
Last updated: August 27, 2024  
Next update: August 2026  
Date range: 2007–2023  
Unit: %  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Inter-Parliamentary Union – processed by Our World in Data

#### Full citation
Inter-Parliamentary Union – processed by Our World in Data. “5.5.1 - Proportion of elected seats held by women in deliberative bodies of local government (%) - SG_GEN_LOCGELS” [dataset]. Inter-Parliamentary Union, “Data from multiple sources” [original data].
Source: Inter-Parliamentary Union – processed by Our World In Data

### What you should know about this data

### How is this data described by its producer - Inter-Parliamentary Union?
Proportion of elected seats held by women in deliberative bodies of local government (%)

Further information available at: https://unstats.un.org/sdgs/metadata/files/Metadata-05-05-01a.pdf and https://unstats.un.org/sdgs/metadata/files/Metadata-05-05-01b.pdf

### Source

#### Inter-Parliamentary Union – Data from multiple sources
Retrieved on: 2024-08-27  
Retrieved from: https://unstats.un.org/sdgs/dataportal  


    