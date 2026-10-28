# Sources and indicator notes

## Primary data source

- World Bank Indicators API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
- API pattern: `https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start}:{end}&format=json&per_page=1000`
- Data retrieval date is shown in the app. The app requests calendar years from current year minus 70 through current year minus 1 and reports the actual non-missing coverage returned by the API.

## Country codes used

| Display | World Bank country code |
|---|---|
| Mexico | MEX |
| Brazil | BRA |
| Argentina | ARG |
| Colombia | COL |
| Chile | CHL |
| Peru | PER |
| Venezuela | VEN |
| Ecuador | ECU |
| Uruguay | URY |

Brazil is included as a Latin American comparison country.

## Indicator mappings

| Indicator | Code | Unit / interpretation |
|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | Years |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | Percent of labor force |
| GDP per capita | `NY.GDP.PCAP.CD` | Current US dollars per person; not average wealth or household income |
| Crude birth rate | `SP.DYN.CBRT.IN` | Births per 1,000 people per year; a rate, not a count |
| Net migration | `SM.POP.NETM` | Net number of migrants; not an emigration count |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | Per 100,000 people |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | Annual percent |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | Percent of GDP |
| Total reserves including gold | `FI.RES.TOTL.CD` | Current US dollars |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | Percent of GDP |

Indicator definitions and metadata are supplied by the World Bank. Values may be revised and years can be missing. The app does not impute missing observations.
