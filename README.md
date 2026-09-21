# GPS Congestion Detection Framework

Author: Alexander Bousman

This repository contains a GPS-based framework for distinguishing traffic-related slowdowns from bus-operational activity in Niterói, Brazil, using map-matched trajectories, rolling speed estimates, General Transit Feet Specification (GTFS) stop and terminal context, Open Street Map (OSM) data, route information, and spatiotemporal clustering. 

The framework also contains a route-vulnerability analysis to measure which routes are more or less disproportionately exposed to road-network congestion. The framework can be generalized to other bairros/neighborhoods, dates, and time windows in cities where GPS, GTFS, and OSM data are available.

## Data Availability and NDA

This project was developed using data provided through the NetMob 2026 Data Challenge for Niterói, Brazil. Access to the original dataset is subject to a data-use and non-disclosure agreement (NDA), so the raw data, derived records containing confidential observations, and certain project outputs are not included in this repository.

This repository currently contains only materials and methodology that do not disclose the underlying NetMob dataset. Additional code, documentation, and reproducible examples may be released following the applicable publication and data-use requirements of the NetMob 2026 Data Challenge.

Without the NetMob dataset, none of the `cluster_analysis` code will run properly. However, the NetMob dataset itself cannot be redistributed through this repository. The repository is meant to serve as a display of the programmed congestion detection methodology. Researchers interested in accessing the original data should request access directly through the NetMob Data Challenge organizers.

### Repo Outline
1. `cluster_analysis` - Contains the code and methodology for the GPS Congestion Detection Framework
2. `ibge` - Contains pandas/GeoPandas data from the Institutio Brasileiro de Geografia e Estatística used to contextualize the GTFS/OSM within city/state borders
3. `NetMob2026_Submission__Full_.pdf` - The full project submission to the NetMob 2026 Data Challenge
