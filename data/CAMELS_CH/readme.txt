# -----------------------------------------------------------------------------------------
CAMELS-CH: hydrometeorological time series and landscape attributes for 331 catchments in hydrologic Switzerland
December 2023

Authors: marvin.hoege@eawag.ch; martina.kauzlaric@giub.unibe.ch; rosi.siber@eawag.ch; ursula.schoenenberger@eawag.ch; pascal.horton@giub.unibe.ch; 
	jan.schwanbeck@giub.unibe.ch; sibylle.wilhelm@giub.unibe.ch; daniel.viviroli@geo.uzh.ch; anna.senoner@c2sm.ethz.ch; floriancic@ifu.baug.ethz.ch; 
	manuela.brunner@slf.ch; sandra.pool@eawag.ch; massimiliano.zappa@wsl.ch; fabrizio.fenicia@eawag.ch

Contact: marvin.hoege@eawag.ch

# -----------------------------------------------------------------------------------------

Folders and content:

- timeseries: daily time series of all hydrometeorological variables in subfolders
	- observation_based: all variables based on observations by BAFU, MeteoSwiss, SLF
	- simulation_based: all variables based on simulations from Prevah and MeteoGrids

- static_attributes: basin attributes in ten categories (observation-based) and two subfolders:
	- simulation_based: hydrologic and climatic attributes based on simulated_based time series
	- supplements: supplementary attributes about soil and geology

- annual_timeseries: landcover and glacier information in annual resolution per catchment

- catchment_delineations: shape files of all basins of hydrologic Switzerland

- glacier_data_original: GLAMOS-based annual time series of area and volume per glacier within Switzerland.
	    Glacier Inventory (GI; Paul et al. 2011, 2020)-based annual time series of area per glacier in France, Italy and France

# -----------------------------------------------------------------------------------------

Specific catchment information:

- for rainfall-runoff analysis, we recommend to exclude the following basins:

	* gauge_ids 2446 (Zihlkanal) and 2447 (Canal_de_la_Broye), both at Lac de Neuchâtel: discharge timeseries partially comprise negative values as both refer to channels with bidirectional flow between lakes. 
	Depending on the water level changes in these lakes, the flow direction in the channels might change as indicated by a positive or negative values in the time series.

	* gauge_id 2327 (Dischmabach): discharge values since 1999 show irregularities and should be used with caution, for details see https://zenodo.org/communities/dischma/

	* the timeseries of the 33 lakes in the dataset

- CAMELS-CH covers 298 rivers/streams and 33 lakes within hydrologic Switzerland. Geographically, the allocations of items are:
	* Switzerland:  196 rivers/streams and 33 lakes
	* Austria:	 33 rivers/streams
	* France:	 32 rivers/streams
	* Germany:	 26 rivers/streams
	* Italy:	 11 rivers/streams

- all catchment IDs within political Switzerland follow the BAFU terminology (four digits: 2xxx). 
  Time series files that refer to specific basins, contain the corresponding ID, e.g. 2004, 2007, ...
  basin IDs in neighboring countries follow 3xxx (Austria), 4xxx (Germany), 5xxx (France) and 6xxx (Italy)

- 15 out of the 33 Austrian catchments coincide with catchments from the LamaH data set (Klingler et al., 2021) with gauge_ids:
	3001,3004,3006,3007,3008,3009,3012,3014,3015,3019,3023,3028,3031,3032,3033


