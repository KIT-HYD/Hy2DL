"""
Mask xarray dataset using a shapefile [closed]
https://stackoverflow.com/a/64587946/9940782
"""
from typing import Tuple, Union
import geopandas as gpd
import xarray as xr
import rasterio as rio
from pathlib import Path
import numpy as np
import rioxarray
from tqdm import tqdm


def prepare_rio_data(
    ds: xr.Dataset, gdf: gpd.GeoDataFrame, lat_dim: str = "lat", lon_dim: str = "lon",
) -> Tuple[xr.Dataset, gpd.GeoDataFrame]:
    #  https://gis.stackexchange.com/q/328128/123489
    ds = ds.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)
    ds = ds.rio.write_crs("epsg:4326")

    gdf = gdf.to_crs("epsg:4326")

    return ds, gdf


def rasterize_all_geoms(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    id_column: str,
    shape_dimension: str = "station_id",
    geometry_column: str = "geometry",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    #  TODO: how to ensure prepare_rio_data() has been run? ds.rio.transform() to be correct
    all_shape_masks = []
    pbar = tqdm(gdf.iterrows(), desc="Rasterising Geometry: ")

    # for each row in GeoDataFrame (:: shapely.Polygon)
    for _, shape_row in pbar:
        object_id = shape_row[id_column]
        pbar.set_postfix_str(f"{object_id}")

        #  rasterize into a boolean mask (:: np.ndarray)
        shape_mask = rio.features.geometry_mask(
            [shape_row[geometry_column]],
            out_shape=(len(ds[lat_dim]), len(ds[lon_dim])),
            transform=ds.rio.transform(),
            all_touched=True,
            invert=True,
        )
        #  convert to xr.Dataarray (label the lat, lon, shape_dimension, dimensions)
        shape_mask = xr.DataArray(shape_mask, dims=(lat_dim, lon_dim))
        shape_mask = shape_mask.assign_coords({shape_dimension: object_id}).expand_dims(
            shape_dimension
        )

        if shape_mask.mean() > 0:
            all_shape_masks.append(shape_mask)
        else:
            print(f"No data for basin {object_id}")

    masks = xr.concat(all_shape_masks, dim=shape_dimension)
    return masks


def create_timeseries_of_masked_datasets(
    ds: xr.Dataset,
    masks: xr.Dataset,
    shape_dimension: str = "station_id",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    use_pbar: bool = True,
) -> xr.Dataset:
    all_mean_datasets = []

    def _one_shape_mean(
        masks: xr.Dataset,
        object_id: str,
        shape_dimension: str,
        lat_dim: str,
        lon_dim: str,
    ) -> xr.Dataset:
        mask = masks.sel({shape_dimension: object_id})
        mean_object = ds.where(mask).mean(dim=[lat_dim, lon_dim])
        mean_object = mean_object.expand_dims(shape_dimension)
        return mean_object

    if use_pbar:
        pbar = tqdm(masks[shape_dimension].values, desc="Chopping ROI: ")
        #  iterate over each mask calculating the mean pixel values in polygon
        for object_id in pbar:
            pbar.set_postfix_str(f"{object_id}")
            mean_object = _one_shape_mean(
                masks=masks,
                object_id=object_id,
                shape_dimension=shape_dimension,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )
            all_mean_datasets.append(mean_object)
    else:
        for object_id in masks[shape_dimension].values:
            mean_object = _one_shape_mean(
                masks=masks,
                object_id=object_id,
                shape_dimension=shape_dimension,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )
            all_mean_datasets.append(mean_object)

    out_ds = xr.concat(all_mean_datasets, dim=shape_dimension)
    return out_ds


def create_camels_basin_timeseries(
    data_obj: Union[Path, xr.Dataset],
    shp_data_dir: Path,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    # 1. Load in shapefile and dataset
    shp = shp_data_dir
    gdf = gpd.read_file(shp)
    if isinstance(data_obj, Path):
        ds = xr.open_dataset(data_obj)
    elif isinstance(data_obj, xr.Dataset):
        ds = data_obj
    else:
        assert False, "Expect Path or xr.Dataset object"

    # 2. Ensure that data properly initialised (e.g. CRS is the same)
    ds, gdf = prepare_rio_data(ds, gdf, lat_dim=lat_dim, lon_dim=lon_dim)
    id_column: str = "ID_STRING"
    shape_dimension: str = "station_id"

    # 3. Create xarray shape masks
    masks = rasterize_all_geoms(
        ds=ds,
        gdf=gdf,
        id_column=id_column,
        shape_dimension=shape_dimension,
        geometry_column="geometry",
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )

    # 4. Create timeseries of mean values
    out_ds = create_timeseries_of_masked_datasets(
        ds=ds,
        masks=masks,
        shape_dimension=shape_dimension,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )

    return out_ds


if __name__ == "__main__":
    data_dir = Path("/datadrive/data")
    shp_data_dir = data_dir / "CAMELS_GB_DATASET"
    path_to_sm_data = data_dir / "esa_cci_sm_gb.nc"

    out_ds = create_camels_basin_timeseries(
        data_obj=path_to_sm_data, shp_data_dir=shp_data_dir
    )

    # save the catchment averaged timeseries of soil moisture
    out_ds.to_netcdf(data_dir / "camels_basin_ESACCI_sm.nc")
