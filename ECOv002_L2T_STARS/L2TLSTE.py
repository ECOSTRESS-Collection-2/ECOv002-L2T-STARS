from __future__ import annotations

import logging
from datetime import datetime
from typing import Union, List

import numpy as np
import rasters as rt
from rasters import Raster

from .ECOSTRESS_granule import ECOSTRESSGranule
from .ECOSTRESS_tiled_granule import ECOSTRESSTiledGranule

logger = logging.getLogger(__name__)

PRODUCT_METADATA_GROUP = "L2 LSTE Metadata"
GRID_NAME = "ECO_L2G_LSTE_70m"

L2G_LSTE_SHORT_NAME = "ECO_L2G_LSTE"
L2G_LSTE_LONG_NAME = "ECOSTRESS Gridded Land Surface Temperature and Emissivity Instantaneous L2 Global 70 m"

L2T_LSTE_SHORT_NAME = "ECO_L2T_LSTE"
L2T_LSTE_LONG_NAME = "ECOSTRESS Tiled Land Surface Temperature and Emissivity Instantaneous L2 Global 70 m"

VARIABLE_NAMES = [
    "water",
    "cloud",
    "view_zenith",
    "height",
    "QC",
    "LST",
    "LST_err",
    "EmisWB"
]

PRIMARY_VARIABLE = "LST"

class L2LSTEGranule(ECOSTRESSGranule):
    _PRIMARY_VARIABLE = PRIMARY_VARIABLE
    _PRODUCT_METADATA_GROUP = PRODUCT_METADATA_GROUP

    def __init__(self, product_filename: str):
        ECOSTRESSGranule.__init__(self, product_filename=product_filename)

        self._LST = None
        self._LST_err = None
        self._EmisWB = None
        self._QC = None
        self._water = None
        self._cloud = None

    @property
    def cloud(self) -> Raster:
        if self._cloud is None:
            self._cloud = self.variable("cloud")

        return self._cloud

    @property
    def LST(self) -> Raster:
        if self._LST is None:
            LST = self.variable("LST")
            LST = rt.where(self.cloud, np.nan, LST)
            self._LST = LST

        return self._LST

    @property
    def ST_K(self) -> Raster:
        return self.LST

    @property
    def ST_C(self) -> Raster:
        return self.ST_K - 273.15

    @property
    def LST_err(self) -> Raster:
        if self._LST_err is None:
            self._LST_err = self.variable("LST_err")

        return self._LST_err

    @property
    def EmisWB(self) -> Raster:
        if self._EmisWB is None:
            self._EmisWB = self.variable("EmisWB")

        return self._EmisWB

    @property
    def emissivity(self) -> Raster:
        return self.EmisWB

    @property
    def QC(self):
        if self._QC is None:
            self._QC = self.variable("QC").astype(np.uint16)

        return self._QC


class L2TLSTE(ECOSTRESSTiledGranule, L2LSTEGranule):
    _PRODUCT_NAME = "L2T_LSTE"
    _PRODUCT_METADATA_GROUP = "ProductMetadata"

    def __init__(
            self,
            product_location: str = None,
            orbit: int = None,
            scene: int = None,
            tile: str = None,
            time_UTC: Union[datetime, str] = None,
            build: str = None,
            process_count: int = None,
            containing_directory: str = None,
            **kwargs):
        L2LSTEGranule.__init__(self, product_filename=product_location)
        ECOSTRESSTiledGranule.__init__(
            self,
            product_location=product_location,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=process_count,
            containing_directory=containing_directory,
            **kwargs
        )

        self._water = None
        self._cloud = None
        self._view_zenith = None
        self._height = None

    @property
    def filename(self) -> str:
        return self.product_location

    @property
    def variables(self) -> List[str]:
        return [
            "LST",
            "LST_err",
            "EmisWB",
            "water",
            "cloud",
            "height",
            "view_zenith",
            "QC"
        ]

    @property
    def water(self) -> Raster:
        if self._water is None:
            self._water = self.variable(variable="water").astype(bool)

        return self._water

    @property
    def cloud(self) -> Raster:
        if self._cloud is None:
            self._cloud = self.variable(variable="cloud").astype(bool)

        return self._cloud

    @property
    def view_zenith(self) -> Raster:
        if self._view_zenith is None:
            self._view_zenith = self.variable(variable="view_zenith")

        return self._view_zenith

    @property
    def height(self) -> Raster:
        if self._height is None:
            self._height = self.variable(variable="height")

        return self._height

    @property
    def elevation_m(self) -> Raster:
        return self.height

    @property
    def elevation_km(self) -> Raster:
        return self.elevation_m / 1000

    # @classmethod
    # def from_scene(
    #         cls,
    #         gridded_granule: ECOSTRESSGriddedGranule,
    #         tile: str,
    #         tile_granule_directory: str = None,
    #         tile_granule_name: str = None,
    #         geometry: RasterGeometry = None,
    #         variables: List[str] = None,
    #         compression: str = None,
    #         overwrite: bool = False,
    #         skip_blank: bool = True) -> L2TLSTE:
    #     if compression is None:
    #         compression = cls._COMPRESSION
    #
    #     if tile_granule_name is None:
    #         tile_granule_name = gridded_granule.tile_granule_name(tile)
    #
    #     if tile_granule_directory is None:
    #         tile_granule_directory = tile_granule_name
    #
    #     logger.info(
    #         f"target granule directory: {cl.dir(tile_granule_directory)}"
    #     )
    #
    #     metadata = gridded_granule.metadata_dict
    #     metadata["StandardMetadata"]["LocalGranuleID"] = f"{tile_granule_name}.zip"
    #     standard_metadata = metadata[cls._STANDARD_METADATA_GROUP]
    #
    #     DATA_FORMAT_TYPE = "COG"
    #     logger.info(f"DataFormatType: {cl.val(DATA_FORMAT_TYPE)}")
    #     standard_metadata["DataFormatType"] = DATA_FORMAT_TYPE
    #
    #     standard_metadata.pop("HDFVersionID")
    #
    #     PROCESSING_LEVEL = "L2T"
    #     logger.info(f"ProcessingLevelID: {cl.val(PROCESSING_LEVEL)}")
    #     standard_metadata["ProcessingLevelID"] = PROCESSING_LEVEL
    #     PROCESSING_LEVEL_DESCRIPTION = "Level 2 Tiled Land Surface Temperature and Emissivity"
    #     logger.info(f"ProcessingLevelDescription: {cl.val(PROCESSING_LEVEL_DESCRIPTION)}")
    #     standard_metadata["ProcessingLevelDescription"] = PROCESSING_LEVEL_DESCRIPTION
    #     SIS_VERSION = "Preliminary"
    #     logger.info(f"SISVersion: {cl.val(SIS_VERSION)}")
    #     standard_metadata["SISVersion"] = SIS_VERSION
    #
    #     short_name = L2T_LSTE_SHORT_NAME
    #     logger.info(f"L2T LSTE short name: {cl.val(short_name)}")
    #     standard_metadata["ShortName"] = short_name
    #
    #     long_name = L2T_LSTE_LONG_NAME
    #     logger.info(f"L2T LSTE long name: {cl.val(long_name)}")
    #     standard_metadata["LongName"] = long_name
    #
    #     logger.info(f"RegionID: {cl.place(tile)}")
    #     standard_metadata["RegionID"] = tile
    #
    #     cell_width = geometry.cell_width
    #     rows, cols = geometry.shape
    #     bbox = geometry.bbox.latlon
    #     x_min, y_min, x_max, y_max = bbox
    #     CRS = geometry.proj4
    #
    #     logger.info(f"ImageLineSpacing: {cl.val(cell_width)}")
    #     standard_metadata["ImageLineSpacing"] = cell_width
    #     logger.info(f"ImagePixelSpacing: {cl.val(cell_width)}")
    #     standard_metadata["ImagePixelSpacing"] = cell_width
    #     logger.info(f"ImageLines: {cl.val(rows)}")
    #     standard_metadata["ImageLines"] = rows
    #     logger.info(f"ImagePixels: {cl.val(cols)}")
    #     standard_metadata["ImagePixels"] = cols
    #     logger.info(f"CRS: {cl.val(CRS)}")
    #     standard_metadata["CRS"] = CRS
    #     logger.info(f"EastBoundingCoordinate: {cl.place(x_max)}")
    #     standard_metadata["EastBoundingCoordinate"] = x_max
    #     logger.info(f"SouthBoundingCoordinate: {cl.place(y_min)}")
    #     standard_metadata["SouthBoundingCoordinate"] = y_min
    #     logger.info(f"WestBoundingCoordinate: {cl.place(x_min)}")
    #     standard_metadata["WestBoundingCoordinate"] = x_min
    #     logger.info(f"NorthBoundingCoordinate: {cl.place(y_max)}")
    #     standard_metadata["NorthBoundingCoordinate"] = y_max
    #
    #     product_metadata = metadata[cls._PRODUCT_METADATA_GROUP]
    #
    #     product_metadata.pop("CloudMaxTemperature")
    #     product_metadata.pop("CloudMeanTemperature")
    #     product_metadata.pop("CloudMinTemperature")
    #     product_metadata.pop("CloudSDevTemperature")
    #     product_metadata.pop("Emis1GoodAvg")
    #     product_metadata.pop("Emis2GoodAvg")
    #     product_metadata.pop("Emis3GoodAvg")
    #     product_metadata.pop("Emis4GoodAvg")
    #     product_metadata.pop("Emis5GoodAvg")
    #     product_metadata.pop("LSTGoodAvg")
    #
    #     orbit = gridded_granule.orbit
    #     scene = gridded_granule.scene
    #     time_UTC = gridded_granule.time_UTC
    #     build = gridded_granule.build
    #     process_count = gridded_granule.process_count
    #
    #     granule = cls(
    #         product_location=tile_granule_directory,
    #         orbit=orbit,
    #         scene=scene,
    #         tile=tile,
    #         time_UTC=time_UTC,
    #         build=build,
    #         process_count=process_count,
    #         compression=compression
    #     )
    #
    #     if variables is None:
    #         output_variables = gridded_granule.tiled_output_variables
    #     else:
    #         output_variables = variables
    #
    #     for j, variable in enumerate(output_variables):
    #         logger.info(f"processing variable: {variable}")
    #         output_filename = join(tile_granule_directory, f"{tile_granule_name}_{variable}.tif")
    #
    #         if exists(output_filename) and not overwrite:
    #             logger.warning(f"file already exists: {cl.file(output_filename)}")
    #             continue
    #
    #         logger.info(
    #             f"started processing variable {variable} ({j + 1} / {len(output_variables)}) "
    #             f"for granule: {tile_granule_name}"
    #         )
    #
    #         timer = Timer()
    #
    #         image = gridded_granule.variable(
    #             variable,
    #             apply_scale=True,
    #             apply_cloud=True,
    #             geometry=geometry
    #         )
    #
    #         if skip_blank and np.all(np.isnan(image)):
    #             raise BlankOutput(f"blank output for layer {variable} at tile {tile} at time {time_UTC}")
    #
    #         granule.add_layer(variable, image)
    #
    #         logger.info(
    #             f"finished processing variable {variable} ({j + 1} / {len(output_variables)}) "
    #             f"for granule: {tile_granule_name} "
    #             f"({cl.time(timer)})"
    #         )
    #
    #     cloud = granule.cloud
    #     cloud_percent = np.count_nonzero(cloud) / cloud.size * 100
    #
    #     logger.info(f"QAPercentCloudCover: {cl.val(cloud_percent)}")
    #     product_metadata["QAPercentCloudCover"] = cloud_percent
    #
    #     ST_K = granule.ST_K
    #     good_percent = np.count_nonzero(~np.isnan(ST_K)) / np.array(ST_K).size * 100
    #     logger.info(f"QAPercentGoodQuality: {cl.val(good_percent)}")
    #     product_metadata["QAPercentGoodQuality"] = good_percent
    #
    #     metadata[cls._STANDARD_METADATA_GROUP] = standard_metadata
    #     metadata[cls._PRODUCT_METADATA_GROUP] = product_metadata
    #     granule.write_metadata(metadata)
    #
    #     return granule



