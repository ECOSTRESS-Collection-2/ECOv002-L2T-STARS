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
