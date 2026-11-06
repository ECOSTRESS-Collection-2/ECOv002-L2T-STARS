from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pytest
from rasters import Raster
from modland import generate_modland_grid

from ECOv002_L2T_STARS.VIIRS.VIIRSDataPool import (
    VIIRSGranule,
    parse_VIIRS_build,
    parse_VIIRS_date,
    parse_VIIRS_product,
    parse_VIIRS_tile,
)
from ECOv002_L2T_STARS.VIIRS.VNP09GA import VNP09GAGranule
from ECOv002_L2T_STARS.VIIRS.VNP43IA4 import VNP43IA4, VNP43IA4Granule
from ECOv002_L2T_STARS.VIIRS.VNP43MA3 import VNP43MA3, VNP43MA3Granule
from ECOv002_L2T_STARS.VNP43NRT.VNP43NRT import VNP43NRTGranule


def test_parse_viirs_helpers():
    filename = "VNP43MA3.A2024345.h08v05.001.2024345.h5"

    assert parse_VIIRS_product(filename) == "VNP43MA3"
    assert parse_VIIRS_date(filename) == date(2024, 12, 10)
    assert parse_VIIRS_tile(filename) == "h08v05"
    assert parse_VIIRS_build(filename) == 1


def test_viirs_granule_paths_and_metadata(tmp_path):
    filename = tmp_path / "VNP43MA3.A2024345.h08v05.001.h5"
    filename.write_bytes(b"")

    granule = VIIRSGranule(
        filename=str(filename),
        working_directory=str(tmp_path),
        products_directory=str(tmp_path / "products"),
    )

    assert granule.tile == "h08v05"
    assert granule.h == 8
    assert granule.v == 5
    assert granule.date_UTC.date() == date(2024, 12, 10)
    assert granule.product_directory("albedo") == str(tmp_path / "products" / "albedo" / "2024.12.10")
    assert granule.product_filename("albedo").endswith("VNP43MA3.A2024345.h08v05.001_albedo.tif")


def test_vnp43nrt_granule_tracks_expected_outputs(tmp_path):
    granule_directory = tmp_path / "VNP43NRT_A2024345_h08v05"
    granule_directory.mkdir()

    required_files = [
        "VNP43NRT_A2024345_h08v05_NDVI.tif",
        "VNP43NRT_A2024345_h08v05_NBAR_I1.tif",
        "VNP43NRT_A2024345_h08v05_NBAR_I2.tif",
        "VNP43NRT_A2024345_h08v05_albedo.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M1.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M1.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M2.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M2.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M3.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M3.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M4.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M4.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M5.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M5.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M7.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M7.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M8.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M8.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M10.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M10.tif",
        "VNP43NRT_A2024345_h08v05_BSA_M11.tif",
        "VNP43NRT_A2024345_h08v05_WSA_M11.tif",
    ]

    for filename in required_files:
        (granule_directory / filename).touch()

    granule = VNP43NRTGranule(str(granule_directory))

    assert granule.granule_ID == "VNP43NRT_A2024345_h08v05"
    assert granule.tile == "h08v05"
    assert granule.h == 8
    assert granule.v == 5
    assert granule.date_UTC.date() == date(2024, 12, 10)
    assert granule.complete is True
    assert granule.variable_filename("NDVI").endswith("VNP43NRT_A2024345_h08v05_NDVI.tif")


def test_vnp09ga_cloud_mask_from_qf1(tmp_path):
    filename = tmp_path / "VNP09GA.A2024345.h08v05.002.h5"

    with h5py.File(filename, "w") as f:
        dataset = f.create_dataset(
            "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SurfReflect_QF1_1",
            data=np.array([[0, 4], [0, 12]], dtype=np.uint16),
            shape=(2, 2),
        )
        dataset.attrs["_FillValue"] = 65535

    granule = VNP09GAGranule(
        filename=str(filename),
        working_directory=str(tmp_path),
        products_directory=str(tmp_path / "products"),
    )

    mask = granule.get_cloud_mask(target_shape=(2, 2))

    assert mask.shape == (2, 2)
    assert np.asarray(mask).tolist() == [[False, True], [False, True]]


def test_vnp43ia4_granule_reflectance_and_ndvi(tmp_path):
    filename = tmp_path / "VNP43IA4.A2024345.h08v05.001.h5"

    with h5py.File(filename, "w") as f:
        red = f.create_dataset(
            "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Nadir_Reflectance_I1",
            data=np.array([[100, 200], [300, 400]], dtype=np.uint16),
        )
        red.attrs["_FillValue"] = 32767

        nir = f.create_dataset(
            "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Nadir_Reflectance_I2",
            data=np.array([[200, 300], [400, 500]], dtype=np.uint16),
        )
        nir.attrs["_FillValue"] = 32767

        qa = f.create_dataset(
            "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/BRDF_Albedo_Band_Mandatory_Quality_I1",
            data=np.array([[0, 0], [0, 0]], dtype=np.uint8),
        )
        qa.attrs["_FillValue"] = 255

        qa_2 = f.create_dataset(
            "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/BRDF_Albedo_Band_Mandatory_Quality_I2",
            data=np.array([[0, 0], [0, 0]], dtype=np.uint8),
        )
        qa_2.attrs["_FillValue"] = 255

    granule = VNP43IA4Granule(
        filename=str(filename),
        working_directory=str(tmp_path),
        products_directory=str(tmp_path / "products"),
    )

    red_image = granule.reflectance(1, apply_QA=False, save_data=False)
    nir_image = granule.reflectance(2, apply_QA=False, save_data=False)

    assert np.allclose(np.asarray(red_image), np.array([[0.01, 0.02], [0.03, 0.04]]))
    assert np.allclose(np.asarray(nir_image), np.array([[0.02, 0.03], [0.04, 0.05]]))
    expected_ndvi = (np.asarray(nir_image) - np.asarray(red_image)) / (np.asarray(nir_image) + np.asarray(red_image))
    assert np.allclose(np.asarray(granule.NDVI), expected_ndvi)


def test_vnp43ia4_product_mosaics_tiles(monkeypatch, tmp_path):
    geometry = generate_modland_grid(8, 5, 2)
    source = Raster(np.ones((2, 2), dtype=float), geometry=geometry)

    class DummyGranule:
        def __init__(self):
            self.albedo = source

        def product(self, product):
            return Raster(np.full((2, 2), 0.5, dtype=float), geometry=geometry)

    monkeypatch.setattr("ECOv002_L2T_STARS.VIIRS.VNP43IA4.find_modland_tiles", lambda boundary: ["h08v05"])
    monkeypatch.setattr(VNP43IA4, "granule", lambda self, date_UTC, tile, download_location=None, build=None: DummyGranule())

    connection = VNP43IA4(working_directory=str(tmp_path))
    result = connection.product("red", date_UTC=date(2024, 12, 10), geometry=geometry, save_data=False)

    assert result.shape == (2, 2)
    assert np.allclose(np.asarray(result), 0.5)


def test_vnp43ma3_albedo_mosaic(monkeypatch, tmp_path):
    geometry = generate_modland_grid(8, 5, 2)
    source = Raster(np.full((2, 2), 0.25, dtype=float), geometry=geometry)

    class DummyGranule:
        @property
        def albedo(self):
            return source

    monkeypatch.setattr("ECOv002_L2T_STARS.VIIRS.VNP43MA3.find_modland_tiles", lambda boundary: ["h08v05"])
    monkeypatch.setattr(VNP43MA3, "granule", lambda self, date_UTC, tile, download_location=None, build=None: DummyGranule())
    monkeypatch.setattr(Raster, "to_geometry", lambda self, geometry, resampling=None: Raster(np.full((2, 2), 0.25, dtype=float), geometry=geometry))

    connection = VNP43MA3(working_directory=str(tmp_path))
    result = connection.albedo(date_UTC=date(2024, 12, 10), geometry=geometry, filename=None)

    assert result.shape == (2, 2)
    assert np.allclose(np.asarray(result), 0.25)
