# ECOSTRESS Collection 2 STARS Data Fusion Product Generating Executable (PGE)

This software package has been refactored from the [ECOSTRESS Collection 2 Gridded & Tiled Products Generator](https://github.com/ECOSTRESS-Collection-2/ECOSTRESS-Collection-2) as an independent product pipeline for the ECOSTRESS Collection 2 STARS data fusion product. This software package will serve as an ECOSTRESS Collection 2 reference for the development of the [ECOSTRESS Collection 3 STARS PGE](https://github.com/ECOSTRESS-Collection-3/ECOv003-L2T-STARS) and the [SBG Collection 1 JET PGE](https://github.com/sbg-tir/SBG-TIR-L2-STARS).

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Margaret C. Johnson](https://github.com/majohnso) (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 398L

[Evan Davis](https://github.com/evandjpl) (he/him)<br>
[evan.w.davis@jpl.nasa.gov](mailto:evan.w.davis@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 397K

[Kerry Cawse-Nicholson](https://github.com/kcawse) (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Claire Villanueva-Weeks](https://github.com/clairesvw) (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Prerequisites

This is a Python package that calls Julia code. Julia must be installed in order to run this package.

## Authentication

This package requires an [EarthData](https://urs.earthdata.nasa.gov/) account and reads EarthData credentials from `~/.netrc` in the following format:

```
machine urs.earthdata.nasa.gov
login <USERNAME>
password <PASSWORD>
```

## Environment

On macOS, there are issues with installing `pykdtree` using pip, so it's better to use a mamba environment and install the `pykdtree` mamba package.

```
mamba create -y -n ECOv002-L2T-STARS -c conda-forge python=3.11 jupyter pykdtree 
mamba activate ECOv002-L2T-STARS
```

## Installation

Install this package from PyPi using the name `ECOv002-L2T-STARS` with dashes:

```
pip install ECOv002-L2T-STARS
```

You can also install development versions of this package directly from a clone of this repository:

```
pip install .
```

## Usage

Import this package with the name `ECOv002_L2T_STARS` with underscores:

```
import ECOv002_L2T_STARS
```
