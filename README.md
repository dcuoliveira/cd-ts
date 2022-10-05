# cd-ts

## Installing conda env

1) Create conda env with python, R and dependencies

    `bash setup_dependencies.sh`

[comment]: <> (## Procedure)

[comment]: <> (1&#41; Install all packages)

[comment]: <> (`bash install.sh`)


[comment]: <> (2&#41; Run tests)

[comment]: <> (`bash test.sh`)


[comment]: <> (3&#41; If needed, run the script to sample the google trends data)

[comment]: <> (`cd src`)

[comment]: <> (`python3 get_trends.py`)


[comment]: <> (4&#41; After we gather multiples samples from gtrends, we combine all of them)

[comment]: <> (by taking the mean and creating the file `data\gtrends.csv`:)

[comment]: <> (`cd src`)

[comment]: <> (`python3 get_trends.py`)


[comment]: <> (5&#41; If needed, run the script to create sector time series:)

[comment]: <> (`cd src`)

[comment]: <> (`python3 create_sectors.py`)


[comment]: <> (6&#41; Run scripts for feature selection)

[comment]: <> (`cd src`)

[comment]: <> (`python3 run_sfi.py`)

[comment]: <> (`python3 run_mdi.py`)

[comment]: <> (`python3 run_mda.py`)

[comment]: <> (`python3 run_granger.py`)

[comment]: <> (`python3 run_huang.py`)

[comment]: <> (`python3 run_IAMB.py`)

[comment]: <> (`python3 run_MMMB.py`)


[comment]: <> (7&#41; Run script for forecast based on one feature selection method and one machine learning model. For example:)

[comment]: <> (`cd src`)

[comment]: <> (`python3 forecast.py "SPX Utilities" MMMB random_forest -i 1 -s 2 -j 2)

[comment]: <> (`)