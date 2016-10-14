# TSDS (Time-Series Data Stack)

Steps:

1. Request for Surface Reflectance data on EarthExplorer
2. Receive email when data is ready
3. Get the raw data from webside like the format 2015.txt
4. use downloadData.py to get tar.gz file
5. use ccUntar.bash to unzip files
6. use ccGetCubeB.py to generate each band's data cube for each year
7. use ccMergeCubeB.py to generate the final datacube of each band
