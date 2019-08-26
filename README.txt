COORDS_FIG1
xy coordinates for organoids depicted in Fig 1

COORD_Tumor_Day6
xy coordinates for all organoids

IMAGES_FIG1
Paired DIC and K14 images for organoids depicted in Fig 1

IMAGES_Tumor_Day6_DIC_K14
Paired DIC and K14 images for all organoids

LICENSE.txt
Open source license

OUTPUT_128
Saved analysis for all organoids using 128 boundary points

OUTPUT_256
Saved analysis for all organoids using 256 boundary points

OUTPUT_ALL
Minimal output for all organoids (organoid_table.txt) results of association tests performed by run_analysis.sh

OUTPUT_FIG1
Result of running run_fig1.sh, spectral analysis of organoid boundaries and extraction of K14 pixels

README.txt 
This file

Within src_ibis:

Veena_Matches.txt
Matching between image file names and coordinate file names

Veena_Orig.txt
Matching between image files names and annotations

analyze_results.py
Within-tumor and between-tumor association tests

ibis2d.py
Calculation of spectral power and extraction of K14 pixels

organoid_wc.txt
Number of organoids for each tumor

run_128.sh
Driver for OUTPUT_128

run_256.sh
Driver for OUTPUT_256

run_analysis.sh
Driver for analyze_results.py

run_fig1.sh
Driver for ibis2d.py
