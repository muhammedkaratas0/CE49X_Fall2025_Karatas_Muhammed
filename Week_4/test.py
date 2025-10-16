1 import pandas as pd
2
3 # Top 3 states by area
4 area = pd . Series ({ ’Alaska ’: 1723337 , ’Texas ’: 695662 ,
5 ’California ’: 423967})
6 # Top 3 states by population
7 population = pd . Series ({ ’California ’: 38332521 , ’Texas ’: 26448193 ,
8 ’New York ’: 19651127})
9
10 # Division aligns indices automatically !
11 density = population / area
12 print ( density )
13 # Alaska NaN (no population data )
14 # California 90.413926
15 # New York NaN (no area data )
16 # Texas 38.018740
17
18 # Result contains UNION of indices
19 # Missing values filled with NaN