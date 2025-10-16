import numpy as np
2
3 # Random data
4 L = np . random . random (100)
5
6 # Summary statistics
7 print ( np . sum( L)) # Sum of all values
8 print ( np . min( L)) # Minimum value
9 print ( np . max( L)) # Maximum value
10 print ( np . mean (L)) # Mean
11 print ( np . std ( L)) # Standard deviation
12 print ( np . var ( L)) # Variance
13
14 # These also work as array methods :
15 print (L. sum () )
16 print (L. min () )
17 print (L. max () )
18 print (L. mean () )
19 print (L. std () )
20
21 # Percentiles
22 print ( np . percentile (L , 25) ) # 1st quartile
23 print ( np . median (L )) # 50 th percentile
24 print ( np . percentile (L , 75) ) # 3rd quartile