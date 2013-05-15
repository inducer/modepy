#! /bin/sh
set -e
OUTF=../modepy/quadrature/xg_quad_data.py
echo "# GENERATED, DO NOT EDIT" > $OUTF
echo "# Xiao-Gimbutas quadratures" >> $OUTF
echo "# http://dx.doi.org/10.1016/j.camwa.2009.10.027" >> $OUTF
echo "import numpy" >> $OUTF
echo "" >> $OUTF
python weights-from-txt.py ~/tmp/triasymq/triasymq_table.txt triangle_table >> $OUTF
echo "" >> $OUTF
python weights-from-txt.py ~/tmp/triasymq/tetraarbq_table.txt tetrahedron_table >> $OUTF
