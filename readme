# CSCI4120_finalproj

In this project, we will do legalization and detailed placement of standard cell design with multi-row
height cells. Given a netlist and a global placement result of n cells and a placement region with
tracks and cell sites, the problem is to place all the cells properly in rows and aligning them with the
cell sites such that the total Manhattan displacement from the given global placement positions is
minimized.

In the given netlist, there will be two kinds of cells, 8T and 12T. The 8T ones have a height of 8
tracks while the 12T ones have a height of 12 tracks. There are constraints in the final placement
that (i) each row can only start at tracks numbered multiples of 4 and (ii) each row is either 8T or
12T. An 8T row can only accommodate 8T cells while a 12T row can only accommodate 12T cells.

Input:
- a netlist of n cells
- size of the whole placement region H×W where H is the total number of tracks and W is the
total number of cell sites in one row
- the position (xi, yi) of the lower left corner of each cell i in the given global placement
solution where xi and yi for i = 1…n are floats
- the height (8T or 12T) and width (in number of cell sites) of each cell
- a displacement upper bound B
- track height and site width

Output:
- number of rows m
- the starting track number of each row, which must be a multiple of 4, and the type of the
row (8T or 12T)
- the new position (pi, qi) of the lower left corner of each cell i where pi is the starting track
number of a row (so pi is a multiple of 4) with the same type (8T or 12T) as cell i and qi is a
non-negative integer less than W.

Objective: Minimize the total Manhattan displacement of all the cells.

Constraint:
The resulting placement should be non-overlapping, and all the cells are placed within the
region of H×W in rows of their respective type without overflow.

For most of the cases, the number of cells is below 500k.


The detail placement script is run as follows:

The basic usage is:
    python detail_placer.py --input <input_file> --output <solution_file>