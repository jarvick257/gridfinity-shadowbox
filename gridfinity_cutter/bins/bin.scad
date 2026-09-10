// gridfinity-cutter wrapper around Gridfinity Rebuilt (vendored submodule).
//
// Renders a bin from the same customizer variables as the library's
// gridfinity-rebuilt-bins.scad (overridden from Python with -D), translated so
// x/y run from the bin's grid corner (gridfinity_cutter.bins convention) and
// with machine-readable echoes. The object pocket is NOT cut here; the Python
// side subtracts it from the rendered STL.

include <gridfinity-rebuilt-openscad/src/core/standard.scad>
use <gridfinity-rebuilt-openscad/src/core/gridfinity-rebuilt-utility.scad>
use <gridfinity-rebuilt-openscad/src/core/gridfinity-rebuilt-holes.scad>
use <gridfinity-rebuilt-openscad/src/core/bin.scad>
use <gridfinity-rebuilt-openscad/src/core/cutouts.scad>
use <gridfinity-rebuilt-openscad/src/helpers/generic-helpers.scad>
use <gridfinity-rebuilt-openscad/src/helpers/grid.scad>
use <gridfinity-rebuilt-openscad/src/helpers/grid_element.scad>

$fa = 4;
$fs = 0.25;

/* [General Settings] */
gridx = 1;
gridy = 1;
gridz = 3;
half_grid = false;

/* [Height] */
gridz_define = 0; // [0:7mm increments - Excludes Stacking Lip, 1:Internal mm - Excludes Base & Stacking Lip, 2:External mm - Excludes Stacking Lip, 3:External mm]
height_internal = 0;
enable_zsnap = false;
include_lip = true;

/* [Compartments] */
divx = 0; // 0 = solid bin (library default is 1)
divy = 0;
depth = 0;

/* [Cylindrical Compartments] */
cut_cylinders = false;
cd = 10;
c_chamfer = 0.5;

/* [Compartment Features] */
style_tab = 1; // [0:Full,1:Auto,2:Left,3:Center,4:Right,5:None]
place_tab = 0; // [0:Everywhere-Normal,1:Top-Left Division]
scoop = 1;

/* [Base Hole Options] */
only_corners = false;
refined_holes = true;
magnet_holes = false;
screw_holes = false;
crush_ribs = true;
chamfer_holes = true;
printable_hole_top = true;
enable_thumbscrew = false;

// ===== IMPLEMENTATION (mirrors gridfinity-rebuilt-bins.scad) ===== //

hole_options = bundle_hole_options(refined_holes, magnet_holes, screw_holes, crush_ribs, chamfer_holes, printable_hole_top);
grid_dimensions = GRID_DIMENSIONS_MM / (half_grid ? 2 : 1);
bin_height_mm = height(gridz, gridz_define, enable_zsnap);

bin1 = new_bin(
    grid_size = [gridx, gridy],
    height_mm = bin_height_mm,
    fill_height = height_internal,
    include_lip = include_lip,
    hole_options = hole_options,
    only_corners = only_corners || half_grid,
    thumbscrew = enable_thumbscrew,
    grid_dimensions = grid_dimensions
);

echo(str("GFC_HEIGHT_MM=", bin_height_mm));
echo(str("GFC_INFILL_MM=", bin_get_infill_size_mm(bin1)));
echo(str("GFC_BBOX_MM=", bin_get_bounding_box(bin1)));

translate([gridx * grid_dimensions.x / 2, gridy * grid_dimensions.y / 2, 0])
bin_render(bin1) {
    bin_subdivide(bin1, [divx, divy]) {
        depth_real = cgs(height=depth).z;
        if (cut_cylinders) {
            cut_chamfered_cylinder(cd/2, depth_real, c_chamfer);
        } else {
            cut_compartment_auto(cgs(height=depth), style_tab, place_tab != 0, scoop);
        }
    }
}
