# build_basys3.tcl — Vivado batch build script for Basys3
# Usage (from NN_Digits/ directory):
#   vivado -mode batch -source syn/build_basys3.tcl

set project_name "mlp_accel_basys3"
set part         "xc7a35tcpg236-1"
set top          "top_basys3"

# Paths relative to the NN_Digits/ directory
set rtl_dir  "rtl"
set hex_dir  "rtl/hex/thermal"
set syn_dir  "syn"
set out_dir  "build"

file mkdir $out_dir

# Create in-memory project
create_project -in_memory -part $part

# Add RTL sources
add_files [list \
    $rtl_dir/top_basys3.v \
    $rtl_dir/mlp_accel.v \
    $rtl_dir/uart/uart_rx.v \
    $rtl_dir/uart/uart_tx.v \
]

# Add hex init files so $readmemh can find them
add_files [list \
    $hex_dir/weight_l1.hex \
    $hex_dir/weight_l2.hex \
    $hex_dir/weight_l3.hex \
    $hex_dir/bias_l1.hex \
    $hex_dir/bias_l2.hex \
    $hex_dir/bias_l3.hex \
]

# Add constraints
add_files -fileset constrs_1 $syn_dir/basys3.xdc

set_property top $top [current_fileset]

# Synthesise — include both hex dirs so $readmemh resolves correctly
synth_design -top $top -part $part -include_dirs [list $hex_dir "rtl/hex"]
write_checkpoint -force $out_dir/post_synth.dcp

# Implement
opt_design
place_design
route_design
write_checkpoint -force $out_dir/post_route.dcp

# Reports
report_timing_summary -file $out_dir/timing.rpt
report_utilization    -file $out_dir/utilization.rpt

# Bitstream
write_bitstream -force $out_dir/$top.bit

puts "Done — bitstream at $out_dir/$top.bit"
