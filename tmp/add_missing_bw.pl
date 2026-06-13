use strict;
use warnings;

my $rust_file = 'E:/rust_work/yishape_math_rust/src/graph.rs';

open(my $fh, '<', $rust_file) or die "Cannot open $rust_file: $!";
my @lines = <$fh>;
close($fh);

# Find the insertion point: "dot" => { in backward_compute (second occurrence)
my $insert_idx;
my $dot_count = 0;
for my $i (0..$#lines) {
    if ($lines[$i] =~ /"dot" => \{/) {
        $dot_count++;
        if ($dot_count == 2) {
            $insert_idx = $i;
            last;
        }
    }
}

unless (defined $insert_idx) {
    die "ERROR: could not find second 'dot' => {\n";
}

print "Inserting before line $insert_idx: $lines[$insert_idx]";

my @new_lines = (
    "        \"squareSum\" => {\n",
    "            let a = inputs.first().ok_or(\"squareSum: missing input\")?;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| 2.0 * v * g).collect()])\n",
    "        }\n",
    "        \"expSum\" => {\n",
    "            let a = inputs.first().ok_or(\"expSum: missing input\")?;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| v.exp() * g).collect()])\n",
    "        }\n",
);

for my $line (reverse @new_lines) {
    splice(@lines, $insert_idx, 0, $line);
}

open($fh, '>', $rust_file) or die "Cannot write $rust_file: $!";
print $fh @lines;
close($fh);

print "[OK] Inserted " . scalar(@new_lines) . " lines\n";
