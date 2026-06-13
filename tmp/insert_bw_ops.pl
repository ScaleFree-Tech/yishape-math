use strict;
use warnings;
use utf8;

my $path = "src/graph.rs";
open(my $fh, "<:encoding(UTF-8)", $path) or die "Cannot open $path: $!";
my @lines = <$fh>;
close($fh);

# Find insertion point: line with Ok(vec![ga, gb]) then } then "dot" => {
my $insert_idx;
for my $i (0..$#lines-3) {
    if ($lines[$i] =~ /Ok\(vec!\[ga, gb\]\)/ && $lines[$i+1] =~ /^\s*\}\s*$/ && $lines[$i+2] =~ /"dot" => \{/) {
        $insert_idx = $i + 2;
        last;
    }
}

unless (defined $insert_idx) {
    die "ERROR: insertion point not found\n";
}

print "Inserting at line $insert_idx: $lines[$insert_idx]";

my @bw_lines = (
    "        \"geluSum\" => {\n",
    "            let a = inputs.first().ok_or(\"geluSum: missing input\")?;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| { let t = (0.7071067811865476 * v).tanh(); g * 0.5 * v * (1.0 - t * t) * 0.7071067811865476 + 0.5 * (1.0 + t) }).collect()])\n",
    "        }\n",
    "        \"geluMean\" => {\n",
    "            let a = inputs.first().ok_or(\"geluMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| { let t = (0.7071067811865476 * v).tanh(); g * 0.5 * v * (1.0 - t * t) * 0.7071067811865476 + 0.5 * (1.0 + t) }).collect()])\n",
    "        }\n",
    "        \"sinSum\" => {\n",
    "            let a = inputs.first().ok_or(\"sinSum: missing input\")?;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| g * v.cos()).collect()])\n",
    "        }\n",
    "        \"sinMean\" => {\n",
    "            let a = inputs.first().ok_or(\"sinMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| g * v.cos()).collect()])\n",
    "        }\n",
    "        \"cosSum\" => {\n",
    "            let a = inputs.first().ok_or(\"cosSum: missing input\")?;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| -g * v.sin()).collect()])\n",
    "        }\n",
    "        \"cosMean\" => {\n",
    "            let a = inputs.first().ok_or(\"cosMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| -g * v.sin()).collect()])\n",
    "        }\n",
    "        \"leakyReluSum\" => {\n",
    "            let a = inputs.first().ok_or(\"leakyReluSum: missing input\")?;\n",
    "            if !has_scalar { return Err(\"leakyReluSum: missing scalar\".into()); }\n",
    "            let alpha = scalar;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha }).collect()])\n",
    "        }\n",
    "        \"leakyReluMean\" => {\n",
    "            let a = inputs.first().ok_or(\"leakyReluMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            if !has_scalar { return Err(\"leakyReluMean: missing scalar\".into()); }\n",
    "            let alpha = scalar;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha }).collect()])\n",
    "        }\n",
    "        \"eluSum\" => {\n",
    "            let a = inputs.first().ok_or(\"eluSum: missing input\")?;\n",
    "            if !has_scalar { return Err(\"eluSum: missing scalar\".into()); }\n",
    "            let alpha = scalar;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha * v.exp() }).collect()])\n",
    "        }\n",
    "        \"eluMean\" => {\n",
    "            let a = inputs.first().ok_or(\"eluMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            if !has_scalar { return Err(\"eluMean: missing scalar\".into()); }\n",
    "            let alpha = scalar;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha * v.exp() }).collect()])\n",
    "        }\n",
    "        \"seluSum\" => {\n",
    "            let a = inputs.first().ok_or(\"seluSum: missing input\")?;\n",
    "            let scale = 1.05070098;\n",
    "            let alpha = 1.67326324;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g * scale } else { g * scale * alpha * v.exp() }).collect()])\n",
    "        }\n",
    "        \"seluMean\" => {\n",
    "            let a = inputs.first().ok_or(\"seluMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            let scale = 1.05070098;\n",
    "            let alpha = 1.67326324;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| if *v > 0.0 { g * scale } else { g * scale * alpha * v.exp() }).collect()])\n",
    "        }\n",
    "        \"softplusSum\" => {\n",
    "            let a = inputs.first().ok_or(\"softplusSum: missing input\")?;\n",
    "            if !has_scalar { return Err(\"softplusSum: missing scalar\".into()); }\n",
    "            let beta = scalar;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| { let s = (beta * v).exp(); g * s / (1.0 + s) }).collect()])\n",
    "        }\n",
    "        \"softplusMean\" => {\n",
    "            let a = inputs.first().ok_or(\"softplusMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            if !has_scalar { return Err(\"softplusMean: missing scalar\".into()); }\n",
    "            let beta = scalar;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| { let s = (beta * v).exp(); g * s / (1.0 + s) }).collect()])\n",
    "        }\n",
    "        \"hardtanhSum\" => {\n",
    "            let a = inputs.first().ok_or(\"hardtanhSum: missing input\")?;\n",
    "            if !has_scalar { return Err(\"hardtanhSum: missing scalar\".into()); }\n",
    "            let max_val = scalar;\n",
    "            let min_val = param2;\n",
    "            let g = grad[0];\n",
    "            Ok(vec![a.iter().map(|v| if *v >= max_val || *v <= min_val { 0.0 } else { g }).collect()])\n",
    "        }\n",
    "        \"hardtanhMean\" => {\n",
    "            let a = inputs.first().ok_or(\"hardtanhMean: missing input\")?;\n",
    "            let n = a.len() as f64;\n",
    "            if !has_scalar { return Err(\"hardtanhMean: missing scalar\".into()); }\n",
    "            let max_val = scalar;\n",
    "            let min_val = param2;\n",
    "            let g = grad[0] / n;\n",
    "            Ok(vec![a.iter().map(|v| if *v >= max_val || *v <= min_val { 0.0 } else { g }).collect()])\n",
    "        }\n",
);

# Insert in reverse order
for my $line (reverse @bw_lines) {
    splice(@lines, $insert_idx, 0, $line);
}

open($fh, ">:encoding(UTF-8)", $path) or die "Cannot write $path: $!";
print $fh @lines;
close($fh);

print "[OK] Inserted " . scalar(@bw_lines) . " lines at line $insert_idx\n";
