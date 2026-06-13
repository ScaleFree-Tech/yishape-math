use strict;
use warnings;

my $java_file = 'src/main/java/com/yishape/lab/math/autodiff/graph/GraphOpSchema.java';
my $rust_file = 'E:/rust_work/yishape_math_rust/src/graph.rs';

open(my $fh, '<', $java_file) or die "Cannot open $java_file: $!";
my $java_content = do { local $/; <$fh> };
close($fh);

open($fh, '<', $rust_file) or die "Cannot open $rust_file: $!";
my $rust_content = do { local $/; <$fh> };
close($fh);

# Extract HPC_PATTERN tags - look for all strings between "HPC_PATTERN" and the closing "));"
my %hpc_tags;
if ($java_content =~ m/HPC_PATTERN\s+=\s+Collections\.unmodifiableSet.*?Arrays\.asList\(\n(.*?)^\s+\)\);/sm) {
    my $str = $1;
    while ($str =~ /"(\w+)"/g) { $hpc_tags{$1} = 1; }
}
# Also try: look for HPC_PATTERN section more broadly
if (!%hpc_tags) {
    if ($java_content =~ m/HPC_PATTERN.*?\n(.*?)\n\s+\)\);/sm) {
        my $str = $1;
        while ($str =~ /"(\w+)"/g) { $hpc_tags{$1} = 1; }
    }
}

# Extract GPU_PATTERN tags
my %gpu_tags;
if ($java_content =~ m/GPU_PATTERN\s+=\s+Collections\.unmodifiableSet.*?Arrays\.asList\(\n(.*?)^\s+\)\);/sm) {
    my $str = $1;
    while ($str =~ /"(\w+)"/g) { $gpu_tags{$1} = 1; }
}
if (!%gpu_tags) {
    if ($java_content =~ m/GPU_PATTERN.*?\n(.*?)\n\s+\)\);/sm) {
        my $str = $1;
        while ($str =~ /"(\w+)"/g) { $gpu_tags{$1} = 1; }
    }
}

# Debug: print what we found
print "HPC_PATTERN found: " . scalar(keys %hpc_tags) . " tags\n";
print "GPU_PATTERN found: " . scalar(keys %gpu_tags) . " tags\n";

# If still empty, try line-by-line approach
if (!%hpc_tags) {
    my $in_hpc = 0;
    for my $line (split /\n/, $java_content) {
        if ($line =~ /HPC_PATTERN\s*=/) { $in_hpc = 1; }
        if ($in_hpc) {
            while ($line =~ /"(\w+)"/g) { $hpc_tags{$1} = 1; }
            if ($line =~ /\)\)/) { $in_hpc = 0; }
        }
    }
}
if (!%gpu_tags) {
    my $in_gpu = 0;
    for my $line (split /\n/, $java_content) {
        if ($line =~ /GPU_PATTERN\s*=/) { $in_gpu = 1; }
        if ($in_gpu) {
            while ($line =~ /"(\w+)"/g) { $gpu_tags{$1} = 1; }
            if ($line =~ /\)\)/) { $in_gpu = 0; }
        }
    }
}

print "After fallback - HPC_PATTERN: " . scalar(keys %hpc_tags) . " tags\n";
print "After fallback - GPU_PATTERN: " . scalar(keys %gpu_tags) . " tags\n";

# Extract Rust forward tags
my %rust_fwd;
my $in_forward = 0;
my $brace_depth = 0;
for my $line (split /\n/, $rust_content) {
    if ($line =~ /fn forward_compute\b/) { $in_forward = 1; }
    if ($in_forward) {
        if ($line =~ /"([^"]+)"\s*=>/) { $rust_fwd{$1} = 1; }
        $brace_depth += ($line =~ /{/g) - ($line =~ /}/g);
        if ($brace_depth <= 0 && $in_forward && $line =~ /^\}$/) {
            $in_forward = 0;
        }
    }
}

# Extract Rust backward tags
my %rust_bw;
my $in_backward = 0;
$brace_depth = 0;
for my $line (split /\n/, $rust_content) {
    if ($line =~ /fn backward_compute\b/) { $in_backward = 1; }
    if ($in_backward) {
        if ($line =~ /"([^"]+)"\s*=>/) { $rust_bw{$1} = 1; }
        $brace_depth += ($line =~ /{/g) - ($line =~ /}/g);
        if ($brace_depth <= 0 && $in_backward && $line =~ /^\}$/) {
            $in_backward = 0;
        }
    }
}

# Check HPC_PATTERN
print "\n=== HPC_PATTERN tags missing from Rust forward ===\n";
my $missing_fwd = 0;
for my $tag (sort keys %hpc_tags) {
    unless ($rust_fwd{$tag}) {
        print "  MISSING FORWARD: $tag\n";
        $missing_fwd++;
    }
}
print "  (none)\n" if $missing_fwd == 0;

print "\n=== HPC_PATTERN tags missing from Rust backward ===\n";
my $missing_bw = 0;
for my $tag (sort keys %hpc_tags) {
    unless ($rust_bw{$tag}) {
        print "  MISSING BACKWARD: $tag\n";
        $missing_bw++;
    }
}
print "  (none)\n" if $missing_bw == 0;

# Check GPU_PATTERN
print "\n=== GPU_PATTERN tags missing from Rust forward ===\n";
my $gpu_missing_fwd = 0;
for my $tag (sort keys %gpu_tags) {
    unless ($rust_fwd{$tag}) {
        print "  MISSING FORWARD: $tag\n";
        $gpu_missing_fwd++;
    }
}
print "  (none)\n" if $gpu_missing_fwd == 0;

print "\n=== GPU_PATTERN tags missing from Rust backward ===\n";
my $gpu_missing_bw = 0;
for my $tag (sort keys %gpu_tags) {
    unless ($rust_bw{$tag}) {
        print "  MISSING BACKWARD: $tag\n";
        $gpu_missing_bw++;
    }
}
print "  (none)\n" if $gpu_missing_bw == 0;

print "\n=== Summary ===\n";
my $hpc_count = scalar(keys %hpc_tags);
my $gpu_count = scalar(keys %gpu_tags);
my $fwd_count = scalar(keys %rust_fwd);
my $bw_count = scalar(keys %rust_bw);
print "HPC_PATTERN: $hpc_count tags\n";
print "GPU_PATTERN: $gpu_count tags\n";
print "Rust forward dispatch: $fwd_count tags\n";
print "Rust backward dispatch: $bw_count tags\n";
