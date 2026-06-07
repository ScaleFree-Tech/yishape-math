# Read patterns for value, inputs
s/\.value\.totalSize()/.value().totalSize()/g
s/\.value\.toDoubleArray()/.value().toDoubleArray()/g
s/\.value\.getStorageData()/.value().getStorageData()/g
s/\.value\.isContiguous()/.value().isContiguous()/g
s/\.value\.offset()/.value().offset()/g
s/\.value\.shape()/.value().shape()/g
s/\.inputs\.size()/.inputs().size()/g
s/\.inputs\.get(/.inputs().get(/g
s/\.inputs\.isEmpty()/.inputs().isEmpty()/g
s/\.inputs\.clear()/.inputs().clear()/g
# for (X x : node.inputs) -> for (X x : node.inputs())
s/for (RereDiffTensor \([a-zA-Z]*\) : \([a-zA-Z]*\)\.inputs)/for (RereDiffTensor \1 : \2.inputs())/g
s/for (RereDiffTensor \([a-zA-Z]*\) : \([a-zA-Z]*\)\.inputs /for (RereDiffTensor \1 : \2.inputs() /g
# opTag reads
s/\.opTag !=/.opTag() !=/g
s/\.opTag ==/.opTag() ==/g
s/\.opTag )/.opTag() )/g
s/\.opTag,/.opTag(),/g
s/\.opTag;/.opTag();/g
# isLeaf reads
s/\.isLeaf !=/.isLeaf() !=/g
s/\.isLeaf ==/.isLeaf() ==/g
s/\.isLeaf )/.isLeaf() )/g
s/\.isLeaf,/.isLeaf(),/g
s/\.isLeaf;/.isLeaf();/g
# scalarParam reads
s/\.scalarParam !=/.scalarParam() !=/g
s/\.scalarParam ==/.scalarParam() ==/g
s/\.scalarParam )/.scalarParam() )/g
s/\.scalarParam,/.scalarParam(),/g
s/\.scalarParam;/.scalarParam();/g
# scalarParam2 reads
s/\.scalarParam2 !=/.scalarParam2() !=/g
s/\.scalarParam2 ==/.scalarParam2() ==/g
s/\.scalarParam2 )/.scalarParam2() )/g
s/\.scalarParam2,/.scalarParam2(),/g
s/\.scalarParam2;/.scalarParam2();/g
# backwardFn
s/\.backwardFn !=/.backwardFn() !=/g
# symbolicBackwardFn
s/\.symbolicBackwardFn !=/.symbolicBackwardFn() !=/g
s/\.symbolicBackwardFn / .symbolicBackwardFn() /g
# grad reads
s/\.grad !=/.gradData() !=/g
s/\.grad ==/.gradData() ==/g
s/\.grad,/.gradData(),/g
s/\.grad;/.gradData();/g
s/\.grad\[/.gradData()[/g
s/\.grad\.clone()/.gradData().clone()/g
s/\.grad\.length/.gradData().length/g
# requiresGrad reads
s/!\.requiresGrad\b/.requiresGrad()/g
s/ && \([a-zA-Z]*\)\.requiresGrad\b/ \&\& \1.requiresGrad()/g
s/\.requiresGrad )/.requiresGrad() )/g
s/\.requiresGrad,/.requiresGrad(),/g
s/\.requiresGrad;/.requiresGrad();/g
# Assignment patterns (handled later separately)
