# Comprehensive field-to-getter replacement for RereDiffTensor privatization
# READ patterns (field used as expression, not assignment)
s/\.value\.totalSize()/.value().totalSize()/g
s/\.value\.toDoubleArray()/.value().toDoubleArray()/g
s/\.value\.getStorageData()/.value().getStorageData()/g
s/\.value\.isContiguous()/.value().isContiguous()/g
s/\.value\.offset()/.value().offset()/g
s/\.value\.shape()/.value().shape()/g
s/\.inputs !=/.inputs() !=/g
s/\.inputs ==/.inputs() ==/g
s/\.inputs\.size()/.inputs().size()/g
s/\.inputs\.get(/.inputs().get(/g
s/\.inputs\.isEmpty()/.inputs().isEmpty()/g
s/\.inputs,/.inputs(),/g
s/\.inputs;/.inputs();/g
s/\.opTag !=/.opTag() !=/g
s/\.opTag ==/.opTag() ==/g
s/\.opTag\.equals/.opTag().equals/g
s/\.opTag)/.opTag())/g
s/\.opTag,/.opTag(),/g
s/\.opTag;/.opTag();/g
s/!\.isLeaf\b/.isLeaf()/g
s/\.isLeaf &&/.isLeaf() \&\&/g
s/\.isLeaf ||/.isLeaf() ||/g
s/\.isLeaf)/.isLeaf())/g
s/\.isLeaf,/.isLeaf(),/g
s/\.isLeaf;/.isLeaf();/g
s/\.scalarParam !=/.scalarParam() !=/g
s/\.scalarParam ==/.scalarParam() ==/g
s/\.scalarParam)/.scalarParam())/g
s/\.scalarParam,/.scalarParam(),/g
s/\.scalarParam;/.scalarParam();/g
s/\.scalarParam2 !=/.scalarParam2() !=/g
s/\.scalarParam2 ==/.scalarParam2() ==/g
s/\.scalarParam2)/.scalarParam2())/g
s/\.scalarParam2,/.scalarParam2(),/g
s/\.scalarParam2;/.scalarParam2();/g
s/\.grad !=/.gradData() !=/g
s/\.grad ==/.gradData() ==/g
s/\.grad)/.gradData())/g
s/\.grad,/.gradData(),/g
s/\.grad;/.gradData();/g
s/\.grad\[/.gradData()[/g
s/\.grad\.clone()/.gradData().clone()/g
s/\.grad\.length/.gradData().length/g
s/\.backwardFn !=/.backwardFn() !=/g
s/\.requiresGrad &&/.requiresGrad() \&\&/g
s/\.requiresGrad ||/.requiresGrad() ||/g
s/\.requiresGrad)/.requiresGrad())/g
s/\.requiresGrad,/.requiresGrad(),/g
s/\.requiresGrad;/.requiresGrad();/g
s/\.symbolicBackwardFn !=/.symbolicBackwardFn() !=/g
s/\.exportShape)/.exportShape())/g
s/\.exportShape,/.exportShape(),/g
s/\.exportShape;/.exportShape();/g
s/\.backwardIndices)/.backwardIndices())/g
s/\.backwardIndices,/.backwardIndices(),/g
s/\.backwardIndices;/.backwardIndices();/g
s/\.backwardIndices\./.backwardIndices()./g
# WRITE patterns (assignment)
s/\.value = \([^;]*\);/\.setValue(\1);/g
s/\.inputs = \([^;]*\);/\.setInputs(\1);/g
s/\.opTag = \([^;]*\);/\.setOpTag(\1);/g
s/\.isLeaf = \([^;]*\);/\.setIsLeaf(\1);/g
s/\.scalarParam = \([^;]*\);/\.setScalarParam(\1);/g
s/\.scalarParam2 = \([^;]*\);/\.setScalarParam2(\1);/g
s/\.grad = \([^;]*\);/\.setGradData(\1);/g
s/\.backwardFn = \([^;]*\);/\.setBackwardFn(\1);/g
s/\.requiresGrad = \([^;]*\);/\.setRequiresGrad(\1);/g
s/\.exportShape = \([^;]*\);/\.setExportShape(\1);/g
s/\.backwardIndices = \([^;]*\);/\.setBackwardIndices(\1);/g
s/\.symbolicBackwardFn = \([^;]*\);/\.setSymbolicBackwardFn(\1);/g
