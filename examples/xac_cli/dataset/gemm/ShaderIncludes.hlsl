//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

// Disable this false warning. Writing this...
//    for (int i = 0; ...; ...)
//    for (int i = 0; ...; ...)
// ...yields this incorrect message (they are not actually nested):
//    loop control variable conflicts with a previous declaration in the outer scope; most recent declaration will be used
#pragma warning(disable: 3078)

#define NCHW_N 0
#define NCHW_K 0

#define NCHW_C 1
#define NCHW_H 2
#define NCHW_W 3

#define NC xy
#define HW zw

#define NCHW_HW_MIN NCHW_H
#define NCHW_HW_MAX NCHW_W

// Newer versions of DXC support short circuiting in logical operators:
// https://github.com/microsoft/DirectXShaderCompiler/wiki/HLSL-2021#logical-operation-short-circuiting-for-scalars
//
// The Xbox shader compiler in newer GDKs enforces use of the new intrinsics and will emit errors such as:
// error: condition for short-circuiting ternary operator must be scalar, for non-scalar types use 'select'
// error: operands for short-circuiting logical binary operator must be scalar, for non-scalar types use 'and'
//
// For now, we opt out of the intrinsics on Windows/WSL since we use a mix of FXC and an older DXC without
// HLSL 2021. We should revisit this in the future:
// https://microsoft.visualstudio.com/OS/_workitems/edit/47666195
#if __HLSL_VERSION >= 2021
    #define SELECT(Cond, V1, V2) select((Cond), (V1), (V2))
    #define AND(LHS, RHS) and((LHS), (RHS))
#else
    #define SELECT(Cond, V1, V2) (Cond) ? (V1) : (V2)
    #define AND(LHS, RHS) (LHS) && (RHS)
#endif

#if (__SHADER_TARGET_MAJOR * 100 + __SHADER_TARGET_MINOR) >= 602
    // Older versions of dxc.exe (and all versions of fxc.exe) did not support
    // data types float16/uint16/int16.
    #define SHADER_TARGET_SUPPORTS_NATIVE_16BIT 1
#endif

inline uint GetGlobalGroupIndex(uint3 groupDim, uint3 groupId)
{
    return groupId.z * groupDim.y * groupDim.x
            + groupId.y * groupDim.x
            + groupId.x;
}

inline uint GetGlobalThreadIndex(uint3 groupDim, uint3 groupId, uint3 blockDim, uint groupThreadIndex) 
{
    const uint elementsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    const uint gridXStride = elementsPerBlock;
    const uint gridYStride = groupDim.x * gridXStride;
    const uint gridZStride = groupDim.y * gridYStride;

    const uint globalThreadIndex = groupId.z * gridZStride 
                                    + groupId.y * gridYStride 
                                    + groupId.x * gridXStride
                                    + groupThreadIndex;

    return globalThreadIndex;
}

inline uint GetGlobalDataIndex(uint globalThreadIndex, uint startDataIndex, uint elementsPerThread)
{
    return startDataIndex + globalThreadIndex * elementsPerThread;
}

//
// Calculates the index of the i-th element of a strided tensor.
//
inline uint GetStridedIndex(uint i, uint4 strides, uint4 sizes)
{
    return strides[0] * ((i / (sizes[3] * sizes[2] * sizes[1]))) +
           strides[1] * ((i / (sizes[3] * sizes[2]           )) % sizes[1]) +
           strides[2] * ((i / (sizes[3]                      )) % sizes[2]) +
           strides[3] * ((i                                   ) % sizes[3]);
}

inline uint GetStridedIndex(uint i, uint2 strides, uint2 sizes)
{
    return strides[0] * ((i / (sizes[1]                      )) % sizes[0]) +
           strides[1] * ((i                                   ) % sizes[1]);
}

// Old alias function to GetCoordinatesFromLogicalIndex().
inline uint4 GetNCHWFromIndex(uint globalIndex, uint4 sizes)
{
    uint4 nchw;
    nchw[NCHW_W] = globalIndex % sizes[NCHW_W]; globalIndex /= sizes[NCHW_W];
    nchw[NCHW_H] = globalIndex % sizes[NCHW_H]; globalIndex /= sizes[NCHW_H];
    nchw[NCHW_C] = globalIndex % sizes[NCHW_C];
    nchw[NCHW_N] = globalIndex / sizes[NCHW_C];
    return nchw;
}

uint4 GetNCHWFromIndexInSingleN(uint globalIndex, uint4 sizes)
{
    uint4 nchw;
    nchw[NCHW_W] = globalIndex % sizes[NCHW_W]; globalIndex /= sizes[NCHW_W];
    nchw[NCHW_H] = globalIndex % sizes[NCHW_H]; globalIndex /= sizes[NCHW_H];
    nchw[NCHW_C] = globalIndex;
    nchw[NCHW_N] = 0;    
                        
    return nchw;
}

// Gets coordinate indices within a tensor from a 1D index, while deciding the order to
// walk that tensor based on memory layout detected through a heuristic using strides.
// Choosing the correct memory layout isn't necessary for correctness, but improves
// memory efficiency given certain non-NCHW layouts.
inline uint4 GetNCHWFromIndexInDetectedLayout(uint index, uint4 sizes, uint4 layoutStrides)
{
    bool isNhwcOrNhcw = (layoutStrides[NCHW_C] < max(layoutStrides[NCHW_H], layoutStrides[NCHW_W]));
    bool isNhcw = isNhwcOrNhcw && (layoutStrides[NCHW_C] > layoutStrides[NCHW_W]);

    uint4 sizesInPreferredOrder = sizes;

    [flatten]
    if (isNhcw)
    {
        sizesInPreferredOrder = uint4(sizes[NCHW_N], sizes[NCHW_H], sizes[NCHW_C], sizes[NCHW_W]);
    }
    else if (isNhwcOrNhcw)
    {
        sizesInPreferredOrder = uint4(sizes[NCHW_N], sizes[NCHW_H], sizes[NCHW_W], sizes[NCHW_C]);
    }

    uint4 indices = GetNCHWFromIndex(index, sizesInPreferredOrder);

    [flatten]
    if (isNhcw)
    {
        indices = uint4(indices[NCHW_N], indices[NCHW_H], indices[NCHW_C], indices[NCHW_W]);
    }
    else if (isNhwcOrNhcw)
    {
        indices = uint4(indices[NCHW_N], indices[NCHW_W], indices[NCHW_C], indices[NCHW_H]);
    }

    return indices;
}

inline uint32_t GetCumulativeSum(uint32_t8 values)
{
    uint32_t cumulativeSum;
    cumulativeSum = values[1][3] + values[1][2] + values[1][1] + values[1][0] +
                    values[0][3] + values[0][2] + values[0][1] + values[0][0];
    return cumulativeSum;
}

inline uint32_t GetCumulativeSum(uint32_t4 values)
{
    return values[3] + values[2] + values[1] + values[0];
}

// Compute the cumulative product of an array, where each output index is the product of all values
// after it. So output index 5 is values[6] * values[7], and index 0 is values[1] * values[2] * ...
// In C++, this is equivalent to a descending std::exclusive_scan with std::multiplies<>.
inline uint32_t8 GetCumulativeDescendingProduct(uint32_t8 values)
{
    uint32_t8 cumulativeProduct;
    /*[7]*/ cumulativeProduct[1][3] = 1;
    /*[6]*/ cumulativeProduct[1][2] = values[1][3];//* cumulativeProduct[1][3];
    /*[5]*/ cumulativeProduct[1][1] = values[1][2]   * cumulativeProduct[1][2];
    /*[4]*/ cumulativeProduct[1][0] = values[1][1]   * cumulativeProduct[1][1];
    /*[3]*/ cumulativeProduct[0][3] = values[1][0]   * cumulativeProduct[1][0];
    /*[2]*/ cumulativeProduct[0][2] = values[0][3]   * cumulativeProduct[0][3];
    /*[1]*/ cumulativeProduct[0][1] = values[0][2]   * cumulativeProduct[0][2];
    /*[0]*/ cumulativeProduct[0][0] = values[0][1]   * cumulativeProduct[0][1];
    return cumulativeProduct;
}

inline uint32_t4 GetCumulativeDescendingProduct(uint32_t4 values)
{
    uint32_t4 cumulativeProduct;
    cumulativeProduct[3] = 1;
    cumulativeProduct[2] = values[3];//* cumulativeProduct[3];
    cumulativeProduct[1] = values[2]   * cumulativeProduct[2];
    cumulativeProduct[0] = values[1]   * cumulativeProduct[1];
    return cumulativeProduct;
}

inline uint32_t GetCumulativeProduct(uint32_t8 values)
{
    return GetCumulativeDescendingProduct(values)[0][0] * values[0][0];
}

inline uint32_t GetCumulativeProduct(uint32_t4 values)
{
    return GetCumulativeDescendingProduct(values)[0] * values[0];
}

// Map from the logical element index (not the linear index) to a series of coordinates.
inline uint32_t8 GetCoordinatesFromLogicalIndex(uint globalIndex, uint32_t8 sizes)
{
    // Compute the cumulative product of sizes, where each array index is the product of all sizes
    // after it. So index 5 is sizes[6] * sizes[7], and index 0 is sizes[1] * sizes[2] * ...
    uint32_t8 cumulativeSizeProducts = GetCumulativeDescendingProduct(sizes);
    uint32_t8 coordinates;
    coordinates[0] = globalIndex / cumulativeSizeProducts[0] % sizes[0]; // Write low  uint32_t4 (values 0..3)
    coordinates[1] = globalIndex / cumulativeSizeProducts[1] % sizes[1]; // Write high uint32_t4 (values 4..7)
    return coordinates;
}

inline uint4 GetCoordinatesFromLogicalIndex(uint globalIndex, uint4 sizes)
{
    // Equivalent to globalIndex / GetCumulativeDescendingProduct(sizes) % sizes.

    uint4 nchw;
    nchw[NCHW_W] = globalIndex % sizes[NCHW_W]; globalIndex /= sizes[NCHW_W];
    nchw[NCHW_H] = globalIndex % sizes[NCHW_H]; globalIndex /= sizes[NCHW_H];
    nchw[NCHW_C] = globalIndex % sizes[NCHW_C];
    nchw[NCHW_N] = globalIndex / sizes[NCHW_C];
    return nchw;
}

uint2 GetHWFromIndexInSingleNC(uint globalIndex, uint4 sizes)
{
    uint2 hw;
    hw[1] =  globalIndex % sizes[NCHW_W];
    hw[0] = (globalIndex / sizes[NCHW_W]);
    
    return hw;
}

inline uint2 GetXYFromIndex(uint globalIndex, uint2 sizes)
{
    uint2 xy;
    xy.x =  globalIndex                                    % sizes.x;
    xy.y = (globalIndex / sizes.x)                         % sizes.y;

    return xy;
}   

inline uint4 GetPackedStrides(uint4 sizes)
{
    uint4 strides;
    strides[3] = 1;
    strides[2] = sizes[3]; // * strides[3]
    strides[1] = sizes[2]     * strides[2];
    strides[0] = sizes[1]     * strides[1];
    return strides;
}

inline uint32_t8 GetPackedStrides(uint32_t8 sizes)
{
    return GetCumulativeDescendingProduct(sizes);
}

// Get logical index given coordinates and sizes
uint GetLogicalIndexFromCoordinatesAndSizes(uint4 coordinates, uint4 sizes)
{
    return dot(coordinates, GetPackedStrides(sizes));
}

uint GetLogicalIndexFromCoordinatesAndSizes(uint32_t8 coordinates, uint32_t8 sizes)
{
    uint32_t8 packedStrides = GetPackedStrides(sizes);
    return dot(coordinates[0], packedStrides[0]) + dot(coordinates[1], packedStrides[1]);
}

// Map 4D coordinates to a 1D a logical element index using sizes alone, ignoring strides.
// For packed arrangements, the element index and element offset are the same thing.
inline uint GetIndexFromIndicesAndSizes(uint4 indices, uint4 sizes)
{
    return GetLogicalIndexFromCoordinatesAndSizes(indices, sizes);
}

// Map 8D coordinates to a 1D linear element offset using strides (not byte offset).
inline uint GetOffsetFromCoordinates(uint32_t8 coordinates, uint32_t8 strides)
{
    return dot(coordinates[0], strides[0]) + dot(coordinates[1], strides[1]);
}

// Map 8D coordinates to a 1D linear element offset using strides (not byte offset).
inline uint GetOffsetFromCoordinates(uint32_t4 coordinates, uint32_t4 strides)
{
    return dot(coordinates, strides);
}

// Old alias function for GetOffsetFromCoordinates().
inline uint GetOffsetFromNCHW(uint4 indices, uint4 strides)
{
    return dot(indices, strides);
}

inline uint GetOffsetFromHW(uint2 indices, uint4 strides)
{
    return indices.x * strides[NCHW_H] + indices.y * strides[NCHW_W];
}

inline bool IsNegativeZero(float value)
{
    return asuint(value) == 0x80000000;
}

// get output tensor's coordinates from memory index
// combine with checking whether the corresponding memory index is padded or not
// return true if padded
bool CheckPaddedGetCoordinates(uint memoryIndex, uint4 strides, uint4 sizes, uint dimensionCount, inout uint4 coordinates)
{
    bool overflow = false;
    uint accum = memoryIndex;

    uint coordinatesArray[4] = {0, 0, 0, 0};
    uint stridesArray[4] = {strides[0], strides[1], strides[2], strides[3]};

    for (uint i = dimensionCount; i > 0; i--)
    {
        uint j = i - 1;
        coordinatesArray[j] = accum / stridesArray[j];
        overflow = overflow || (coordinatesArray[j] >= sizes[j]);

        accum %= stridesArray[j];
    }
    bool gap = accum != 0;

    coordinates = uint4(coordinatesArray[0], coordinatesArray[1], coordinatesArray[2], coordinatesArray[3]);

    return overflow || gap;
}

bool CheckPaddedGetCoordinates(uint memoryIndex, uint32_t8 strides, uint32_t8 sizes, uint dimensionCount, inout uint32_t8 coordinates)
{
    bool overflow = false;
    uint accum = memoryIndex;

    uint coordinatesArray[8] =  {0, 0, 0, 0, 0, 0, 0, 0};
    uint stridesArray[8] = {strides[0][0], strides[0][1], strides[0][2], strides[0][3], strides[1][0], strides[1][1], strides[1][2], strides[1][3]};
    uint sizesArray[8] =   {sizes[0][0], sizes[0][1], sizes[0][2], sizes[0][3], sizes[1][0], sizes[1][1], sizes[1][2], sizes[1][3]};

    for (uint i = dimensionCount; i > 0; i--)
    {
        uint j = i-1;
        coordinatesArray[j] = accum / stridesArray[j];
        overflow = overflow || (coordinatesArray[j] >= sizesArray[j]);

        accum %= stridesArray[j];
    }
    bool gap = accum != 0;

    coordinates[0] = uint4(coordinatesArray[0], coordinatesArray[1], coordinatesArray[2], coordinatesArray[3]);
    coordinates[1] = uint4(coordinatesArray[4], coordinatesArray[5], coordinatesArray[6], coordinatesArray[7]);

    return overflow || gap;
}

// revert coordinates from sorting.
void RevertSortedCoordinates(inout uint4 coordinates, uint4 order)
{
    uint tmpArray[4];

    tmpArray[order[3]] = coordinates[3];
    tmpArray[order[2]] = coordinates[2];
    tmpArray[order[1]] = coordinates[1];
    tmpArray[order[0]] = coordinates[0];
    coordinates = uint4(tmpArray[0], tmpArray[1], tmpArray[2], tmpArray[3]);
}

void RevertSortedCoordinates(inout uint32_t8 coordinates, uint32_t8 order)
{
    uint tmpArray[8];

    tmpArray[order[0][0]] = coordinates[0][0];
    tmpArray[order[0][1]] = coordinates[0][1];
    tmpArray[order[0][2]] = coordinates[0][2];
    tmpArray[order[0][3]] = coordinates[0][3];

    tmpArray[order[1][0]] = coordinates[1][0];
    tmpArray[order[1][1]] = coordinates[1][1];
    tmpArray[order[1][2]] = coordinates[1][2];
    tmpArray[order[1][3]] = coordinates[1][3];

    coordinates[0] = uint4(tmpArray[0], tmpArray[1], tmpArray[2], tmpArray[3]);
    coordinates[1] = uint4(tmpArray[4], tmpArray[5], tmpArray[6], tmpArray[7]);
}

#ifdef INDEX_BUFFER_DATA_TYPE
uint HandleNegativeIndex(INDEX_BUFFER_DATA_TYPE rawIndexValue, uint indexLimit)
{
    // If the index value is negative, then treat it as being relative from the end of
    // respective dimension (e.g. the input/output tensor along the current axis).
#if INDEX_BUFFER_DATA_TYPE_int64_emulated
    // Note that UAV's have a limit of 4 billion elements. So any higher bits will be
    // truncated anyway. We only pay attention to the sign here.
    return (uint)(rawIndexValue.low + (rawIndexValue.high < 0 ? indexLimit : 0));
#elif INDEX_BUFFER_DATA_TYPE_uint64_emulated
    // Since uint64 cannot be represented by a uint, clamp it to the max value of its low part.
    return rawIndexValue.high > 0 ? 0xFFFFFFFF : rawIndexValue.low;
#else
    return (uint)(rawIndexValue + (rawIndexValue < 0 ? indexLimit : 0));
#endif
}
#endif

inline int32_t CeilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// Only define this type for operators which explicitly define DIMENSION_COUNT.
#ifdef DIMENSION_COUNT
    #if DIMENSION_COUNT == 8
        typedef uint32_t8 DimensionsType;
        typedef int32_t8 SignedDimensionsType;
    #elif DIMENSION_COUNT == 5 // Round up to 8.
        typedef uint32_t8 DimensionsType;
        typedef int32_t8 SignedDimensionsType;
    #elif DIMENSION_COUNT == 4
        typedef uint32_t4 DimensionsType;
        typedef int32_t4 SignedDimensionsType;
    #else
        #error "Unsupported DIMENSION_COUNT"
    #endif
#endif

// Simple helper to abstract access to the dimensions in the cbuffer so the same logic can
// be shared between uint32_t8 and uint4_t4 shaders. For the 8D, the function is just a nop.
// For the 4D shader, access the beginning 4D (ignoring the latter 4 which are neutral).
#ifdef DIMENSION_COUNT
inline DimensionsType ToDimensionsTypeLeftAligned(uint32_t8 dimensions)
{
#if DIMENSION_COUNT > 4
    return dimensions;
#elif DIMENSION_COUNT == 4
    return dimensions[0];
#endif
}
inline SignedDimensionsType ToSignedDimensionsTypeLeftAligned(int32_t8 dimensions)
{
#if DIMENSION_COUNT > 4
    return dimensions;
#elif DIMENSION_COUNT == 4
    return dimensions[0];
#endif
}

inline DimensionsType ToDimensionsTypeRightAligned(uint32_t8 dimensions)
{
#if DIMENSION_COUNT > 4
    return dimensions;
#elif DIMENSION_COUNT == 4
    return dimensions[1];
#endif
}
inline SignedDimensionsType ToSignedDimensionsTypeRightAligned(int32_t8 dimensions)
{
#if DIMENSION_COUNT > 4
    return dimensions;
#elif DIMENSION_COUNT == 4
    return dimensions[1];
#endif
}
#endif // #ifdef DIMENSION_COUNT

// Simple helper function to abstract between uint32_t8 vs uint4_t4, since operator [] cannot
// work directly on them both consistently. Sadly we can't just access the uint32_t8 array
// using a single bracket (e.g. dimensions[5]) because HLSL vectors/matrices can only range
// in size 1 to 4.
#ifdef DIMENSION_COUNT
inline uint32_t GetDimensionValueLeftAligned(uint32_t8 dimensions, uint32_t axis)
{
#if DIMENSION_COUNT > 4
    return dimensions[axis >> 2][axis & 3];
#elif DIMENSION_COUNT == 4
    // In the 4D shader, only access the first 4D of the 8D dimensions passed in the cbuffer.
    return dimensions[0][axis]; // Axis ranges 0 to 3.
#endif
}

inline uint32_t GetDimensionValueRightAligned(uint32_t8 dimensions, uint32_t axis)
{
#if DIMENSION_COUNT > 4
    return dimensions[axis >> 2][axis & 3];
#elif DIMENSION_COUNT == 4
    // In the 4D shader, only access the first 4D of the 8D dimensions passed in the cbuffer.
    return dimensions[1][axis]; // Axis ranges 0 to 3.
#endif
}

inline uint32_t GetDimensionValueLeftAligned(uint32_t4 dimensions, uint32_t axis)
{
    return dimensions[axis]; // Axis ranges 0 to 3.
}

#if DIMENSION_COUNT > 4
inline void SetDimensionValue(inout uint32_t8 dimensions, uint32_t axis, uint32_t value)
{
    dimensions[axis >> 2][axis & 3] = value;
}
// Works around error X3500: array reference cannot be used as an l-value.
inline void SetDimensionValueConstantIndex(inout uint32_t8 dimensions, uint32_t axis, uint32_t value)
{
    dimensions[0] = SELECT(uint4(0,1,2,3) == (uint4)axis, value, dimensions[0]);
    dimensions[1] = SELECT(uint4(4,5,6,7) == (uint4)axis, value, dimensions[1]);
    
}
#elif DIMENSION_COUNT == 4
inline void SetDimensionValue(inout uint32_t4 dimensions, uint32_t axis, uint32_t value)
{
    dimensions[axis] = value; // Axis ranges 0 to 3.
}
// Works around error X3500: array reference cannot be used as an l-value.
inline void SetDimensionValueConstantIndex(inout uint32_t4 dimensions, uint32_t axis, uint32_t value)
{
    dimensions = SELECT(uint4(0,1,2,3) == (uint4)axis, value, dimensions);
}
#endif

#if DIMENSION_COUNT == 8
inline uint32_t8 ZeroDimensionValue()
{
    uint32_t8 output;
    output[0] = uint4(0,0,0,0);
    output[1] = uint4(0,0,0,0);
    return output;
}
#elif DIMENSION_COUNT == 4
inline uint32_t4 ZeroDimensionValue()
{
    return uint4(0,0,0,0);
}
#endif

#if DIMENSION_COUNT == 8
inline uint32_t8 MultiplyDimensionsType(uint32_t8 input1, uint32_t8 input2)
{
    uint32_t8 output;
    output[0] = input1[0] * input2[0];
    output[1] = input1[1] * input2[1];
    return output;
}
#elif DIMENSION_COUNT == 4
inline uint32_t4 MultiplyDimensionsType(uint32_t4 input1, uint32_t4 input2)
{
    return input1 * input2;
}
#endif
#endif // #ifdef DIMENSION_COUNT

inline bool LogicalEquals(uint64_emulated a, uint64_emulated b)
{
    return a.low == b.low && a.high == b.high;
}

inline bool LogicalEquals(int64_emulated a, int64_emulated b)
{
    return a.low == b.low && a.high == b.high;
}

inline bool LogicalLessThan(uint64_emulated a, uint64_emulated b)
{
    return a.high < b.high || (a.high == b.high && a.low < b.low);
}

inline bool LogicalLessThan(int64_emulated a, int64_emulated b)
{
    return a.high < b.high || (a.high == b.high && a.low < b.low);
}

inline bool LogicalLessThanOrEqual(uint64_emulated a, uint64_emulated b)
{
    return a.high < b.high || (a.high == b.high && a.low <= b.low);
}

inline bool LogicalLessThanOrEqual(int64_emulated a, int64_emulated b)
{
    return a.high < b.high || (a.high == b.high && a.low <= b.low);
}

inline bool LogicalGreaterThan(uint64_emulated a, uint64_emulated b)
{
    return a.high > b.high || (a.high == b.high && a.low > b.low);
}

inline bool LogicalGreaterThan(int64_emulated a, int64_emulated b)
{
    return a.high > b.high || (a.high == b.high && a.low > b.low);
}

inline bool LogicalGreaterThanOrEqual(uint64_emulated a, uint64_emulated b)
{
    return a.high > b.high || (a.high == b.high && a.low >= b.low);
}

inline bool LogicalGreaterThanOrEqual(int64_emulated a, int64_emulated b)
{
    return a.high > b.high || (a.high == b.high && a.low >= b.low);
}

inline uint32_t4 Less(uint32_t4 a, uint32_t4 b)
{
    return a < b;
}

inline uint32_t8 Less(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] < b[0];
    output[1] = a[1] < b[1];
    return output;
}

inline uint32_t4 GreaterOrEqual(uint32_t4 a, uint32_t4 b)
{
    return a >= b;
}

inline uint32_t8 GreaterOrEqual(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] >= b[0];
    output[1] = a[1] >= b[1];
    return output;
}

inline bool Any(uint32_t4 a)
{
    return any(a);
}

inline bool Any(uint32_t8 a)
{
    return any(a[0]) || any(a[1]);
}

inline uint64_emulated CastToUint64(int64_emulated a)
{
    return uint64_emulated::Construct(a.low, a.high);
}

inline int64_emulated CastToInt64(uint64_emulated a)
{
    return int64_emulated::Construct(a.low, a.high);
}

// Identity operation simplifies some generic logic, so one can blindly overload without regard to the type.
inline uint64_emulated CastToUint64(uint64_emulated a)
{
    return a;
}

// Identity operation simplifies some generic logic, so one can blindly overload without regard to the type.
inline int64_emulated CastToInt64(int64_emulated a)
{
    return a;
}

inline uint32_t4 Add(uint32_t4 a, uint32_t4 b)
{
    return a + b;
}

inline uint32_t8 Add(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] + b[0];
    output[1] = a[1] + b[1];
    return output;
}

inline int32_t4 Add(int32_t4 a, int32_t4 b)
{
    return a + b;
}

inline int32_t8 Add(int32_t8 a, int32_t8 b)
{
    int32_t8 output;
    output[0] = a[0] + b[0];
    output[1] = a[1] + b[1];
    return output;
}

inline uint32_t4 Add(uint32_t4 a, uint32_t s)
{
    return a + s;
}

inline uint32_t8 Add(uint32_t8 a, uint32_t s)
{
    uint32_t8 output;
    output[0] = a[0] + s;
    output[1] = a[1] + s;
    return output;
}

inline int32_t4 Add(int32_t4 a, uint32_t s)
{
    return a + s;
}

inline int32_t8 Add(int32_t8 a, uint32_t s)
{
    int32_t8 output;
    output[0] = a[0] + s;
    output[1] = a[1] + s;
    return output;
}

inline uint64_emulated Add(uint64_emulated a, uint64_emulated b)
{
    uint64_emulated c;
    c.low = a.low + b.low;
    c.high = a.high + b.high + (c.low < a.low); // Add with carry.
    return c;
}

inline int64_emulated Add(int64_emulated a, int64_emulated b)
{
    return CastToInt64(Add(CastToUint64(a), CastToUint64(b)));
}

inline int32_t4 Sub(int32_t4 a, int32_t4 b)
{
    return a - b;
}

inline int32_t8 Sub(int32_t8 a, int32_t8 b)
{
    int32_t8 output;
    output[0] = a[0] - b[0];
    output[1] = a[1] - b[1];
    return output;
}

inline uint32_t4 Sub(uint32_t4 a, uint32_t4 b)
{
    return a - b;
}

inline uint32_t8 Sub(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] - b[0];
    output[1] = a[1] - b[1];
    return output;
}

inline uint32_t4 Sub(uint32_t4 a, uint32_t s)
{
    return a - s;
}

inline uint32_t8 Sub(uint32_t8 a, uint32_t s)
{
    uint32_t8 output;
    output[0] = a[0] - s;
    output[1] = a[1] - s;
    return output;
}

inline int32_t4 Sub(int32_t4 a, uint32_t s)
{
    return a - s;
}

inline int32_t8 Sub(int32_t8 a, uint32_t s)
{
    int32_t8 output;
    output[0] = a[0] - s;
    output[1] = a[1] - s;
    return output;
}

inline uint64_emulated Subtract(uint64_emulated a, uint64_emulated b)
{
    uint64_emulated c;
    c.low = a.low - b.low;
    c.high = a.high - b.high - (c.low > a.low); // Subtract with borrow.
    return c;
}

inline int64_emulated Subtract(int64_emulated a, int64_emulated b)
{
    return CastToInt64(Subtract(CastToUint64(a), CastToUint64(b)));
}

inline uint32_t4 Mul(uint32_t4 a, uint32_t4 b)
{
    return a * b;
}

inline uint32_t8 Mul(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] * b[0];
    output[1] = a[1] * b[1];
    return output;
}

inline int32_t4 Mul(int32_t4 a, int32_t4 b)
{
    return a * b;
}

inline int32_t8 Mul(int32_t8 a, int32_t8 b)
{
    int32_t8 output;
    output[0] = a[0] * b[0];
    output[1] = a[1] * b[1];
    return output;
}

inline uint32_t4 Mul(uint32_t4 a, uint32_t s)
{
    return a * s;
}

inline uint32_t8 Mul(uint32_t8 a, uint32_t s)
{
    uint32_t8 output;
    output[0] = a[0] * s;
    output[1] = a[1] * s;
    return output;
}

inline int32_t4 Mul(int32_t4 a, int32_t s)
{
    return a * s;
}

inline int32_t8 Mul(int32_t8 a, int32_t s)
{
    int32_t8 output;
    output[0] = a[0] * s;
    output[1] = a[1] * s;
    return output;
}

inline int64_emulated Negate(int64_emulated a)
{
    int64_emulated result;
    result.low = -((int32_t)a.low);
    result.high = ~a.high + (result.low == 0);
    return result;
}

inline int64_emulated Abs(int64_emulated a)
{
    [flatten]
    if (a.high < 0)
    {
        a = Negate(a);
    }
    return a;
}

inline uint64_emulated Abs(uint64_emulated a)
{
    return a;
}

inline int64_emulated Sign(int64_emulated a) 
{
    int64_emulated sign;
    sign.high = (a.high < 0) ? -1 : 0;
    sign.low = (a.high < 0) ? -1 : !(a.high == 0 && a.low == 0);
    return sign;
}

inline uint64_emulated Sign(uint64_emulated a) 
{
    uint64_emulated sign;
    sign.high = 0;
    sign.low = !(a.high == 0 && a.low == 0);
    return sign;
}

inline uint64_emulated Multiply(uint64_emulated a, uint64_emulated b)
{
    // This follows the same multiplication formula as for standard
    // gradeschool decimal, consecutively multiplying parts by each
    // other from the top and bottom. e.g.
    //
    //  34 * 28 = 952
    //
    //         3   4
    //    x    2   8
    //   -----------
    //       8*3 8*4
    // + 2*3 2*4
    // -------------
    //        24  32
    //     6   8
    // -------------
    //     2   3
    //         4   2
    //     6   8
    // -------------
    //     9   5   2
    //
    // Except those chunks will be 16-bits each instead of 1 decimal digit.
    //
    // Sadly HLSL does not expose D3D's umul uint32 x uint32 -> uint64,
    // making the code more complicated than it should be.
    //
    // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#22.12.13%20umul
    // https://github.com/microsoft/DirectXShaderCompiler/issues/2821

    // Read the 16-bit units. In some cases, they will kept as 32-bits
    // for efficiency when overflow has no consequence on the result.
    const uint32_t a0 = a.low & 0xFFFF;
    const uint32_t a1 = a.low >> 16;
//  const uint32_t a2 = a.high & 0xFFFF;
//  const uint32_t a3 = a.high >> 16;
    const uint32_t a23 = a.high;

    const uint32_t b0 = b.low & 0xFFFF;
    const uint32_t b1 = b.low >> 16;
    const uint32_t b01 = b.low;
//  const uint32_t b2 = b.high & 0xFFFF;
    const uint32_t b12 = (b.high << 16) | b1;
//  const uint32_t b3 = b.high >> 16;
    const uint32_t b23 = b.high;
        
    // start multiplying
    // Don't bother multiplying combinations that would
    // overflow anyway (c13, c22, c23, c31, c32, c33).
    //
    //                <---64 bits--->
    //                <-4 x 16 bits->
    //                B3  B2  B1  B0 (multiplicand)
    //              x A3  A2  A1  A0 (multiplier)
    //              ----------------
    //              |C03 C02 C01 C00
    //           C13|C12 C11 C10   0
    //       C23 C22|C21 C20   0   0
    // + C33 C32 C31|C30   0   0   0
    // -----------------------------
    //    A6  A5  A4| A3  A2  A1  A0
    //   (overflow) |<---64 bits--->
    //
    uint32_t c00       = a0  * b0;
    uint32_t c01       = a0  * b1;
    uint32_t c10       = a1  * b0;
    uint32_t c02_03    = a0  * b23; // Can safely compute both {a0*b3, a0*b2} in a single 32-bit multiply.
    uint32_t c11_12    = a1  * b12; // Can safely compute both {a1*b2, a1*b1} in a single 32-bit multiply.
    uint32_t c20_21_30 = a23 * b01; // Can safely compute {a2*b0, a2*b1, a3*b0} in a single 32-bit multiply. a3*b1 is irrevelant.

    // final addition with carry
    // Add columns, carrying from right to left. Note each product is 32-bits,
    // meaning the 16-bit columns actually overlap. So the upper half of C00
    // in column 0 becomes the lowest 16 bits carried into the next higher
    // column 1, and the same for column 1 into column 2.
    //
    //                       <-32b->
    //              |        |   C00
    //              |    <---C10   0
    //              |    |   C01   0
    //              |<---C02   0   0   * for efficiency, some products are actually combined,
    //              |    C11   0   0     like C02 and C03 into C02_03.
    //              |    C20   0   0
    //              |C03   0   0   0
    //           C13|C12   0   0   0
    //       C23 C22|C21   0   0   0
    // + C33 C32 C31|C30   0   0   0
    // -----------------------------
    //    A6  A5  A4| A3  A2  A1  A0
    //   (overflow) |<---64 bits--->

    uint32_t answer0, answer1, answer23;
    answer0   = c00; // Last column. Upper 16 bits will be masked later.
    answer1   = answer0 >> 16; // Carry into next column.
    answer1  += (c01 & 0xFFFF) + (c10 & 0xFFFF); // Add the lower halves of column 1.
    answer23  = answer1 >> 16; // Carry into next column.
    answer23 += (c01 >> 16) + (c10 >> 16); // Add the upper halves of column 1.
    answer23 += c02_03 + c11_12 + c20_21_30; // Tally column 2. Any carry past bit 31 doesn't matter now.
    // There is no answer3 column because the upper 16 bits of answer2 already satisfy that fully.

    uint64_emulated answer;
    answer.low  = (answer0 & 0xFFFF) | (answer1 << 16);
    answer.high = answer23;

    return answer;
}

inline int64_emulated Multiply(int64_emulated a, int64_emulated b)
{
    int64_emulated answer = CastToInt64(Multiply(CastToUint64(Abs(a)), CastToUint64(Abs(b))));
    [flatten]
    if ((a.high ^ b.high) < 0)
    {
        answer = Negate(answer);
    }

    return answer;
}

// Return 0 if a == 0 (not 0xFFFFFFFF).
inline uint32_t GetLowestSetBitIndex(uint64_emulated a)
{
    // HLSL firstbitlow returns 0xFFFFFFFF if the input is 0. So explicitly test for zero to avoid a huge index that wraps around.
    return (a.low  != 0) ? firstbitlow(a.low)
         : (a.high != 0) ? firstbitlow(a.high) + 32
         : 0;
}

// Return 0 if a == 0 (not 0xFFFFFFFF).
inline uint32_t GetHighestSetBitIndex(uint64_emulated a)
{
    // HLSL firstbithigh returns 0xFFFFFFFF if the input is 0. So explicitly test for zero to avoid a huge index that wraps around.
    return (a.high != 0) ? firstbithigh(a.high) + 32
         : (a.low  != 0) ? firstbithigh(a.low)
         : 0;
}

inline int64_emulated Square(int64_emulated a)
{
    return Multiply(a, a);
}

inline uint64_emulated Square(uint64_emulated a)
{
    return Multiply(a, a);
}

inline uint64_emulated Identity(uint64_emulated a)
{
    return a;
}

inline uint64_emulated BitShiftLeft(uint64_emulated a, uint32_t shift)
{
    uint64_emulated result;
    uint32_t inverseShift = 32 - shift;
    result.low   = (shift        < 32) ? (a.low  << shift) : 0;
    result.high  = (shift        < 32) ? (a.high << shift) : 0;
    result.high |= (inverseShift < 32) ? (a.low  >> inverseShift) : 0;
    result.high |= (shift        > 32) ? (a.low  << (shift - 32)) : 0;
    return result;
}

inline uint64_emulated BitShiftLeft(uint64_emulated a, uint64_emulated shift)
{
    return BitShiftLeft(a, shift.low);
}

inline int64_emulated BitShiftLeft(int64_emulated a, uint32_t shift)
{
    return CastToInt64(BitShiftLeft(CastToUint64(a), shift));
}

// Optimized version for single bit shifts.
inline uint64_emulated BitShiftLeftByOne(uint64_emulated a)
{
    uint64_emulated result;
    result.low  = a.low << 1;
    result.high = (a.high << 1) | (a.low >> 31);
    return result;
}

inline uint64_emulated BitShiftRight(uint64_emulated a, uint32_t shift)
{
    uint64_emulated result;
    uint32_t inverseShift = 32 - shift;
    result.high = (shift        < 32) ? (a.high  >> shift) : 0;
    result.low  = (shift        < 32) ? (a.low   >> shift) : 0;
    result.low |= (inverseShift < 32) ? (a.high  << inverseShift) : 0;
    result.low |= (shift        > 32) ? (a.high  >> (shift - 32)) : 0;
    return result;
}

inline uint64_emulated BitShiftRight(uint64_emulated a, uint64_emulated shift)
{
    return BitShiftRight(a, shift.low);
}

// Optimized version for single bit shifts.
inline uint64_emulated BitShiftRightByOne(uint64_emulated a)
{
    uint64_emulated result;
    result.low  = (a.low >> 1) | (a.high << 31);
    result.high = a.high >> 1;
    return result;
}

inline void ComputeQuotientRemainder(
    uint64_emulated dividend,
    uint64_emulated divisor,
    out uint64_emulated quotientAnswer,
    out uint64_emulated remainderAnswer
    )
{
    // Sadly HLSL does not expose D3D's udiv uint64 / uint32 -> uint32 instruction
    // making the code slower than it should be.
    //
    // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#22.12.9%20udiv
    // https://github.com/microsoft/DirectXShaderCompiler/issues/2821

    uint64_emulated quotient = {0,0};

    // Get the number of numerically leading zeros to align them. e.g.:
    //
    // original dividend: 00011001 (3 leading zeros)
    // original divisor:  00000010 (6 leading zeros)
    // aligned divisor:   00010000
    //
    // Note HLSL's firstbithigh() actually returns the index of the highest bit, not the number of leading zeros,
    // but it's functionally equivalent since we're only interested in the delta between them.
    //
    const uint32_t dividendHighestBitIndex = GetHighestSetBitIndex(dividend);
    const uint32_t divisorHighestBitIndex = GetHighestSetBitIndex(divisor);
    const uint32_t divisorAdjustment = dividendHighestBitIndex - min(divisorHighestBitIndex, dividendHighestBitIndex);

    // Align divisor to dividend, clamping to 64 to enable a loop unrolling maximum bound.
    // In practice, the number of iterations should be much less.
    divisor = BitShiftLeft(divisor, divisorAdjustment);
    int32_t loopCount = min(divisorAdjustment, 64);

    do
    {
        quotient = BitShiftLeftByOne(quotient);
        if (LogicalGreaterThanOrEqual(dividend, divisor))
        {
            dividend = Subtract(dividend, divisor);
            quotient.low |= 1;
        }

        divisor = BitShiftRightByOne(divisor);
    }
    while (loopCount-- > 0);

    quotientAnswer = quotient;
    remainderAnswer = dividend;
}

inline uint64_emulated Divide(uint64_emulated dividend, uint64_emulated divisor)
{
    uint64_emulated quotient, remainder;
    ComputeQuotientRemainder(dividend, divisor, /*out*/ quotient, /*out*/ remainder);
    return quotient;
}

inline int64_emulated Divide(int64_emulated dividend, int64_emulated divisor)
{
    uint64_emulated unsignedDividend = CastToUint64(Abs(dividend));
    uint64_emulated unsignedDivisor = CastToUint64(Abs(divisor));
    uint64_emulated unsignedQuotient, unsignedRemainder;
    ComputeQuotientRemainder(unsignedDividend, unsignedDivisor, /*out*/ unsignedQuotient, /*out*/ unsignedRemainder);
    int64_emulated quotient = CastToInt64(unsignedQuotient);

    // Restore sign to quotient.
    [flatten]
    if ((dividend.high ^ divisor.high) < 0)
    {
        quotient = Negate(quotient);
    }
    return quotient;
}

inline uint64_emulated BitAnd(uint64_emulated x, uint64_emulated y)
{
    uint64_emulated result = {x.low & y.low, x.high & y.high};
    return result;
}

inline uint64_emulated BitOr(uint64_emulated x, uint64_emulated y)
{
    uint64_emulated result = {x.low | y.low, x.high | y.high};
    return result;
}

inline uint64_emulated BitXor(uint64_emulated x, uint64_emulated y)
{
    uint64_emulated result = {x.low ^ y.low, x.high ^ y.high};
    return result;
}

inline uint64_emulated BitNot(uint64_emulated x)
{
    uint64_emulated result = {~x.low, ~x.high};
    return result;
}

inline uint BitCount(uint64_emulated x) { return countbits(x.low) + countbits(x.high); }

inline uint32_t4 Min(uint32_t4 a, uint32_t4 b)
{
    return min(a, b);
}

inline uint32_t8 Min(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = min(a[0], b[0]);
    output[1] = min(a[1], b[1]);
    return output;
}

inline uint32_t4 Max(uint32_t4 a, uint32_t4 s)
{
    return max(a, s);
}

inline uint32_t8 Max(uint32_t8 a, uint32_t8 s)
{
    uint32_t8 output;
    output[0] = max(a[0], s[0]);
    output[1] = max(a[1], s[1]);
    return output;
}

inline uint32_t4 Clamp(uint32_t4 a, uint32_t4 min, uint32_t4 max)
{
    return clamp(a, min, max);
}

inline uint32_t8 Clamp(uint32_t8 a, uint32_t8 min, uint32_t8 max)
{
    uint32_t8 output;
    output[0] = clamp(a[0], min[0], max[0]);
    output[1] = clamp(a[1], min[1], max[1]);
    return output;
}

inline uint32_t4 CastUnsigned(int32_t4 a)
{
    return uint32_t4(a);
}

inline uint32_t8 CastUnsigned(int32_t8 a)
{
    uint32_t8 output;
    output[0] = uint32_t4(a[0]);
    output[1] = uint32_t4(a[1]);
    return output;
}

inline int32_t4 CastSigned(uint32_t4 a)
{
    return int32_t4(a);
}

inline int32_t8 CastSigned(uint32_t8 a)
{
    int32_t8 output;
    output[0] = int32_t4(a[0]);
    output[1] = int32_t4(a[1]);
    return output;
}

inline int64_emulated CastToInt64(int32_t a)
{
    return int64_emulated::Construct(a, a >> 31u); // Sign extended.
}

inline int64_emulated CastToInt64(uint32_t a)
{
    return int64_emulated::Construct(a, 0); // Zero extended.
}

inline float Erf(float x)
{
    // Constants for polynomial approximation.
    float a1 = 0.254829592;
    float a2 = -0.284496736;
    float a3 = 1.421413741;
    float a4 = -1.453152027;
    float a5 = 1.061405429;
    float p  = 0.3275911;

    // Save the sign of x.
    float signValue = sign(x);
    x = abs(x);

    // Approximate the formula: 2/sqrt(pi) * integrate(i = 0 to x, e ^ -(i^2))
    float t = 1.0 / (1.0 + p * x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x * x);
    return signValue * y;
}

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline int64_emulated CastToInt64(int16_t a)
{
    return int64_emulated::Construct(a, int32_t(a) >> 31u); // Sign extended.
}
#endif

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline int64_emulated CastToInt64(uint16_t a)
{
    return int64_emulated::Construct(a, 0); // Zero extended.
}
#endif

inline int64_emulated CastToInt64(float32_t a)
{
    const uint32_t mantissaMask = 0x007FFFFF;
    const uint32_t mantissaHiddenOneBit = 0x00800000;
    const uint32_t mantissaShift = 0;
    const uint32_t mantissaBitSize = 23; // Actual bits, excluding hidden leading one.
    const uint32_t signMask = 0x80000000;
    const uint32_t signShift = 31;
    const uint32_t exponentMask = 0x7F800000;
    const uint32_t exponentShift = 23;
    const uint32_t exponentBase = 127; // Identity exponent.
    const uint32_t exponentInfinity = 255;
    const uint32_t integerBitSize = 64;

    uint32_t rawFloat = asuint(a);

    // Extract exponent -127 to 128, where 127 -> 0 identity.
    const uint32_t exponent = ((rawFloat & exponentMask) >> exponentShift);
    const int32_t shift = exponent - (exponentBase + mantissaBitSize);

    // Restore implicit leading one.
    const uint32_t rawMantissa = rawFloat & mantissaMask;
    const uint32_t fullMantissa = rawMantissa | (exponent ? mantissaHiddenOneBit : 0);

    const bool isNegative = (int32_t)rawFloat < 0;

    int64_emulated result;

    [flatten]
    if (exponent >= exponentBase + integerBitSize - 1) // -1 for sign bit.
    {
        // Bigger than an integer can handle.
        result.low = 0xFFFFFFFF;
        result.high = 0x7FFFFFFF;
    }
    else if (exponent < exponentBase - 1)
    {
        // Smaller than 0.5 (and so cannot be an integer).
        result.low = 0;
        result.high = 0;
    }
    else if (shift > 0) // exponent >= exponentBase + mantissaBitSize
    {
        // Result is a very large integer. So just shift left.
        result.low = fullMantissa;
        result.high = 0;
        result = BitShiftLeft(result, shift);
    }
    else
    {
        // Otherwise small integer or fractional component.
        result.low = fullMantissa >> -shift;
        result.high = 0;

        // This function defaults to truncation, like C++ static_cast.
        // To round to nearest integer or halves to nearest evens,
        // you could enable the code below, but to mimic static_cast,
        //
        // uint32_t alignedMantissa = fullMantissa << (32 + shift);
        // result.low += ((alignedMantissa > 0x80000000UL)
        //             | ((alignedMantissa == 0x80000000UL) & (result.low & 1)));
    }

    [flatten]
    if (isNegative)
    {
        result = Negate(result);
    }

    return result;
}

inline int64_emulated CastToInt64(float64_t a)
{
    const uint32_t mantissaMaskHigh = 0x000FFFFF;
    const uint32_t mantissaHiddenOneBitHigh = 0x00100000;
    const uint32_t mantissaShift = 0;
    const uint32_t mantissaBitSize = 52; // Actual bits, excluding hidden leading one.
    const uint32_t signMask = 0x80000000;
    const uint32_t signShift = 31;
    const uint32_t exponentMaskHigh = 0x7FF00000;
    const uint32_t exponentShift = 20;
    const uint32_t exponentBase = 1023; // Identity exponent.
    const uint32_t exponentInfinity = 2047;
    const uint32_t integerBitSize = 64;

    uint64_emulated rawFloat;
    asuint(a, /*out*/ rawFloat.low, /*out*/ rawFloat.high);

    // Extract exponent -127 to 128, where 127 -> 0 identity.
    const uint32_t exponent = ((rawFloat.high & exponentMaskHigh) >> exponentShift);
    const int32_t shift = exponent - (exponentBase + mantissaBitSize);

    // Restore implicit leading one.
    uint64_emulated fullMantissa = rawFloat;
    fullMantissa.high &= mantissaMaskHigh;
    fullMantissa.high |= (exponent ? mantissaHiddenOneBitHigh : 0);

    const bool isNegative = (int32_t)rawFloat.high < 0;

    int64_emulated result;

    [flatten]
    if (exponent >= exponentBase + integerBitSize - 1) // -1 for sign bit.
    {
        // Bigger than an integer can handle.
        result.low = 0xFFFFFFFF;
        result.high = 0x7FFFFFFF;
    }
    else if (exponent < exponentBase - 1)
    {
        // Smaller than 0.5 (and so cannot be an integer).
        result.low = 0;
        result.high = 0;
    }
    else if (shift > 0) // exponent >= exponentBase + mantissaBitSize
    {
        // Result is fully an integer. So just shift left.
        result = BitShiftLeft(CastToInt64(fullMantissa), shift);
    }
    else
    {
        // Otherwise small integer or fractional component.
        result = CastToInt64(BitShiftRight(CastToUint64(fullMantissa), -shift));

        // To round to nearest integer or halves to nearest evens,
        // you would enable the code below, but to mimic static_cast,
        // just truncate instead.
        //
        // uint32_t alignedMantissa = fullMantissa << (32 + shift);
        // result.low += ((alignedMantissa > 0x80000000UL)
        //             | ((alignedMantissa == 0x80000000UL) & (result.low & 1)));
    }

    [flatten]
    if (isNegative)
    {
        result = Negate(result);
    }

    return result;
}

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline int64_emulated CastToInt64(float16_t a)
{
    return CastToInt64(float32_t(a));
}
#endif

inline uint64_emulated CastToUint64(int32_t a)
{
    return uint64_emulated::Construct(a, a >> 31u); // Sign extended.
}

inline uint64_emulated CastToUint64(uint32_t a)
{
    return uint64_emulated::Construct(a, 0u); // Zero extended.
}

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline uint64_emulated CastToUint64(int16_t a)
{
    return uint64_emulated::Construct(a, int32_t(a) >> 31u); // Sign extended.
}
#endif

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline uint64_emulated CastToUint64(uint16_t a)
{
    return uint64_emulated::Construct(a, 0); // Zero extended.
}
#endif

#if SHADER_TARGET_SUPPORTS_NATIVE_16BIT
inline uint64_emulated CastToUint64(float16_t a)
{
    return CastToUint64(CastToInt64(float32_t(a)));
}
#endif

inline uint64_emulated CastToUint64(float32_t a)
{
    return CastToUint64(CastToInt64(a));
}

inline uint64_emulated CastToUint64(float64_t a)
{
    return CastToUint64(CastToInt64(a));
}

inline uint32_t CastToUint32(int64_emulated a)
{
    return a.low;
}

inline int32_t CastToInt32(int64_emulated a)
{
    return a.low;
}

inline float32_t CastToFloat32(int64_emulated a)
{
    const uint32_t mantissaMask = 0x007FFFFF;
    const uint32_t mantissaHiddenOneBit = 0x00800000;
    const uint32_t mantissaShift = 0;
    const uint32_t mantissaBitSize = 23; // Actual bits, excluding hidden leading one.
    const uint32_t signMask = 0x80000000;
    const uint32_t signShift = 31;
    const uint32_t exponentMask = 0x7F800000;
    const uint32_t exponentShift = 23;
    const uint32_t exponentBase = 127; // Identity exponent.
    const uint32_t exponentInfinity = 255;
    const uint32_t integerBitSize = 64;

    uint32_t rawFloat = 0;
    [flatten]
    if (a.high == 0 && a.low == 0)
    {
        return 0.0;
    }

    const bool isNegative = (int32_t)a.high < 0;
    [flatten]
    if (isNegative)
    {
        // Get absolute value.
        a = Negate(a);
        rawFloat = 0x80000000;
    }

    // Get most significant bit to determine exponent shift later.
    uint32_t index = GetHighestSetBitIndex(CastToUint64(a));

    int32_t shift = index - mantissaBitSize;
    [flatten]
    if (shift > 0)
    {
        // Large values needs shifting down to mantissa location.
        // (note the shift is zero extended, not sign extended)
        a = CastToInt64(BitShiftRight(CastToUint64(a), shift));
    }
    else
    {
        // Small integer needs shifting up to mantissa location.
        a.low <<= -shift;
    }

    int32_t exponent = shift + mantissaBitSize + exponentBase;

    rawFloat |= exponent << exponentShift;
    rawFloat |= a.low & mantissaMask;

    return asfloat(rawFloat);
}

inline float64_t CastToFloat64(int64_emulated a)
{
    const uint32_t mantissaMaskHigh = 0x000FFFFF;
    const uint32_t mantissaHiddenOneBitHigh = 0x00100000;
    const uint32_t mantissaShift = 0;
    const uint32_t mantissaBitSize = 52; // Actual bits, excluding hidden leading one.
    const uint32_t signMask = 0x80000000;
    const uint32_t signShift = 31;
    const uint32_t exponentMaskHigh = 0x7FF00000;
    const uint32_t exponentShift = 20;
    const uint32_t exponentBase = 1023; // Identity exponent.
    const uint32_t exponentInfinity = 2047;
    const uint32_t integerBitSize = 64;

    int64_emulated rawFloat = { 0, 0 };
    [flatten]
    if (a.high == 0 && a.low == 0)
    {
        return float64_t(0.0);
    }

    const bool isNegative = (int32_t)a.high < 0;
    [flatten]
    if (isNegative)
    {
        // Get absolute value.
        a = Negate(a);
        rawFloat.high = 0x80000000;
    }

    uint32_t index = GetHighestSetBitIndex(CastToUint64(a));

    int32_t shift = index - mantissaBitSize;
    [flatten]
    if (shift > 0)
    {
        // Large values needs shifting down to mantissa location.
        // (note it's a zero extended, not sign extended shift)
        a = CastToInt64(BitShiftRight(CastToUint64(a), shift));
    }
    else
    {
        // Small integer needs shifting up to mantissa location.
        a = CastToInt64(BitShiftLeft(CastToUint64(a), -shift));
    }

    int32_t exponent = shift + mantissaBitSize + exponentBase;

    rawFloat.high |= exponent << exponentShift;
    rawFloat.high |= a.high & mantissaMaskHigh;
    rawFloat.low |= a.low;

    return asdouble((uint32_t)rawFloat.low, (uint32_t)rawFloat.high);
}

inline float64_t CastToFloat64(uint64_emulated a)
{
    return CastToFloat64(CastToInt64(a));
}

inline uint32_t CastToUint32(uint64_emulated a)
{
    return a.low;
}

inline int32_t CastToInt32(uint64_emulated a)
{
    return a.low;
}

inline float32_t CastToFloat32(uint64_emulated a)
{
    return CastToFloat32(CastToInt64(a));
}

inline int32_t4 Mod(int32_t4 a, int32_t4 b)
{
    return a % b;
}

inline int32_t8 Mod(int32_t8 a, int32_t8 b)
{
    int32_t8 output;
    output[0] = a[0] % b[0];
    output[1] = a[1] % b[1];
    return output;
}

inline uint32_t4 Mod(uint32_t4 a, uint32_t4 b)
{
    return a % b;
}

inline uint32_t8 Mod(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] % b[0];
    output[1] = a[1] % b[1];
    return output;
}

inline int32_t4 Div(int32_t4 a, int32_t4 b)
{
    return a / b;
}

inline int32_t8 Div(int32_t8 a, int32_t8 b)
{
    int32_t8 output;
    output[0] = a[0] / b[0];
    output[1] = a[1] / b[1];
    return output;
}

inline uint32_t4 Div(uint32_t4 a, uint32_t4 b)
{
    return a / b;
}

inline uint32_t8 Div(uint32_t8 a, uint32_t8 b)
{
    uint32_t8 output;
    output[0] = a[0] / b[0];
    output[1] = a[1] / b[1];
    return output;
}

inline int32_t4 Abs(int32_t4 a)
{
    return abs(a); 
}

inline int32_t8 Abs(int32_t8 a)
{
    int32_t8 output;
    output[0] = abs(a[0]);
    output[1] = abs(a[1]);
    return output;
}

inline uint64_emulated Min(uint64_emulated a, uint64_emulated b)
{
    // "Less(a, b) ? a : b" should work, but HLSL errantly returns: error X3020: type mismatch between conditional values
    [flatten] if (LogicalLessThan(a, b)) return a; else return b;
}

inline int64_emulated Min(int64_emulated a, int64_emulated b)
{
    [flatten] if (LogicalLessThan(a, b)) return a; else return b;
}

inline int32_t4 Max(int32_t4 a, int32_t s)
{
    return max(a, s);
}

inline int32_t8 Max(int32_t8 a, int32_t s)
{
    int32_t8 output;
    output[0] = max(a[0], s);
    output[1] = max(a[1], s);
    return output;
}

inline uint64_emulated Max(uint64_emulated a, uint64_emulated b)
{
    // "Greater(a, b) ? a : b" should work, but HLSL errantly returns: error X3020: type mismatch between conditional values
    [flatten] if (LogicalGreaterThan(a, b)) return a; else return b;
}

inline int64_emulated Max(int64_emulated a, int64_emulated b)
{
    [flatten] if (LogicalGreaterThan(a, b)) return a; else return b;
}

#include "Math5D.hlsl"

// From FLT_MAX in CRT
#define min_float32_t (-3.402823466e+38F)
#define min_float min_float32_t
#define min_or_inf_float32_t -1.#INF
#define min_or_inf_float min_or_inf_float32_t

#define min_uint16_t (0)
#define min_int16_t (-32768)
#define min_uint32_t (0)
#define min_int32_t (-2147483648)
#define min_uint min_uint32_t
#define min_int min_int32_t
#define min_or_inf_uint16_t min_uint16_t
#define min_or_inf_int16_t min_int16_t
#define min_or_inf_uint32_t min_uint32_t
#define min_or_inf_int32_t min_int32_t
#define min_or_inf_uint min_uint
#define min_or_inf_int min_int
#define min_int64_emulated {0, 0x80000000}
#define min_uint64_emulated {0, 0}

#define max_float32_t (3.402823466e+38F)
#define max_float max_float32_t
#define max_or_inf_float32_t 1.#INF
#define max_or_inf_float max_or_inf_float32_t

#define max_uint16_t (65535)
#define max_int16_t (32767)
#define max_uint32_t (4294967295U)
#define max_int32_t (2147483647)
#define max_uint max_uint32_t
#define max_int max_int32_t
#define max_or_inf_uint16_t max_uint16_t
#define max_or_inf_int16_t max_int16_t
#define max_or_inf_uint32_t max_uint32_t
#define max_or_inf_int32_t max_int32_t
#define max_or_inf_uint max_uint
#define max_or_inf_int max_int
#define max_int64_emulated {0xFFFFFFFF, 0x7FFFFFFF}
#define max_uint64_emulated {0xFFFFFFFF, 0xFFFFFFFF}

#define min_min16float (-65504.0f)
#define min_float16_t  (-65504.0f)

#define JOIN2(A,B) A ## B
#define JOIN(A,B) JOIN2(A,B)

#define MOD(A, B) ((A) - ((B) * ((A) / (B))))
