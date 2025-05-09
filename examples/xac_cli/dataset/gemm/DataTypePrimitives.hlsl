#ifndef __DATA_TYPE_PRIMITIVES_HLSL__
#define __DATA_TYPE_PRIMITIVES_HLSL__

// Define shared primitive data types to share between HLSL and C++.
// DXC already supports standard C++ sized types (e.g. uint32_t), while older FXC does not.
// So define the missing typedef's for FXC.
#if !defined(__hlsl_dx_compiler)
typedef int    int32_t;
typedef uint   uint32_t;
typedef float  float32_t;
typedef double float64_t;
typedef int2   int32_t2;
typedef int4   int32_t4;
typedef uint2  uint32_t2;
typedef uint3  uint32_t3;
typedef uint4  uint32_t4;
typedef float2 float32_t2;
typedef float4 float32_t4;
#endif

// The uint32_t8 is essentially a vector of 8 entries, but HLSL does not actually support
// vector<uint, 8>, which would be nice because intrinsic functions could be applied directly.
// Expressing it as matrix<uint, 2, 4> would enable direct use of intrinsic functions,
// but HLSL reserves 64-bytes instead of 32-bytes (as if it was matrix<uint, 4, 4>), wasting
// half the space of the cbuffer. So the next best thing is two uint4's.
typedef uint32_t4 uint32_t8[2];
typedef int32_t4  int32_t8[2];

// Aliases for shader generation simplicity - they are not distinct types.
typedef float32_t float32_typed;
typedef uint32_t uint32_typed;
typedef int32_t int32_typed;

// Alias to clarify that bool's have different sizes between C++ vs HLSL in the cbuffer.
// HLSL bool is 4 bytes whereas C++ bool is 1 byte.
typedef bool bool32_t;

// Composite struct for hardware that lacks true u/int64_t.
struct uint64_emulated
{
    uint32_t low;
    uint32_t high;

    // HLSL doesn't support constructors for user types,
    // but it supports named static construction methods.
    static uint64_emulated Construct(uint32_t low, uint32_t high)
    {
        uint64_emulated value = { low, high };
        return value;
    }
};

struct int64_emulated
{
    int32_t low;
    int32_t high;

    static int64_emulated Construct(int32_t low, int32_t high)
    {
        int64_emulated value = { low, high };
        return value;
    }
};

#endif // __DATA_TYPE_PRIMITIVES_HLSL__
