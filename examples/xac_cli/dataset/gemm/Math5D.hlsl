struct uint5
{
    uint x;
    uint y;
    uint z;
    uint w;
    uint v;
};

struct int5
{
    int x;
    int y;
    int z;
    int w;
    int v;
};

inline uint5 Widen(int5 a)
{
    uint5 output = { (uint)a.x, (uint)a.y, (uint)a.z, (uint)a.w, (uint)a.v };
    return output;
}

inline int5 Narrow(uint5 a)
{
    int5 output = { a.x, a.y, a.z, a.w, a.v };
    return output;
}

//
// Converts a 1D offset into 5D indices.
//
inline uint5 GetNCHWFromIndex(uint offset, uint5 sizes)
{
    uint5 output;
    
    output.v = offset % sizes.v;
    output.w = (offset / sizes.v) % sizes.w;
    output.z = (offset / (sizes.v * sizes.w)) % sizes.z;
    output.y = (offset / (sizes.v * sizes.w * sizes.z)) % sizes.y;
    output.x = (offset / (sizes.v * sizes.w * sizes.z * sizes.y)) % sizes.x;

    return output;
}

inline uint5 GetIndicesFromOffsetSingleN(uint offset, uint5 sizes)
{
    uint5 output;
    
    output.v = offset % sizes.v;
    output.w = (offset / sizes.v) % sizes.w;
    output.z = (offset / (sizes.v * sizes.w)) % sizes.z;
    output.y = (offset / (sizes.v * sizes.w * sizes.z));
    output.x = 0;

    return output;
}

//
// Converts 5D indices into a 1D offset.
//
inline uint GetOffsetFromNCHW(uint5 indices, uint5 strides)
{
    return indices.x * strides.x +
           indices.y * strides.y +
           indices.z * strides.z +
           indices.w * strides.w +
           indices.v * strides.v;
}

inline uint GetStridedIndex5D(uint i, uint5 strides, uint5 sizes)
{
    return strides.x * ((i / (sizes.v * sizes.w * sizes.z * sizes.y   ))) +
           strides.y * ((i / (sizes.v * sizes.w * sizes.z   )) % sizes.y) +
           strides.z * ((i / (sizes.v * sizes.w             )) % sizes.z) +
           strides.w * ((i / (sizes.v                       )) % sizes.w) +
           strides.v * ((i                                   ) % sizes.v);
}

uint5 GetPackedStrides(uint5 sizes)
{
    uint5 strides;
    strides.v = 1;
    strides.w = sizes.v; // * strides.v
    strides.z = sizes.w     * strides.w;
    strides.y = sizes.z     * strides.z;
    strides.x = sizes.y     * strides.y;
    return strides;
}

inline uint5 Add(uint5 a, uint5 b)
{
    uint5 output = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w, a.v + b.v };
    return output;
}

inline int5 Add(int5 a, int5 b)
{
    int5 output = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w, a.v + b.v };
    return output;
}

inline int5 Add(int5 a, int b)
{
    int5 output = { a.x + b, a.y + b, a.z + b, a.w + b, a.v + b };
    return output;
}

inline uint5 Sub(uint5 a, uint5 b)
{
    uint5 output = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w, a.v - b.v };
    return output;
}

inline uint5 Sub(uint5 a, uint s)
{
    uint5 output = { a.x - s, a.y - s, a.z - s, a.w - s, a.v - s };
    return output;
}

inline int5 Sub(int5 a, int5 b)
{
    int5 output = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w, a.v - b.v };
    return output;
}

inline int5 Sub(int5 a, uint s)
{
    int5 output = { a.x - s, a.y - s, a.z - s, a.w - s, a.v - s };
    return output;
}

inline uint5 Mul(uint5 a, uint5 b)
{
    uint5 output = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w, a.v * b.v };
    return output;
}

inline uint5 Mul(uint5 a, uint s)
{
    uint5 output = { a.x * s, a.y * s, a.z * s, a.w * s, a.v * s };
    return output;
}

inline int5 Mul(int5 a, int s)
{
    int5 output = { a.x * s, a.y * s, a.z * s, a.w * s, a.v * s };
    return output;
}

inline int5 Div(int5 a, int5 b)
{
    int5 output = { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w, a.v / b.v };
    return output;
}

inline uint5 Madd(uint5 a, uint5 b, uint5 c)
{
    uint5 output = { a.x * b.x + c.x, 
                     a.y * b.y + c.y,
                     a.z * b.z + c.z,
                     a.w * b.w + c.w,
                     a.v * b.v + c.v };

    return output;
}

inline uint5 Less(uint5 a, uint5 b)
{
    uint5 output = { a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w, a.v < b.v };
    return output;
}

inline uint5 GreaterOrEqual(uint5 a, uint5 b)
{
    uint5 output = { a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w, a.v >= b.v };
    return output;
}

inline bool Any(uint5 a)
{
    return (a.x != 0) || (a.y != 0) || (a.z != 0) || (a.w != 0) || (a.v != 0);
}

inline uint5 Clamp(uint5 a, uint5 min, uint5 max)
{
    uint5 output = { clamp(a.x, min.x, max.x), 
                     clamp(a.y, min.y, max.y),
                     clamp(a.z, min.z, max.z),
                     clamp(a.w, min.w, max.w),
                     clamp(a.v, min.v, max.v) };

    return output;
}

inline uint5 Abs(uint5 a)
{
    uint5 output = { abs(a.x), abs(a.y), abs(a.z), abs(a.w), abs(a.v) };
    return output;
}

inline int5 Abs(int5 a)
{
    int5 output = { abs(a.x), abs(a.y), abs(a.z), abs(a.w), abs(a.v) };
    return output;
}

inline int5 Max(int5 a, int s)
{
    int5 output = { max(a.x, s), max(a.y, s), max(a.z, s), max(a.w, s), max(a.v, s) };
    return output;
}

inline int5 Mod(int5 a, int5 b)
{
    int5 output = { a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w, a.v % b.v };
    return output;
}

inline uint Dot(uint5 a, uint5 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + a.v * b.v;
}

//
// Converts 5D indices into a 1D linear index using sizes, ignoring strides.
//
inline uint GetIndexFromIndicesAndSizes(uint5 indices, uint5 sizes)
{
    return Dot(indices, GetPackedStrides(sizes));
}
