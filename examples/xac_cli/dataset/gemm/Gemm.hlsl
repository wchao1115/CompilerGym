//------------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//------------------------------------------------------------------------------
// This implements a Gemm Algorithm using Shared memory.
//
                  
#include "DatatypeDefsOld.hlsl"

// Operand Streams        
RWBUFFER(TBUF) matA : register(u0); // Op(matA) is a m x n matrix
RWBUFFER(TBUF) matB : register(u1); // Op(matB) is a n x k matrix

#ifdef BIAS
RWBUFFER(TBUF) matC : register(u2); // m x k matrix
RWBUFFER(TBUF) result : register(u3); // m x k matrix
#else
RWBUFFER(TBUF) result : register(u2); // m x k matrix
#endif // BIAS

// Parameters
cbuffer Parameters : register (b0) 
{
    uint2 matAStrides;
    uint2 matBStrides;
    uint2 matCStrides;
    uint2 resultStrides;
    uint startColIndex;
    uint startRowIndex;
    uint startBatchIndex;
    uint m; // row count of matrix operand 1 & result
    uint n; // col count of matrix operand 1 & row count of matrix operand 2 
    uint k; // col count of matrix operand 2 & result
    uint2 batchShape;
    uint2 matABatchStrides;
    uint2 matBBatchStrides;
    uint2 matCBatchStrides;
    uint2 resultBatchStrides;
    TCONST alpha; // scalar multiplier alpha
    TCONST beta;  // scalar multiplier beta

    // For address clamping, to avoid reading out of range values.  This is necessary
    // due to flattening of 'if' statements, and the fact that MCDM drivers are not
    // required to handle out of bounds reads, even for volatile descriptors.
    uint matAMaxIndex;
    uint matBMaxIndex;
};

#include "ShaderIncludes.hlsl"
#if defined(SMALL_GEMM)
    #define TILESIZE_X 16
    #define TILESIZE_Y 16
    #define TILESIZE_Z 32
    #define NUM_THREADS_X 16
    #define NUM_THREADS_Y 16
#elif defined(LARGE_GEMM)
    #define TILESIZE_X 64
    #define TILESIZE_Y 64
    #define TILESIZE_Z 16
    #define NUM_THREADS_X 16
    #define NUM_THREADS_Y 8
#elif defined(BASIC_GEMM)
    #define TILESIZE_X 32
    #define TILESIZE_Y 32
    #define TILESIZE_Z 16
    #define NUM_THREADS_X 16
    #define NUM_THREADS_Y 8
#endif

#define NUM_SUBTILES_X (TILESIZE_X / NUM_THREADS_X)
#define NUM_SUBTILES_Y (TILESIZE_Y / NUM_THREADS_Y)

groupshared TCONST aTile[TILESIZE_Y][TILESIZE_Z];
groupshared TCONST bTile[TILESIZE_Z][TILESIZE_X];

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS_X,NUM_THREADS_Y,1)]
void CSMain(
    uint3 groupId : SV_GroupID,
    uint3 threadId : SV_GroupThreadId
    )
{
    const uint row = groupId.y * TILESIZE_Y + startRowIndex;
    const uint col = groupId.x * TILESIZE_X + startColIndex;
    
    const uint batchElement = groupId.z + startBatchIndex;

    uint matAOffset   = GetStridedIndex(batchElement, matABatchStrides,   batchShape);
    uint matBOffset   = GetStridedIndex(batchElement, matBBatchStrides,   batchShape);

    TCONST accum[NUM_SUBTILES_Y][NUM_SUBTILES_X];
    for (uint y = 0; y < NUM_SUBTILES_Y; y++)
    {
        [unroll]
        for (uint x = 0; x < NUM_SUBTILES_X; x++)
        {
            accum[y][x] = 0;
        }
    }

    for (uint step = 0; step < n; step += TILESIZE_Z)
    {    
        // These are accumulated here before being used.  The memory reads are serviced 
        // in parallel with subsequent instructions, hiding latency.
        TBUF aValues[NUM_SUBTILES_Y][TILESIZE_Z / NUM_THREADS_X];

        [unroll]
        for (uint z1 = 0; z1 < TILESIZE_Z; z1 += NUM_THREADS_X)
        {
            [unroll]
            for (uint y = 0; y < NUM_SUBTILES_Y; y++)
            {
                const uint aRow = row + y * NUM_THREADS_Y + threadId.y;
                const uint aCol = step + z1 + threadId.x;

                TBUF aValue = 0;
                [flatten]
                if (aRow < m && aCol < n)
                {
                    uint inputAIndex = aRow * matAStrides.y + aCol * matAStrides.x + matAOffset;
                    aValue = BUFFERLOAD(TBUF, matA, min(inputAIndex, matAMaxIndex));
                }
                aValues[y][z1 / NUM_THREADS_X] = aValue;
            }
        }
                                           
        TBUF bValues[TILESIZE_Z / NUM_THREADS_Y][ NUM_SUBTILES_X];

        [unroll]
        for (uint i = 0; i < TILESIZE_Z; i += NUM_THREADS_Y)
        {
            [unroll]
            for (uint x = 0; x < NUM_SUBTILES_X; x++)
            {
                const uint bRow = step + i + threadId.y;
                const uint bCol = col + x * NUM_THREADS_X + threadId.x;

                TBUF bValue = 0;
                [flatten]
                if (bRow < n && bCol < k)
                {
                    uint inputBIndex = bRow * matBStrides.y + bCol * matBStrides.x + matBOffset;
                    bValue = BUFFERLOAD(TBUF, matB, min(inputBIndex, matBMaxIndex));
                }

                bValues[i / NUM_THREADS_Y][x]  = bValue;
            }
        }           
        
        GroupMemoryBarrierWithGroupSync();   

         // Write out all of the accumulated data and filter values into the tiles in shared memory.
        [unroll]
        for (uint zi = 0; zi < TILESIZE_Z / NUM_THREADS_X; ++zi)      
        {  
            [unroll]
            for (uint y = 0; y < NUM_SUBTILES_Y; y++)
            {
                aTile[y * NUM_THREADS_Y + threadId.y][zi*NUM_THREADS_X + threadId.x] = aValues[y][zi];
            }
        }     
              
        [unroll] 
        for (uint zi = 0; zi < TILESIZE_Z / NUM_THREADS_Y; ++zi)      
        {  
            [unroll]
            for (uint x = 0; x < NUM_SUBTILES_X; x++)
            {
               bTile[zi*NUM_THREADS_Y + threadId.y][x * NUM_THREADS_X + threadId.x] = bValues[zi][x];
            }
        }     
        GroupMemoryBarrierWithGroupSync();

        for (uint z2 = 0; z2 < TILESIZE_Z; z2++)
        {
            TCONST aFrag[NUM_SUBTILES_Y];
            [unroll]
            for (uint y = 0; y < NUM_SUBTILES_Y; y++)
            {
                aFrag[y] = aTile[y * NUM_THREADS_Y + threadId.y][z2];
            }
            TCONST bFrag[NUM_SUBTILES_X];
            [unroll]
            for (uint x = 0; x < NUM_SUBTILES_X; x++)
            {
                bFrag[x] = bTile[z2][x * NUM_THREADS_X + threadId.x];
            }

            for (uint x2 = 0; x2 < NUM_SUBTILES_X; x2++)
            {
                [unroll]
                for(uint y2 = 0; y2 < NUM_SUBTILES_Y; y2++)
                {
                    accum[y2][x2] += aFrag[y2] * bFrag[x2];
                }
            }
        }
                                                
    }

    uint matCOffset   = GetStridedIndex(batchElement, matCBatchStrides,   batchShape);
    uint resultOffset = GetStridedIndex(batchElement, resultBatchStrides, batchShape);

    for (uint y = 0; y < NUM_SUBTILES_Y; y++)
    {
        for (uint x = 0; x < NUM_SUBTILES_X; x++)
        {
            const uint yIndex = row + y * NUM_THREADS_Y + threadId.y;
            const uint xIndex = col + x * NUM_THREADS_X + threadId.x;

            if (yIndex < m && xIndex < k)
            {
                const uint opIndex = yIndex * resultStrides.y + xIndex * resultStrides.x;

#ifdef BIAS
                const uint cIndex = yIndex * matCStrides.y + xIndex * matCStrides.x;
                BUFFERSTORE(result, opIndex + resultOffset, alpha * accum[y][x] + beta * BUFFERLOAD(TBUF, matC, cIndex + matCOffset));
#else
                BUFFERSTORE(result, opIndex + resultOffset, alpha * accum[y][x]);
#endif
            }
        }
    }
}