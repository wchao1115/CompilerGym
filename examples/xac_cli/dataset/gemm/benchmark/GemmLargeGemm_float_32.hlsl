#define LARGE_GEMM 1
#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=3, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=28, b0)"
#define T float
#define T_Precision 32
#define T_float 1
#include "Gemm.hlsl"
