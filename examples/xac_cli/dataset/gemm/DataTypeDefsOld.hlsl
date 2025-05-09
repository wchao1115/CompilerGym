// DataType definitions which expand the Shader.json macros into useful types for HLSL shaders.

#include "DataTypePrimitives.hlsl"

#if USE_DATA_TYPE_NEW_H
    // Catch the possibility of using TensorUavDataType in Shaders.json while including the old header, which leads to confusing build errors.
    #error "Include the new DataTypeDefs.h instead."
#endif

// Summary for shaders with floating-point input tensors:
//
// T_Precision | Compiler  | TBUF       | TCOMPUTE  | TCONST    | Buffer Type          | Execution Flags
// ------------|-----------|------------|-----------|-----------|----------------------|----------------
// 16          | DXC       | float16_t  | float16_t | float     | (RW)StructuredBuffer | ALLOW_HALF_PRECISION_COMPUTATION = true
// 32          | FXC       | float      | float     | float     | (RW)StructuredBuffer | 
// b16c32      | DXC       | float16_t  | float     | float     | (RW)StructuredBuffer |
// typed_c32   | FXC       | float      | float     | float     | (RW)Buffer           | 

// TCOMPUTE/TCONST, TFLOAT_COMPUTE, TUINT_COMPUTE, TINT_COMPUTE
// TCOMPUTE is a precision flag controlled by ALLOW_HALF_PRECISION_COMPUTATION flag. It is only float16_t if native dxil is being 
// used. It also abstracts the type of the shader, as such is only defined in the single type case, TBUF. If the user is defining
// multiple types like TBUF_IN1 and TBUF_OUT then they should be using the explicit macros TFLOAT_COMPUTE, TUINT_COMPUTE, TINT_COMPUTE.
// TCONST is the 32 bit version of TCOMPUTE in case the user wants to make sure full precision is used. 

// TBUF vs TBUF_IN1 and TBUF_OUT
// If the shader defines T = TYPE then both the input and output buffers share the same datatype. 
// if the shader defines different input/output types or precisions then TBUF isn't defined and the 
// Author must use TBUF_IN1 and TBUF_OUT

// Create Buffer types to abstract if the buffer is 16 or 32 bits.
#if T_Precision == 16 || T_Precision_b16c32 == 1
    // 16 bit datatypes are supported so define the buffer types for use
    #define TFLOAT_BUF float16_t
    #define TUINT_BUF  uint16_t
    #define TINT_BUF   int16_t
#else
    // 16 bit types are not supported so define the buffer types to be 32 bit. 
    // Other macros will ensure that we use typed UAV's if buffers aren't 32 bit.
    #define TFLOAT_BUF float
    #define TUINT_BUF  uint
    #define TINT_BUF   int
#endif

// Make a macro to abstract compute type
#if T_Precision == 16
    // 16 bit datatypes are supported. So define the buffer types for use.
    #define TFLOAT_COMPUTE float16_t
    #define TUINT_COMPUTE  uint16_t
    #define TINT_COMPUTE   int16_t
#else
    // 16 bit types are not supported. So define the buffer types to be 32 bit.
    // Other macros will ensure that we typed uav if buffers aren't 32 bit.
    #define TFLOAT_COMPUTE float
    #define TUINT_COMPUTE  uint
    #define TINT_COMPUTE   int
#endif

// Either types are manually defined by T1, T2, T_OUT. Aka Elementwise if, quantize, dequantize, cast. 
// Or they are all defined at once with T_TYPE
#if defined(T1) || defined(T_OUT)
    // Handle Override by creating a macro for both input and output
    #if T_Input_Precision_Override == 32
        #define TFLOAT_INPUT float
        #define TUINT_INPUT uint
        #define TINT_INPUT int
    #else
        #define TFLOAT_INPUT TFLOAT_BUF
        #define TUINT_INPUT TUINT_BUF
        #define TINT_INPUT TINT_BUF
    #endif

    #if T_Output_Precision_Override == 32
        #define TFLOAT_OUTPUT float
        #define TUINT_OUTPUT uint
        #define TINT_OUTPUT int
    #else
        #define TFLOAT_OUTPUT TFLOAT_BUF
        #define TUINT_OUTPUT TUINT_BUF
        #define TINT_OUTPUT TINT_BUF
    #endif

    // TBUF is NOT defined for this case because inputs are manually set.

    // Make a macro defined type for input buffer 1. This abstracts if it is a float, uint, or int.
    #if T1_float == 1
        #define TBUF_IN1 TFLOAT_INPUT
    #elif T1_uint 
        #define TBUF_IN1 TUINT_INPUT
    #elif T1_int
        #define TBUF_IN1 TINT_INPUT
    #endif

    // Make a macro defined type for input buffer 2. This abstracts if it is a float, uint, or int. 
    #if T2_float == 1
        #define TBUF_IN2 TFLOAT_INPUT
    #elif T2_uint 
        #define TBUF_IN2 TUINT_INPUT
    #elif T2_int
        #define TBUF_IN2 TINT_INPUT
    #endif

    // Make a macro defined type for output buffer 1. This abstracts if it is a float, uint, or int. 
    // Multiple output types aren't supported because the only ops which use it manually specify a type.
    #if T_OUT_float == 1
        #define TBUF_OUT TFLOAT_OUTPUT
    #elif T_OUT_uint 
        #define TBUF_OUT TUINT_OUTPUT
    #elif T_OUT_int
        #define TBUF_OUT TINT_OUTPUT
    #endif
#elif defined(T)
    // T_TYPE means 3 things. 1 defines the compute data type. Defines that all input buffers are of TYPE. 
    // Make a macro for Full Precision constant values. Also make an abstraction around Compute for typing.
    #if T_float == 1
        #define TBUF TFLOAT_BUF
        #define TCONST float
        #define TCOMPUTE TFLOAT_COMPUTE
        #define TCOMPUTE_FULL float
    #elif T_uint == 1
        #define TBUF TUINT_BUF
        #define TCONST uint
        #define TCOMPUTE TUINT_COMPUTE
        #define TCOMPUTE_FULL uint
    #elif T_int == 1
        #define TBUF TINT_BUF
        #define TCONST int
        #define TCOMPUTE TINT_COMPUTE
        #define TCOMPUTE_FULL int
    #endif
#endif

// Here are the Buffer macros which wrap buffer type. ByteAddressBuffer not supported.
#ifndef T_Precision_typed_c32 // IF NOT TYPED UAV
    #define RWBUFFER(TYPE) RWStructuredBuffer<TYPE>
    #define BUFFER(TYPE) StructuredBuffer<TYPE>
#elif T_Precision_typed_c32 == 1
    #define T_Uses_Typed_UAVs 1
    #define RWBUFFER(TYPE) RWBuffer<TYPE>
    #define BUFFER(TYPE) Buffer<TYPE>
#endif

// Input and output UAV's must be the same type in DataTypeDefs.h (either both structured or both typed UAV),
// but at least define the macros in case shaders use them.
#define INPUT_RWBUFFER RWBUFFER
#define OUTPUT_RWBUFFER RWBUFFER

#define BUFFERLOAD(TYPE, buf, elementIndex) buf[elementIndex]
#define BUFFERSTORE(buf, elementIndex, value) buf[elementIndex] = value

// Typed UAV's (RWBuffer) clamp output values to numeric limits, whereas structured buffers (RWStructuredBuffer) just
// just truncate to the lowest bits like C++ does. This is problematic for shaders because it causes different
// behavior in the algorithm output depending on what UAV type which should be independent to output correctness. To
// keep consistent rules with C++ and to keep consistent behavior between both UAV types, call this macro to get a TBUF
// representation which does not saturate.
#if T_Uses_Typed_UAVs && defined(T_uint)
    // Avoid RWBuffer saturation limits by masking off the upper bits.
    #define GET_UNSATURATED_TBUF_VALUE(value, bitMask, bitSize) ((TBUF)((value) & (bitMask)))
#elif T_Uses_Typed_UAVs && defined(T_int)
    // Avoid RWBuffer saturation limits by extending the sign.
    #define GET_UNSATURATED_TBUF_VALUE(value, bitMask, bitSize) ((TBUF)((value) << (32 - (bitSize)) >> (32 - (bitSize))))
#else // defined(T_float) or not typed UAV
    // Other UAV types (RWStructuredBuffer) which truncate integer bits go through here.
    // Also, floating point formats (float16/32) need proper conversion and go through here.
    #define GET_UNSATURATED_TBUF_VALUE(value, bitMask, bitSize) ((TBUF)(value))
#endif


// Downcasting from a higher precision data type to a lower precision data type has undefined behavior if the value doesn't fit in the
// smaller destination type (for fp32->fp16 see https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/LangRef.rst#id768). 
// Some GPUs overflow to infinity with floats (desired), and others clamp to numeric limits (not desired). For example, 40000 + 40000 = 65504 with 
// clamping, and it equals +inf when using overflow. The branching here is unfortunate but unavoidable because of IHV differences.
//
// ASSERT_FP32_ARTIHMETIC_FOR_FP16_TYPES exists for cases where shaders ignore T_Precision==16 and unconditionally upcast to FP32 for arithmetic.
// For example, ALLOW_HALF_PRECISION_COMPUTE is 1, but the HLSL is written to always cast operands to float32; the shader generation system can't
// know that the HLSL is doing this and will noop the call to SATURATE_TO_INF without an explicit assertion that the argument will in fact be float32.
#if (defined(T_float) || defined(T1_float) || defined(T2_float)) && (T_Uses_Typed_UAVs || T_Precision_b16c32 == 1 || ASSERT_FP32_ARTIHMETIC_FOR_FP16_TYPES == 1)
    #define SATURATE_TO_INF(value) (((value) > 65504.0) ? 1.#INF : (((value) < -65504.0) ? -1.#INF : (value)))
#else
    #define SATURATE_TO_INF(value) (value)
#endif

// Explicitly Undefine T, T1, T_OUT because no one should use it directly
#undef T
#undef T1
#undef T2
#undef T_OUT
