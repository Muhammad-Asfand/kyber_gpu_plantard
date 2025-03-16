#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/params.h"
#include "../include/reduce.h"
#include <stdio.h>


#define CONST_PLANT 1976  //-2^32 mod q
#define Q 3329
#define Qinv_Plant 1806234369

__global__ void polyvec_decompress_gpu(int16_t *r, uint8_t *a)
{
  uint32_t i,tid = threadIdx.x,k;
  uint32_t bIdx1 = blockIdx.x * KYBER_K * KYBER_N;
  uint32_t bIdx2 = blockIdx.x * KYBER_INDCPA_BYTES;
  uint16_t t[8];

#if (KYBER_POLYVECCOMPRESSEDBYTES == (KYBER_K * 352))
    for(i=0;i<KYBER_K;i++) {
      t[0] = (a[bIdx2+ i*KYBER_N/8*11 + 0 + tid*11] >> 0) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  1 + tid*11] << 8);
      t[1] = (a[bIdx2 + i*KYBER_N/8*11 + 1 + tid*11] >> 3) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  2 + tid*11] << 5);
      t[2] = (a[bIdx2 + i*KYBER_N/8*11 + 2 + tid*11] >> 6) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  3 + tid*11] << 2) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 + 4+ tid*11] << 10);
      t[3] = (a[bIdx2 + i*KYBER_N/8*11 + 4 + tid*11] >> 1) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  5 + tid*11] << 7);
      t[4] = (a[bIdx2 + i*KYBER_N/8*11 + 5 + tid*11] >> 4) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  6 + tid*11] << 4);
      t[5] = (a[bIdx2 + i*KYBER_N/8*11 + 6 + tid*11] >> 7) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  7 + tid*11] << 1) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 + 8 + tid*11] << 9);
      t[6] = (a[bIdx2 + i*KYBER_N/8*11 + 8 + tid*11] >> 2) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 +  9 + tid*11] << 6);
      t[7] = (a[bIdx2 + i*KYBER_N/8*11 + 9 + tid*11] >> 5) | ((uint16_t)a[bIdx2 + i*KYBER_N/8*11 + 10 + tid*11] << 3);      


      // t[0] = (a[0] >> 0) | ((uint16_t)a[ 1] << 8);
      // t[1] = (a[1] >> 3) | ((uint16_t)a[ 2] << 5);
      // t[2] = (a[2] >> 6) | ((uint16_t)a[ 3] << 2) | ((uint16_t)a[4] << 10);
      // t[3] = (a[4] >> 1) | ((uint16_t)a[ 5] << 7);
      // t[4] = (a[5] >> 4) | ((uint16_t)a[ 6] << 4);
      // t[5] = (a[6] >> 7) | ((uint16_t)a[ 7] << 1) | ((uint16_t)a[8] << 9);
      // t[6] = (a[8] >> 2) | ((uint16_t)a[ 9] << 6);
      // t[7] = (a[9] >> 5) | ((uint16_t)a[10] << 3);


      for(k=0;k<8;k++)
        r[bIdx1 + i*KYBER_N + 8*tid+k] = ((uint32_t)(t[k] & 0x7FF)*KYBER_Q + 1024) >> 11;  
  }
#else

  for(i=0;i<KYBER_K;i++) {    
    t[0] = (a[bIdx2 + i*KYBER_N/4*5 +0 + tid*5] >> 0) | ((uint16_t)a[bIdx2 + i*KYBER_N/4*5 +1 + tid*5] << 8);
    t[1] = (a[bIdx2 + i*KYBER_N/4*5 +1 + tid*5] >> 2) | ((uint16_t)a[bIdx2 + i*KYBER_N/4*5 +2 + tid*5] << 6);
    t[2] = (a[bIdx2 + i*KYBER_N/4*5 +2 + tid*5] >> 4) | ((uint16_t)a[bIdx2 + i*KYBER_N/4*5 +3 + tid*5] << 4);
    t[3] = (a[bIdx2 + i*KYBER_N/4*5 +3 + tid*5] >> 6) | ((uint16_t)a[bIdx2 + i*KYBER_N/4*5 +4 + tid*5] << 2);      
    for(k=0;k<4;k++)
      r[bIdx1 + i*KYBER_N + 4*tid+k] = ((uint32_t)(t[k] & 0x3FF)*KYBER_Q + 512) >> 10;    
  }
#endif
}


__global__ void poly_decompress_gpu(int16_t *r, uint8_t *a)
{  
  uint32_t bIdx1 = blockIdx.x * KYBER_N;
  uint32_t bIdx2 = blockIdx.x * KYBER_INDCPA_BYTES;
  uint8_t t[8], j;

#if (KYBER_POLYCOMPRESSEDBYTES == 96)
    uint32_t tid = KYBER_POLYVECCOMPRESSEDBYTES + threadIdx.x*3;
    t[0] = (a[bIdx2 + tid + 0] >> 0);
    t[1] = (a[bIdx2 + tid + 0] >> 3);
    t[2] = (a[bIdx2 + tid + 0] >> 6) | (a[bIdx2 + tid + 1] << 2);
    t[3] = (a[bIdx2 + tid + 1] >> 1);
    t[4] = (a[bIdx2 + tid + 1] >> 4);
    t[5] = (a[bIdx2 + tid + 1] >> 7) | (a[bIdx2 + tid + 2] << 1);
    t[6] = (a[bIdx2 + tid + 2] >> 2);
    t[7] = (a[bIdx2 + tid + 2] >> 5);
    for(j=0;j<8;j++)
      r[bIdx1 + 8*threadIdx.x+j] = ((uint16_t)(t[j] & 7)*KYBER_Q + 4) >> 3;
#elif (KYBER_POLYCOMPRESSEDBYTES == 128)
  // for(i=0;i<KYBER_N/2;i++) {
    uint32_t tid = KYBER_POLYVECCOMPRESSEDBYTES + threadIdx.x;
    r[bIdx1 + 2*threadIdx.x+0] = (((uint16_t)(a[bIdx2 + tid + 0] & 15)*KYBER_Q) + 8) >> 4;
    r[bIdx1 + 2*threadIdx.x+1] = (((uint16_t)(a[bIdx2 + tid + 0] >> 4)*KYBER_Q) + 8) >> 4;    
  // }
#elif (KYBER_POLYCOMPRESSEDBYTES == 160)    
    uint32_t tid = KYBER_POLYVECCOMPRESSEDBYTES + threadIdx.x*5;
    t[0] = (a[bIdx2 + tid + 0] >> 0);
    t[1] = (a[bIdx2 + tid + 0] >> 5) | (a[bIdx2 + tid + 1] << 3);
    t[2] = (a[bIdx2 + tid + 1] >> 2) ;
    t[3] = (a[bIdx2 + tid + 1] >> 7) | (a[bIdx2 + tid + 2] << 1);
    t[4] = (a[bIdx2 + tid + 2] >> 4) | (a[bIdx2 + tid + 3] << 4);
    t[5] = (a[bIdx2 + tid + 3] >> 1);
    t[6] = (a[bIdx2 + tid + 3] >> 6) | (a[bIdx2 + tid + 4] << 2);
    t[7] = (a[bIdx2 + tid + 4] >> 3);

    for(j=0;j<8;j++)
      r[bIdx1 + 8*threadIdx.x+j] = ((uint16_t)(t[j] & 31)*KYBER_Q + 16) >> 5;
#endif
}

__global__ void poly_frommsg_gpu(int16_t *msgpoly, uint8_t *msg)
{
  uint32_t j;
  int16_t mask;
  uint32_t tid = threadIdx.x;
  uint32_t bIdx = blockIdx.x*KYBER_N;
  uint32_t bIdx2 = blockIdx.x*KYBER_INDCPA_MSGBYTES;
  for(j=0;j<8;j++) {
      mask = -(int16_t)((msg[bIdx2 + tid] >> j)&1);
      msgpoly[bIdx + 8*tid+j] = mask & ((KYBER_Q+1)/2);
  }  
}

__constant__ uint32_t zetas_plantard_cpu[128] = {
        1290168, 2230699446, 3328631909, 4243360600, 3408622288, 812805467, 2447447570, 1094061961, 1370157786, 2475831253, 249002310, 
        1028263423, 3594406395, 4205945745, 734105255, 2252632292, 381889553, 3157039644, 1727534158, 1904287092, 3929849920, 72249375, 
        2889974991, 1719793153, 1839778722, 2701610550, 690239563, 3718262466, 3087370604, 3714391964, 2546790461, 1059227441, 372858381, 
        427045412, 4196914574, 2265533966, 1544330386, 2972545705, 2937711185, 2651294021, 838608815, 2550660963, 3242190693, 815385801, 
        3696329620, 42575525, 1703020977, 2470670584, 2991898216, 1851390229, 1041165097, 583155668, 1855260731, 3700200122, 1979116802, 
        3098982111, 3415073125, 3376368103, 1910737929, 836028480, 3191874164, 4012420634, 1583035408, 1174052340, 21932846, 3562152210, 
        752167598, 3417653460, 2112004045, 932791035, 2951903026, 1419184148, 1817845876, 3434425636, 4233039261, 300609006, 975366560, 
        2781600929, 3889854731, 3935010590, 2197155094, 2130066389, 3598276897, 2308109491, 2382939200, 1228239371, 1884934581, 3466679822, 
        1211467195, 2977706375, 3144137970, 3080919767, 945692709, 3015121229, 345764865, 826997308, 2043625172, 2964804700, 2628071007, 
        4154339049, 483812778, 3288636719, 2696449880, 2122325384, 1371447954, 411563403, 3577634219, 976656727, 2708061387, 723783916, 
        3181552825, 3346694253, 3617629408, 1408862808, 519937465, 1323711759, 1474661346, 2773859924, 3580214553, 1143088323, 2221668274, 
        1563682897, 2417773720, 1327582262, 2722253228, 3786641338, 1141798155, 2779020594
};

__constant__ uint32_t zetas_inv_plantard_cpu[128]={
          1515946703, 3153169142, 508325959, 1572714069, 2967385035, 1877193577, 2731284400, 2073299023, 3151878974, 714752744, 1521107373, 
          2820305951, 2971255538, 3775029832, 2886104489, 677337889, 948273044, 1113414472, 3571183381, 1586905910, 3318310570, 717333078, 
          3883403894, 2923519343, 2172641913, 1598517417, 1006330578, 3811154519, 140628248, 1666896290, 1330162597, 2251342125, 3467969989, 
          3949202432, 1279846068, 3349274588, 1214047530, 1150829327, 1317260922, 3083500102, 828287475, 2410032716, 3066727926, 1912028097, 
          1986857806, 696690400, 2164900908, 2097812203, 359956707, 405112566, 1513366368, 3319600737, 3994358291, 61928036, 860541661, 
          2477121421, 2875783149, 1343064271, 3362176262, 2182963252, 877313837, 3542799699, 732815087, 4273034451, 3120914957, 2711931889, 
          282546663, 1103093133, 3458938817, 2384229368, 918599194, 879894172, 1195985186, 2315850495, 594767175, 2439706566, 3711811629, 
          3253802200, 2443577068, 1303069081, 1824296713, 2591946320, 4252391772, 598637677, 3479581496, 1052776604, 1744306334, 3456358482, 
          1643673276, 1357256112, 1322421592, 2750636911, 2029433331, 98052723, 3867921885, 3922108916, 3235739856, 1748176836, 580575333, 
          1207596693, 576704831, 3604727734, 1593356747, 2455188575, 2575174144, 1404992306, 4222717922, 365117377, 2390680205, 2567433139, 
          1137927653, 3913077744, 2042335005, 3560862042, 89021552, 700560902, 3266703874, 4045964987, 1819136044, 2924809511, 3200905336, 
          1847519727, 3482161830, 886345009, 51606697, 966335388, 3317020402, 2435836064
};

__constant__ int16_t zetas_gpu[128] = {
  2285, 2571, 2970, 1812, 1493, 1422, 287, 202, 3158, 622, 1577, 182, 962,
  2127, 1855, 1468, 573, 2004, 264, 383, 2500, 1458, 1727, 3199, 2648, 1017,
  732, 608, 1787, 411, 3124, 1758, 1223, 652, 2777, 1015, 2036, 1491, 3047,
  1785, 516, 3321, 3009, 2663, 1711, 2167, 126, 1469, 2476, 3239, 3058, 830,
  107, 1908, 3082, 2378, 2931, 961, 1821, 2604, 448, 2264, 677, 2054, 2226,
  430, 555, 843, 2078, 871, 1550, 105, 422, 587, 177, 3094, 3038, 2869, 1574,
  1653, 3083, 778, 1159, 3182, 2552, 1483, 2727, 1119, 1739, 644, 2457, 349,
  418, 329, 3173, 3254, 817, 1097, 603, 610, 1322, 2044, 1864, 384, 2114, 3193,
  1218, 1994, 2455, 220, 2142, 1670, 2144, 1799, 2051, 794, 1819, 2475, 2459,
  478, 3221, 3021, 996, 991, 958, 1869, 1522, 1628
};

__constant__ int16_t zetas_inv_gpu[128] = {
  1701, 1807, 1460, 2371, 2338, 2333, 308, 108, 2851, 870, 854, 1510, 2535,
  1278, 1530, 1185, 1659, 1187, 3109, 874, 1335, 2111, 136, 1215, 2945, 1465,
  1285, 2007, 2719, 2726, 2232, 2512, 75, 156, 3000, 2911, 2980, 872, 2685,
  1590, 2210, 602, 1846, 777, 147, 2170, 2551, 246, 1676, 1755, 460, 291, 235,
  3152, 2742, 2907, 3224, 1779, 2458, 1251, 2486, 2774, 2899, 1103, 1275, 2652,
  1065, 2881, 725, 1508, 2368, 398, 951, 247, 1421, 3222, 2499, 271, 90, 853,
  1860, 3203, 1162, 1618, 666, 320, 8, 2813, 1544, 282, 1838, 1293, 2314, 552,
  2677, 2106, 1571, 205, 2918, 1542, 2721, 2597, 2312, 681, 130, 1602, 1871,
  829, 2946, 3065, 1325, 2756, 1861, 1474, 1202, 2367, 3147, 1752, 2707, 171,
  3127, 3042, 1907, 1836, 1517, 359, 758, 1441
};

__device__ int16_t plant_mul(int32_t a, int32_t b) {
  int32_t t;

  t = __mulhi(a, b);         // Perform signed 32-bit multiplication
  t += 8;                    // Adjust for rounding
  t *= Q;                    // Scale with modulus
  t >>= 16;                  // Scale down to fit the range
  
  return (int16_t)t;
}


__device__ int16_t plant_red(int32_t a) {
  int32_t t;

    t = __mulhi(a, Qinv_Plant);  
    t *= Q;                              // Scale back using modulus
    t = a - t;                           // Perform reduction
    t += (t >> 31) & Q;                  // Add Q if t is negative using bitwise operations
    return (int16_t)t;
}

__device__ int16_t montgomery_reduce_gpu(int32_t a)
{
  int32_t t;
  int16_t u;

  u = a*QINV;
  t = (int32_t)u*KYBER_Q;
  t = a - t;
  t >>= 16;
  return t;
}

__device__ int16_t montgomery_reduce_gpu64(int64_t a)
{
  // int64_t t;
  // int16_t u;

  // u = a*QINV;
  // t = (int64_t)u*KYBER_Q;
  // t = a - t;
  // t >>= 16;
  // return t;

  int64_t t, u;
  int32_t m;
  // printf("input: %ld\n", a);
  u = a*Q_INV64;  
  m = u;
  // printf("m: %d\n", m);
  t = (int64_t)m*KYBER_Q;
  t = a - t;
  // printf("t: %ld\n", t);  
  t = t >> 32;
  // printf("t/R: %ld\n", t);
  return t;
}

__device__ static int16_t fqmul_gpu(int16_t a, int16_t b) {
  return montgomery_reduce_gpu((int32_t)a*b);
}

__device__ int16_t barrett_reduce_gpu(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q/2)/KYBER_Q;

  t  = (int32_t)v*a >> (int32_t)26;
  t *= KYBER_Q;
  return a - t;
}

__device__ int16_t csubq_gpu(int16_t a) {
  a -= KYBER_Q;
  a += (a >> 15) & KYBER_Q;
  return a;
}

__device__ void poly_reduce_gpu(int16_t *r)
{
  uint32_t i, tid = threadIdx.x;
  
  for(i=0;i<KYBER_N/blockDim.x;i++)    
    r[i*blockDim.x + tid] = plant_red(r[i*blockDim.x + tid]);  
}
// Use N threads to process
__global__ void polyvec_reduce_gpu(int16_t *r)
{
  uint32_t i, tid = threadIdx.x;
  uint32_t bIdx = blockIdx.x*KYBER_K*KYBER_N;

  for(i=0;i<KYBER_K;i++)  
    r[bIdx + i*KYBER_N + tid] = barrett_reduce_gpu(r[bIdx + i*KYBER_N + tid]);
}
// Use N/2 threads to process
__global__ void poly_reduce_g(int16_t *r)
{
  uint32_t bid = blockIdx.x;
  uint32_t tid = threadIdx.x;
  
  r[bid * KYBER_N + tid] = barrett_reduce_gpu(r[bid * KYBER_N + tid]);
  // r[bid * KYBER_N + tid + blockDim.x] = barrett_reduce_gpu(r[bid * KYBER_N + tid + blockDim.x]);
}

// // Addressing + fqmul in PTX, interleaved
__device__ void ntt_gpu5(int16_t *a){    
    uint32_t tid = threadIdx.x, temp2, idx1, idx2;
    uint32_t len =128, j;
    int16_t t, zeta, tmp3 = KYBER_Q, u;
    int32_t x, t1;
    float f_level, temp1, temp3, two, f_len;
    float f_tid = threadIdx.x;
    uint64_t tmp2;

    f_level = 1.0, two = 2.0;        
    temp1 = (tid/len);
    j = temp1 * len + tid;    
    idx1 = f_level + temp1;     
    
    for(len = 128; len >= 2; len >>= 1) {           
      f_len = len/2;
      zeta = zetas_gpu[idx1];                
            // t = fqmul_gpu(zeta, a[j + len]);        
      asm volatile ("{\n\t"        
        "mul.wide.s16 %0, %10, %11;\n\t"  //x = (int32_t)zeta * a[j + len]; 
        "div.approx.f32 %5, %16, %14;\n\t"// (f_tid/(f_len);       
        "cvt.rzi.u32.f32 %6, %5;\n\t"     // convert (f_tid/f_len) to integer
        "mul.wide.s32 %1, %0, %12;\n\t"         
        "mul.f32 %7, %7, %15;\n\t"        // f_level = f_level * two;        
        "cvt.s16.s32 %2, %1;\n\t"         // u = x*QINV;    
        "mul.wide.s16 %3, %2, %13;\n\t"   // t1 = (int32_t)u*KYBER_Q;
        "add.f32 %5, %7, %5;\n\t"         // idx1 = f_level + temp1; 
        "cvt.rzi.u32.f32 %8, %5;\n\t"     // convert to integer idx
        "cvt.rz.f32.u32 %5, %6;\n\t"      // convert (f_tid/f_len) to float
        "sub.s32 %3, %0, %3;\n\t"         // t1 = x - t1;
        "mul.f32 %5, %5, %14;\n\t"        // (f_tid/f_len) * f_len;   
        "shr.b32 %3, %3, 16;\n\t"         // t = t1 >> 16;  
        "add.f32 %5, %5, %16;\n\t"        // (f_tid/f_len) * f_len + f_tid;
        "cvt.s16.s32 %4, %3;\n\t"         
        "}"
      : "+r"(x), "+l"(tmp2), "+h"(u) , "+r"(t1), "+h"(t), "+f"(temp1), "+r"(temp2), "+f"(f_level), "+r"(idx1), "+r"(j): "h"(zeta), "h"(a[j+len]), "r"(QINV), "h"(tmp3), "f"(f_len), "f"(two), "f"(f_tid)) ;   
    
      a[j + len] = a[j] - t;
      a[j] = a[j] + t;
      j = temp1;
      __syncthreads();        
    }
}
// Only addressing in PTX (floating point)
__device__ void ntt_gpu4(int16_t *a){    
    uint32_t tid = threadIdx.x, idx1, temp2;
    uint32_t len, j;
    int16_t t, zeta;
    float f_level, temp1, two, f_len;
    float f_tid = threadIdx.x;
    f_level = 1.0, two = 2.0;

    for(len = 128; len >= 2; len >>= 1) {   
      f_len = len;
      asm volatile ("{\n\t"
       "div.approx.f32 %2, %7, %5;\n\t"  // (f_tid/f_len);      
       "cvt.rzi.u32.f32 %3, %2;\n\t"     // convert (f_tid/f_len) to integer
       "cvt.rzi.u32.f32 %1, %0;\n\t"     // convert f_level to integer      
       "add.u32 %1, %1, %3;\n\t"         // idx1 = idx1 + temp2;
       "mul.f32 %0, %0, %6;\n\t"         // f_level = f_level * two;
       "cvt.rz.f32.u32 %2, %3;\n\t"     // convert (f_tid/f_len) to float
       "mul.f32 %2, %2, %5;\n\t"        // (f_tid/f_len) * f_len;  
       "add.f32 %2, %2, %7;\n\t"        // (f_tid/f_len) * f_len + f_tid;  
       "cvt.rzi.u32.f32 %4, %2;\n\t"    // j = (f_tid/f_len) * f_len + f_tid;  
      "}"
      : "+f"(f_level), "+r"(idx1), "+f"(temp1), "+r"(temp2), "+r"(j) : "f"(f_len), "f"(two), "f"(f_tid)) ;        
      zeta = zetas_gpu[idx1];                
      
      t = fqmul_gpu(zeta, a[j + len]);    
      a[j + len] = a[j] - t;
      a[j] = a[j] + t;
        __syncthreads();          
    }
}
 // Addressing + fqmul in PTX (float and int separate)
__device__ void ntt_gpu3(int16_t *a){    
    uint32_t tid = threadIdx.x, temp2, idx1, idx2;
    uint32_t len =128, j;
    int16_t t, zeta, tmp3 = KYBER_Q, u;
    int32_t x, t1;
    float f_level, temp1, temp3, two, f_len;
    float f_tid = threadIdx.x;
    uint64_t tmp2;

    f_level = 1.0, two = 2.0;        
    temp1 = (tid/len);
    j = temp1 * len + tid;    
    idx1 = f_level + temp1;     
    
    for(len = 128; len >= 2; len >>= 1) {           
      f_len = len/2;
      zeta = zetas_gpu[idx1];                
      // t = fqmul_gpu(zeta, a[j + len]);        
      asm volatile ("{\n\t"        
        "mul.wide.s16 %0, %5, %6;\n\t"  //x = (int32_t)zeta * a[j + len];       
        "mul.wide.s32 %1, %0, %7;\n\t" 
        "cvt.s16.s32 %2, %1;\n\t"       // u = x*QINV;    
        "mul.wide.s16 %3, %2, %8;\n\t"  // t1 = (int32_t)u*KYBER_Q;
        "sub.s32 %3, %0, %3;\n\t"       // t1 = x - t1;
        "shr.b32 %3, %3, 16;\n\t"       //
        "cvt.s16.s32 %4, %3;\n\t"       // t = t1 >> 16;  
        "}"
      : "+r"(x), "+l"(tmp2), "+h"(u) , "+r"(t1), "+h"(t): "h"(zeta), "h"(a[j+len]), "r"(QINV), "h"(tmp3)) ;   

      a[j + len] = a[j] - t;
      a[j] = a[j] + t;
      __syncthreads();        
      asm volatile ("{\n\t"
        // "div.approx.f32 %2, %5, %6;\n\t" //  f_len/2
        "div.approx.f32 %2, %8, %6;\n\t"    // (f_tid/(f_len);      
        "cvt.rzi.u32.f32 %3, %2;\n\t"   // convert (f_tid/f_len) to integer       
        "mul.f32 %0, %0, %7;\n\t"       // f_level = f_level * two;        
        "add.f32 %2, %2, %0;\n\t"       // idx1 = f_level + temp1; 
        "cvt.rzi.u32.f32 %5, %2;\n\t"   // convert to integer idx
        "cvt.rz.f32.u32 %2, %3;\n\t"    // convert (f_tid/f_len) to float
        "mul.f32 %2, %2, %6;\n\t"       // (f_tid/f_len) * f_len;        
        "add.f32 %2, %2, %8;\n\t"       // (f_tid/f_len) * f_len + f_tid;                  
        "cvt.rzi.u32.f32 %4, %2;\n\t"   // j = (f_tid/f_len) * f_len + f_tid;          
        "}"
      : "+f"(f_level), "+r"(idx2), "+f"(temp1), "+r"(temp2), "+r"(j), "+r"(idx1) : "f"(f_len), "f"(two), "f"(f_tid)) ;                               
      // printf("tid: %u len: %u ==> %u %u \n", tid, len, level + temp2, idx);
    }
}

// //precompute the next addresses
// __device__ void ntt_gpu(int16_t *a){    
//     uint32_t tid = threadIdx.x;
//     uint32_t len =128, j, level, s;
//     int16_t t, zeta;
//     level = 1;
    
//     s = (tid/len);
//     j = s * len + tid;          
//     for(len = 128; len >= 2; len >>= 1) {        
//         zeta = zetas_gpu[level + s];                

//         t = fqmul_gpu(zeta, a[j + len]);        
//         a[j + len] = a[j] - t;
//         a[j] = a[j] + t;
//         __syncthreads();        
//         level = level << 1;
//         s = (tid/(len/2));
//         j = s * (len/2) + tid;              
//         // printf("len: %u tid: %u s: %u\n", len, tid, s);
//     }
// }

//precompute the next addresses + PTX fqmul
__device__ void ntt_gpu2(int16_t *a){    
    uint32_t tid = threadIdx.x;
    uint32_t len =128, j, level, s;
    int32_t x, t1;  
    int16_t t, zeta, u, tmp3 = KYBER_Q;
    int64_t tmp2;;
    level = 1;
    
    s = (tid/len);
    j = s * len + tid;          
    for(len = 128; len >= 2; len >>= 1) {        
        zeta = zetas_gpu[level + s];             
        // t = fqmul_gpu(zeta, a[j + len]);     
      asm volatile ("{\n\t"        
        "mul.wide.s16 %0, %5, %6;\n\t"  //x = (int32_t)zeta * a[j + len];       
        "mul.wide.s32 %1, %0, %7;\n\t" 
        "cvt.s16.s32 %2, %1;\n\t"       // u = x*QINV;    
        "mul.wide.s16 %3, %2, %8;\n\t"  // t1 = (int32_t)u*KYBER_Q;
        "sub.s32 %3, %0, %3;\n\t"       // t1 = x - t1;
        "shr.b32 %3, %3, 16;\n\t"       //
        "cvt.s16.s32 %4, %3;\n\t"       // t = t1 >> 16;  
        "}"
      : "+r"(x), "+l"(tmp2), "+h"(u) , "+r"(t1), "+h"(t): "h"(zeta), "h"(a[j+len]), "r"(QINV), "h"(tmp3)) ;      
        a[j + len] = a[j] - t;
        a[j] = a[j] + t;
        __syncthreads();        
        level = level << 1;
        x = len/2;
        s = tid/x;
        j = s * x + tid;              
        // printf("len: %u tid: %u s: %u\n", len, tid, s);
    }
}

 // PTX fqmul
__device__ void ntt_gpu1(int16_t *a){    
    uint32_t tid = threadIdx.x;
    uint32_t len, j, level, s;
    int32_t x, t1;        
    int16_t t, zeta, u, tmp3 = KYBER_Q;
    int64_t tmp2;
    level = 1;
    
    for(len = 128; len >= 2; len >>= 1) {        
        zeta = zetas_gpu[level + (tid/len)];                
        j = (tid/len) * len + tid;          
            
        // t = fqmul_gpu(zeta, a[j + len]);        
        // x = (int32_t)zeta * a[j + len];    
        //    
      asm volatile ("{\n\t"        
        "mul.wide.s16 %0, %5, %6;\n\t"  //x = (int32_t)zeta * a[j + len];       
        "mul.wide.s32 %1, %0, %7;\n\t" 
        "cvt.s16.s32 %2, %1;\n\t"       // u = x*QINV;    
        "mul.wide.s16 %3, %2, %8;\n\t"  // t1 = (int32_t)u*KYBER_Q;
        "sub.s32 %3, %0, %3;\n\t"       // t1 = x - t1;
        "shr.b32 %3, %3, 16;\n\t"       //
        "cvt.s16.s32 %4, %3;\n\t"       // t = t1 >> 16;  
        "}"
      : "+r"(x), "+l"(tmp2), "+h"(u) , "+r"(t1), "+h"(t): "h"(zeta), "h"(a[j+len]), "r"(QINV), "h"(tmp3)) ;        

        a[j + len] = a[j] - t;
        a[j] = a[j] + t;
        __syncthreads();        
        level = level << 1;
    }
}

//  // Original C code, unroll-2, this version is slower than shared memory
// __device__ void ntt_gpu_ori(int16_t *a){    
//     uint32_t tid = threadIdx.x;    
//     uint32_t len, j1, level, s, j2;
//     int16_t t1, t2, zeta1, zeta2, g1, g2, g3, g4;
//     level = 1;
//     __shared__ int16_t s_a[KYBER_N];
//     s_a [tid] = a[tid]; 
//     s_a [tid + 128] = a[tid + 128];    
//     __syncthreads();

//     // Level 7 to 4
//     #pragma unroll
//     for(len = 128; len >= 4; len >>= 2) {        
//         zeta1 = zetas_gpu[level + (tid/len)];                
//         zeta2 = zetas_gpu[(level<<1) + (tid/(len/2))];                
//         j1 = (tid/len) * len + tid;          
//         j2 = (tid/(len/2)) * len/2 + tid;       
//         t1 = fqmul_gpu(zeta1, s_a[j1 + len]);        
//         s_a[j1 + len] = s_a[j1] - t1;
//         s_a[j1] = s_a[j1] + t1;
//         t2 = fqmul_gpu(zeta2, s_a[j2 + (len/2)]);        
//         s_a[j2 + (len/2)] = s_a[j2] - t2;
//         s_a[j2] = s_a[j2] + t2;                  
//         // printf("len: %u tid: %u zeta: %d: %d j: %u %u t: %d %d\n", len, tid, zeta1, zeta2, j1, j2, t1, t2);
//         __syncthreads();        
//         level = level << 2;
//         // if(tid==0) printf("%u %u\n", len, level);
//     }

//     // Last level (level 1)
//     zeta1 = zetas_gpu[level + (tid/len)];                        
//     j1 = (tid/len) * len + tid;                  
//     t1 = fqmul_gpu(zeta1, s_a[j1 + len]);        
//     a[j1 + len] = s_a[j1] - t1;
//     a[j1] = s_a[j1] + t1;
//     // a [tid] = s_a[tid]; 
//     // a [tid + 128] = s_a[tid + 128];
// }

 // Original C code shared memory
__device__ void ntt_gpu_ori_sm(int16_t *a){    
    uint32_t tid = threadIdx.x;
    uint32_t len, j, level, s;
    int16_t t, zeta;
    level = 1;
    __shared__ int16_t s_a[KYBER_N];
    s_a [tid] = a[tid]; 
    s_a [tid + 128] = a[tid + 128];    
    __syncthreads();

    for(len = 128; len >= 2; len >>= 1) {        
        zeta = zetas_gpu[level + (tid/len)];                
        j = (tid/len) * len + tid;                  
        t = fqmul_gpu(zeta, s_a[j + len]);    
        // if(len<64) printf("len %u tid %u zeta %d j %d t %d\n", len, tid, zeta, j, t);    
        s_a[j + len] = s_a[j] - t;
        s_a[j] = s_a[j] + t;
        __syncthreads();        
        level = level << 1;
    }
    a [tid] = s_a[tid]; 
    a [tid + 128] = s_a[tid + 128];
}

 // Original C code
__device__ void ntt_gpu_combine(int16_t *a){    
    uint32_t tid = threadIdx.x;
    uint32_t len, j1, j2, level, s;
    int16_t t, zeta, g1, g2, g3, g4;
    level = 1;    
    __shared__ int16_t s_a[KYBER_N];
    s_a [tid] = a[tid]; 
    s_a [tid + 64] = a[tid + 64];    
    s_a [tid + 128] = a[tid + 128]; 
    s_a [tid + 192] = a[tid + 192];    
    __syncthreads();
    len = 128;
        zeta = zetas_plantard_cpu[level + (tid/len)];                
        j1 = (tid/len) * len + tid;          
        j2 = (tid/len) * len + tid + 64;          
        t = plant_mul(zeta, s_a[j1 + len]);        
        g1 = s_a[j1] - t; //a[j1 + len]
        g2 = s_a[j1] + t; //a[j1] 

        t = plant_mul(zeta, s_a[j2 + len]);        
        g3 = s_a[j2] - t; //a[j2 + len]
        g4 = s_a[j2] + t; //a[j2]
    level = level << 1;
    // __syncthreads();
    len = 64;
        zeta = zetas_plantard_cpu[level + (tid/len)];                
        j1 = (tid/len) * len + tid;          
        j2 = (tid/len) * len + tid + 128;          
        t = plant_mul(zeta, g4);        
        s_a[j1 + len] = g2 - t; //a[j1 + len]
        s_a[j1] = g2 + t; //a[j1] 

        zeta = zetas_plantard_cpu[level + ((tid+64)/len)]; 
        t = plant_mul(zeta, g3);        
        s_a[j2 + len] = g1 - t; //a[j2 + len]
        s_a[j2] = g1 + t; //a[j2]
    level = level << 1;
    __syncthreads();
    for(len = 32; len >= 2; len >>= 1) {        
        zeta = zetas_plantard_cpu[level + (tid/len)];                
        j1 = (tid/len) * len + tid;          
        j2 = (tid/len) * len + tid + 128;           
        t = plant_mul(zeta, s_a[j1 + len]);        
        s_a[j1 + len] = s_a[j1] - t;
        s_a[j1] = s_a[j1] + t;                    

        zeta = zetas_plantard_cpu[level + ((tid+64)/len)];  
        t = plant_mul(zeta, s_a[j2 + len]);        
        s_a[j2 + len] = s_a[j2] - t;
        s_a[j2] = s_a[j2] + t;                  
        level = level << 1;
    }
    __syncthreads();
    a [tid] = s_a[tid]; 
    a [tid + 64] = s_a[tid + 64];
    a [tid + 128] = s_a[tid + 128]; 
    a [tid + 192] = s_a[tid + 192];
}

 // Original C code
__device__ void ntt_gpu_ori(int16_t *a){    
    uint32_t tid = threadIdx.x;
    uint32_t len, j, level, s;
    int16_t t, zeta;
    level = 1;    
    for(len = 128; len >= 2; len >>= 1) {        
        zeta = zetas_gpu[level + (tid/len)];                
        j = (tid/len) * len + tid;          
        t = fqmul_gpu(zeta, a[j + len]);        
        a[j + len] = a[j] - t;
        a[j] = a[j] + t;
        __syncthreads();        
        level = level << 1;
    }
}

__device__ void invntt_gpu(int16_t *r) {
  uint32_t start, len, j, k;
  uint32_t f = 2435836064;
  int16_t t, zeta;
  uint32_t tid = threadIdx.x;
  uint32_t stride, level;
  __shared__ int16_t s_r[KYBER_N*KYBER_K];
  s_r [tid] = r[tid]; 
  s_r [tid + 128] = r[tid + 128];    
  __syncthreads();

  k = 0;  stride = 0;
  level = KYBER_N >> 1;
  for(len = 2; len <= 128; len <<= 1) {    
    k = stride + tid/len;
    level = level >> 1;    
    zeta = zetas_inv_plantard_cpu[k];
    j = (tid/len) * len + tid;
    __syncthreads();  
    t = s_r[j];
    s_r[j] = plant_red((t + s_r[j + len]) * CONST_PLANT);
    // __syncthreads();
    s_r[j + len] = t - s_r[j + len];
    s_r[j + len] = plant_mul(zeta, s_r[j + len]);
    // __syncthreads();
    stride = stride + level;
  }
  // r [tid] = s_r[tid]; 
  // r [tid + 128] = s_r[tid + 128];

  r[tid] = plant_mul(s_r[tid], zetas_inv_plantard_cpu[127]);
  r[tid + blockDim.x] = plant_mul(s_r[tid + blockDim.x], zetas_inv_plantard_cpu[127]);
  // __syncthreads();
}

__global__ void poly_invntt_tomont_gpu(int16_t *r) {
  uint32_t i;
  uint32_t bIdx = blockIdx.x*KYBER_N;
  invntt_gpu(r + bIdx);
}

__global__ void polyvec_invntt_tomont_red_gpu(int16_t *r, uint32_t repeat) {
  uint32_t i;
  uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K;  

  for(i=0;i<repeat;i++)
  {
    invntt_gpu(r + i*KYBER_N + bIdx);
    // __syncthreads();
    poly_reduce_gpu(r + i*KYBER_N + bIdx);
    // __syncthreads();
  }
}

__global__ void polyvec_ntt(int16_t *r)
{
    uint32_t i=0;
    uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K;
    // Process KYBER_K of NTT, each one is KYBER_N long    
    for(i=0;i<KYBER_K;i++)
    {  
      // ntt_gpu_ori(r + i*KYBER_N + bIdx);
      ntt_gpu_ori_sm(r + i*KYBER_N + bIdx);
      // __syncthreads();
      poly_reduce_gpu(r + i*KYBER_N + bIdx);      
    }
}

__global__ void polyvec_ntt2(int16_t *r)
{
    uint32_t i=0;
    uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K;
    // Process KYBER_K of NTT, each one is KYBER_N long    
    for(i=0;i<KYBER_K;i++)
    {  
      ntt_gpu_combine(r + i*KYBER_N + bIdx);      
      __syncthreads();
      poly_reduce_gpu(r + i*KYBER_N + bIdx);      
    }
}

__global__ void unpack_pk_gpu(int16_t *pk, uint8_t *seed,
                      const uint8_t *packedpk)
{
  uint32_t i = 0, j = 0;
  uint32_t tid = threadIdx.x;
  uint32_t bIdx1 = blockIdx.x * KYBER_INDCPA_PUBLICKEYBYTES;
  uint32_t bIdx2 = blockIdx.x * KYBER_N * KYBER_K;
  uint32_t bIdx3 = blockIdx.x * KYBER_SYMBYTES;
  for(i=0;i<KYBER_K;i++)
  { 
      pk[bIdx2 + i * KYBER_N + 2*tid]   = ((packedpk[bIdx1 + i * KYBER_POLYBYTES + 3*tid+0] >> 0) | ((uint16_t)packedpk[bIdx1 + i * KYBER_POLYBYTES + 3*tid+1] << 8)) & 0xFFF;
      pk[bIdx2 + i * KYBER_N + 2*tid+1] = ((packedpk[bIdx1 + i * KYBER_POLYBYTES + 3*tid+1] >> 4) | ((uint16_t)packedpk[bIdx1 + i * KYBER_POLYBYTES + 3*tid+2] << 4)) & 0xFFF;
  }

  if(tid<KYBER_SYMBYTES)
    seed[bIdx3 + tid] = packedpk[bIdx1 + tid+KYBER_POLYVECBYTES];  
}



__device__ void basemul_gpu(int16_t r[2],  const int16_t a[2],  const int16_t b[2], int16_t zeta)
{
  r[0]  = fqmul_gpu(a[1], b[1]);
  r[0]  = fqmul_gpu(r[0], zeta);
  r[0] += fqmul_gpu(a[0], b[0]);

  r[1]  = fqmul_gpu(a[0], b[1]);
  r[1] += fqmul_gpu(a[1], b[0]);
}

__device__ void poly_basemul_montgomery_gpu(int16_t *r, const int16_t *a, const int16_t *b)
{
  uint32_t i, tid = threadIdx.x;
  int16_t zeta = zetas_gpu[64+tid];
  basemul_gpu(&r[4*tid], &a[4*tid], &b[4*tid], zeta);
  // __syncthreads();
  basemul_gpu(&r[4*tid+2], &a[4*tid+2], &b[4*tid+2],-zeta);
  // __syncthreads();
}

__device__  void poly_add_gpu(int16_t *r, const int16_t *a, const int16_t *b)
{
  uint32_t i, tid = threadIdx.x;
  for(i=0;i<KYBER_N/blockDim.x;i++)    
    r[tid + i*blockDim.x] = a[tid + i*blockDim.x] + b[tid + i*blockDim.x];
  __syncthreads();
}
__global__  void poly_sub_gpu(int16_t *r, const int16_t *a, const int16_t *b)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  r[tid + bid*blockDim.x] = a[tid + bid*blockDim.x] - b[tid + bid*blockDim.x];
  __syncthreads();
}

__device__ void poly_csubq_gpu(int16_t *r)
{
  uint32_t i, tid = threadIdx.x;  
  uint32_t bid = blockIdx.x;
  r[bid*blockDim.x + tid] = csubq_gpu(r[bid*blockDim.x + tid]);
}

// __device__ void polyvec_csubq_gpu(int16_t *r)
// {
//   uint32_t i;
//   for(i=0;i<KYBER_K;i++)
//     poly_csubq_gpu(r + i*KYBER_N);
// }

__global__ void polyvec_add_gpu(int16_t *r, const int16_t *a, const int16_t *b, uint32_t repeat)
{
  uint32_t tid = threadIdx.x; 
  uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K;   
  r[tid + bIdx] = a[tid + bIdx] + b[tid + bIdx];    
  r[tid + bIdx + KYBER_N] = a[tid + bIdx + KYBER_N] + b[tid + bIdx + KYBER_N];
}

__global__ void polyvec_add_gpu3(int16_t *r, const int16_t *a, const int16_t *b)
{
  uint32_t tid = threadIdx.x, i; 
  uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K;   

  for(i=0;i<KYBER_K;i++)
  {
    r[tid + i*KYBER_N + bIdx] = a[tid + i*KYBER_N + bIdx] + b[tid + i*KYBER_N + bIdx];     
    // __syncthreads();
  }
}

__global__ void polyvec_add_gpu2(int16_t *r, const int16_t *a, const int16_t *b)
{
  uint32_t i;  
  uint32_t bIdx = blockIdx.x*KYBER_N; 
  poly_add_gpu(r + bIdx , a + bIdx , b + bIdx);
}

// // shared memory, faster
__global__ void polyvec_pointwise_acc_montgomery_gpu(int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b)
{
  uint32_t i, j, tid = threadIdx.x;    
  uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K; 
  uint32_t bIdx2 = blockIdx.x*KYBER_N*KYBER_K*KYBER_K; 
  __shared__ int16_t s_r[KYBER_N*KYBER_K], s_t[KYBER_N];

  for(j=0;j<KYBER_K;j++)
  {
    uint32_t j_idx = bIdx2 + j*KYBER_N*KYBER_K ;
    poly_basemul_montgomery_gpu(s_r + j*KYBER_N, a + j_idx , b + bIdx);
    for(i=1;i<KYBER_K;i++) {
      poly_basemul_montgomery_gpu(s_t, a + i*KYBER_N + j_idx , b + i*KYBER_N + bIdx);
       __syncthreads();
      poly_add_gpu(s_r + j*KYBER_N, s_r + j*KYBER_N, s_t);      
    }    
    poly_reduce_gpu(s_r + j*KYBER_N);
  }

  for(j=0;j<KYBER_K;j++)
  {
    for(i=0;i<KYBER_N;i=i+blockDim.x)
      r[bIdx + tid + j*KYBER_N + i] = s_r[tid + j*KYBER_N + i]; 
  }
}

// naive version
// __global__ void polyvec_pointwise_acc_montgomery_gpu(int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b)
// {
//   uint32_t i, j;  
//   uint32_t bIdx = blockIdx.x*KYBER_N*KYBER_K; 
//   uint32_t bIdx2 = blockIdx.x*KYBER_N*KYBER_K*KYBER_K; 

//   for(j=0;j<KYBER_K;j++)
//   {
//     uint32_t j_idx = bIdx2 + j*KYBER_N*KYBER_K ;
//     poly_basemul_montgomery_gpu(r + j*KYBER_N + bIdx, a + j_idx , b + bIdx);
//     for(i=1;i<KYBER_K;i++) {
//       poly_basemul_montgomery_gpu(t + bIdx, a + i*KYBER_N + j_idx , b + i*KYBER_N + bIdx);
//        __syncthreads();
//       poly_add_gpu(r + j*KYBER_N + bIdx, r + j*KYBER_N + bIdx, t + bIdx);      
//     }    
//     poly_reduce_gpu(r + j*KYBER_N + bIdx);
//   }
// }

// shared memory, faster
__global__ void polyvec_pointwise_acc_montgomery_gpu2
(int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b)
{
  uint32_t i, tid = threadIdx.x;
  uint32_t bIdx = blockIdx.x*KYBER_N; 
  uint32_t bIdx2 = blockIdx.x*KYBER_N*KYBER_K; 
  __shared__ int16_t s_r[KYBER_N], s_t[KYBER_N];

  poly_basemul_montgomery_gpu(s_r, a + bIdx2 , b + bIdx2);  
  for(i=1;i<KYBER_K;i++) {
    poly_basemul_montgomery_gpu(s_t, a + i*KYBER_N + bIdx2 , b + i*KYBER_N+ bIdx2);
           __syncthreads();
    poly_add_gpu(s_r, s_r, s_t);
  }  
  poly_reduce_gpu(s_r);  

  for(i=0;i<KYBER_N;i=i+blockDim.x)
      r[bIdx + tid + i] = s_r[tid + i]; 
}

// naive version
// __global__ void polyvec_pointwise_acc_montgomery_gpu2
// (int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b)
// {
//   uint32_t i;  
//   uint32_t bIdx = blockIdx.x*KYBER_N; 
//   uint32_t bIdx2 = blockIdx.x*KYBER_N*KYBER_K; 
  
//   poly_basemul_montgomery_gpu(r + bIdx, a + bIdx2 , b + bIdx2);
  
//   for(i=1;i<KYBER_K;i++) {
//     poly_basemul_montgomery_gpu(t + bIdx, a + i*KYBER_N + bIdx2 , b + i*KYBER_N+ bIdx2);
//            __syncthreads();
//     poly_add_gpu(r + bIdx, r + bIdx, t + bIdx);
//   }  
//   poly_reduce_gpu(r + bIdx);  
// }

__global__ void polyvec_compress_gpu(uint8_t *r, int16_t *a)
{
  uint32_t i,k;
  uint32_t tid = threadIdx.x;
  uint32_t bIdx = blockIdx.x * KYBER_INDCPA_BYTES;
  uint32_t bIdx2 = blockIdx.x * KYBER_K * KYBER_N;
  // polyvec_csubq_gpu(a);

#if (KYBER_POLYVECCOMPRESSEDBYTES == (KYBER_K * 352))
  uint16_t t[8];
    
  for(i=0;i<KYBER_K;i++) {    
      for(k=0;k<8;k++)
        t[k] = ((((uint32_t)a[bIdx2 + i*KYBER_N + 8*tid+k] << 11) + KYBER_Q/2)
                / KYBER_Q) & 0x7ff;
      r[bIdx + i*352 + tid*11 + 0] = (t[0] >>  0);
      r[bIdx + i*352 + tid*11 +  1] = (t[0] >>  8) | (t[1] << 3);
      r[bIdx + i*352 + tid*11 +  2] = (t[1] >>  5) | (t[2] << 6);
      r[bIdx + i*352 + tid*11 +  3] = (t[2] >>  2);
      r[bIdx + i*352 + tid*11 +  4] = (t[2] >> 10) | (t[3] << 1);
      r[bIdx + i*352 + tid*11 +  5] = (t[3] >>  7) | (t[4] << 4);
      r[bIdx + i*352 + tid*11 +  6] = (t[4] >>  4) | (t[5] << 7);
      r[bIdx + i*352 + tid*11 +  7] = (t[5] >>  1);
      r[bIdx + i*352 + tid*11 +  8] = (t[5] >>  9) | (t[6] << 2);
      r[bIdx + i*352 + tid*11 +  9] = (t[6] >>  6) | (t[7] << 5);
      r[bIdx + i*352 + tid*11 + 10] = (t[7] >>  3);
  }  
#elif (KYBER_POLYVECCOMPRESSEDBYTES == (KYBER_K * 320))
  uint16_t t[4];//This may create race condition
  for(i=0;i<KYBER_K;i++) {    
      for(k=0;k<4;k++)
        t[k] = ((((uint32_t)a[bIdx2 + i*KYBER_N + 4*tid+k] << 10) + KYBER_Q/2)
                / KYBER_Q) & 0x3ff;
      // if (blockIdx.x==2) printf("i: %u tid: %u ==> %d ==> %u %u %u %u\n", i, tid, a[bIdx2 + i*KYBER_N + 4*tid+k], t[0], t[1], t[2], t[3]);
      r[bIdx + i*320 + tid*5 + 0] = (t[0] >> 0);
      r[bIdx + i*320 + tid*5 + 1] = (t[0] >> 8) | (t[1] << 2);
      r[bIdx + i*320 + tid*5 + 2] = (t[1] >> 6) | (t[2] << 4);
      r[bIdx + i*320 + tid*5 + 3] = (t[2] >> 4) | (t[3] << 6);
      r[bIdx + i*320 + tid*5 + 4] = (t[3] >> 2);    
  }
  #endif
}

__global__ void poly_compress_gpu(uint8_t *r, int16_t *a)
{
  uint32_t j;
  uint32_t tid = threadIdx.x;
  uint8_t t[8];
  uint32_t bIdx = blockIdx.x * KYBER_INDCPA_BYTES;
  uint32_t bIdx2 = blockIdx.x * KYBER_N;

#if (KYBER_POLYCOMPRESSEDBYTES == 96)
  for(j=0;j<8;j++)
      t[j] = ((((uint16_t)a[bIdx2 + 8*tid+j] << 3) + KYBER_Q/2)/KYBER_Q) & 7;    // __syncthreads();
  r[bIdx + tid*3 + 0] = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
  r[bIdx + tid*3 + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
  r[bIdx + tid*3 + 2] = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);    
#elif (KYBER_POLYCOMPRESSEDBYTES == 128)
  for(j=0;j<8;j++)
      t[j] = ((((uint16_t)a[bIdx2 + 8*tid+j] << 4) + KYBER_Q/2)/KYBER_Q) & 15;    // __syncthreads();
  r[bIdx + tid*4 + 0] = t[0] | (t[1] << 4);
  r[bIdx + tid*4 + 1] = t[2] | (t[3] << 4);
  r[bIdx + tid*4 + 2] = t[4] | (t[5] << 4);   
  r[bIdx + tid*4 + 3] = t[6] | (t[7] << 4);   
#elif (KYBER_POLYCOMPRESSEDBYTES == 160)
  for(j=0;j<8;j++)
      t[j] = ((((uint16_t)a[bIdx2 + 8*tid+j] << 5) + KYBER_Q/2)/KYBER_Q) & 31;    // __syncthreads();
  r[bIdx + tid*5 + 0] = (t[0] >> 0) | (t[1] << 5);
  r[bIdx + tid*5 + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
  r[bIdx + tid*5 + 2] = (t[3] >> 1) | (t[4] << 4);   
  r[bIdx + tid*5 + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6); 
  r[bIdx + tid*5 + 4] = (t[6] >> 2) | (t[7] << 3);   
#endif
}

// #elif (KYBER_POLYCOMPRESSEDBYTES == 128)
//   for(i=0;i<KYBER_N/8;i++) {
//     for(j=0;j<8;j++)
//       t[j] = ((((uint16_t)a->coeffs[8*i+j] << 4) + KYBER_Q/2)/KYBER_Q) & 15;

//     r[0] = t[0] | (t[1] << 4);
//     r[1] = t[2] | (t[3] << 4);
//     r[2] = t[4] | (t[5] << 4);
//     r[3] = t[6] | (t[7] << 4);
//     r += 4;
//   }
// #elif (KYBER_POLYCOMPRESSEDBYTES == 160)
//   for(i=0;i<KYBER_N/8;i++) {
//     for(j=0;j<8;j++)
//       t[j] = ((((uint32_t)a->coeffs[8*i+j] << 5) + KYBER_Q/2)/KYBER_Q) & 31;

//     r[0] = (t[0] >> 0) | (t[1] << 5);
//     r[1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
//     r[2] = (t[3] >> 1) | (t[4] << 4);
//     r[3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6);
//     r[4] = (t[6] >> 2) | (t[7] << 3);
//     r += 5;
//   }
// }
__global__ void polycopy(int16_t *out,  int16_t *in)
{
  uint32_t k, tid = threadIdx.x;
  uint32_t bid = blockIdx.x;      
  out[bid*KYBER_N + tid] = in[bid*(2*KYBER_K+1)*KYBER_N + k*KYBER_N+tid];  
}

__global__ void poly_veccopy(int16_t *out,  int16_t *in)
{
  uint32_t k, tid = threadIdx.x;
  uint32_t bid = blockIdx.x;      
  for(k=0; k<KYBER_K; k++)  
    out[bid*KYBER_K*KYBER_N + k*KYBER_N + tid] = in[bid*(2*KYBER_K+1)*KYBER_N + k*KYBER_N+tid];  
}

__device__ uint32_t load32_littleendian(const uint8_t x[4])
{
  uint32_t r;
  r  = (uint32_t)x[0];
  r |= (uint32_t)x[1] << 8;
  r |= (uint32_t)x[2] << 16;
  r |= (uint32_t)x[3] << 24;
  return r;
}

// version 2, coalesced memory access
__global__ void cbd_gpu2(int16_t *r, const uint8_t *buf)
{
#if KYBER_ETA != 2
#error "poly_getnoise in poly.c only supports eta=2"
#endif
  uint32_t tid = threadIdx.x;
  uint32_t t, d, bid = blockIdx.x;
  int16_t a,b;
  
  t  = load32_littleendian(buf+bid*(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4 +4*(tid/8));
  d  = t & 0x55555555;
  d += (t>>1) & 0x55555555;
    
  a = (d >> (4*(tid%8)+0)) & 0x3;
  b = (d >> (4*(tid%8)+2)) & 0x3;
  r[bid*(2*KYBER_K+1)*KYBER_N + tid] = a - b;
}

// version 1, suffer from non-coalesced memory access
__global__ void cbd_gpu(int16_t *r, const uint8_t *buf)
{
#if KYBER_ETA != 2
#error "poly_getnoise in poly.c only supports eta=2"
#endif
    uint32_t j, tid = threadIdx.x;
    uint32_t t,d, bid = blockIdx.x;
    int16_t a,b;
  
    t  = load32_littleendian(buf+bid*(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4 + 4*tid);
    d  = t & 0x55555555;
    d += (t>>1) & 0x55555555;

    for(j=0;j<8;j++) {
      a = (d >> (4*j+0)) & 0x3;
      b = (d >> (4*j+2)) & 0x3;
      r[bid*(2*KYBER_K+1) * KYBER_N + 8*tid+j] = a - b;
    }
}
//Generate buf with double size. Place good samples in r at first iteration.
//Then check for pos that is still 0, place good samples in r at second iterat.
__global__ void rej_uniform_gpu(int16_t *r, unsigned int len, const uint8_t *buf)
{  
  uint32_t tid = threadIdx.x;
  uint32_t bIdx1 = 0;  
  uint32_t bIdx2 = blockIdx.x * KYBER_N*KYBER_K*KYBER_K;
  uint16_t val;

  val = buf[bIdx1 + tid*2] | ((uint16_t)buf[bIdx1 + tid*2+1] << 8);
  
  if(val < 19*KYBER_Q) {
      val -= (val >> 12)*KYBER_Q; // Barrett reduction      
      r[bIdx2 + tid] = (int16_t)val;      
  }
  __syncthreads();
  val = buf[bIdx1 + tid*2 + blockDim.x] | ((uint16_t)buf[bIdx1 + tid*2+1 + blockDim.x] << 8);
  if(val < 19*KYBER_Q) {
    val -= (val >> 12)*KYBER_Q; // Barrett reduction
    //Only replace the empty space.
    if(r[bIdx2 + tid] == 0)  r[bIdx2 + tid] = (int16_t)val;      
  }
};

__global__ void poly_tomsg_gpu(uint8_t *msg, int16_t *a)
{
  uint32_t i,j, tid = threadIdx.x;
  uint16_t t;
  uint32_t bid = blockIdx.x;
  poly_csubq_gpu(a);

  if(tid <KYBER_N/8)
  {
    msg[bid*KYBER_INDCPA_MSGBYTES + tid] = 0;
    for(j=0;j<8;j++) {
      t = ((((uint16_t)a[bid*blockDim.x + 8*tid+j] << 1) + KYBER_Q/2)/KYBER_Q) & 1;
      msg[bid*KYBER_INDCPA_MSGBYTES + tid] |= t << j;
    }
  }
}
