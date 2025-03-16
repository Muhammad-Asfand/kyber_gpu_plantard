// Include C++ header files.
// #include <iostream>
#include <stdint.h>
#include <stdio.h>
#include "include/params.h"
#include "include/poly.h"
#include "include/reduce.h"
#include "include/constants.h"
// #include "include/ntt.cuh"
// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#define R 65536
#define R_INV 169  // R_INV = -multinv(R) % Q
#define Q_INV 62209  // 62209 or 3327 // Q*Q_INV % R = -1 ==> Q_INV = -multinv(Q) % R
#define Q 3329

#define R64 4294967296 /// 2^56 //72057594037927936
#define R_INV64 1929 // R_INV = multinv(R) % Q
#define Q_INV64 1806234369 // or 2488732927 Q*Q_INV % R = -1 ==> Q_INV = -multinv(Q) % R
//

int16_t montgomery_reduce64(int64_t a)
{
  int64_t t;
  int16_t u;
  printf("input: %ld\n", a);
  u = (int64_t)a*Q_INV;    
  printf("u: %d\n", u);
  t = (int64_t)u*Q;
  t = a - t;
  printf("t: %ld\n", t);  
  t = t >> 16;  
  return (int16_t)t;
}

int16_t montgomery_reduce32(int32_t a)
{
  int32_t t;
  int16_t u;
  
  // printf("mont32 input: %d\n", a);
  u = a*QINV;  // u = a * Q' mod R  
  t = (int32_t)u*KYBER_Q;
  t = a - t;  
  t >>= 16;
  return (int16_t) t;
}

int16_t barrett_reduce2(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q/2)/KYBER_Q;

  t  = (int32_t)v*a >> 26;
  t *= KYBER_Q;
  return a - t;
}

int16_t barrett_reduce2_64(int64_t a) {
  int64_t t;
  const int64_t v = (((int64_t)1U << 52) + KYBER_Q/2)/KYBER_Q;

  t  = (int64_t)v*a >> 52;
  t *= KYBER_Q;
  return a - t;
}
// not working
int16_t montgomery_reduce_seil(int32_t a)
{
  int16_t t;
  int16_t u;
  
  t = a>>13;
  printf("t: %d\n", t);  
  u = a&(8191);  // 2^13 = 8192
  printf("u: %d\n", u);
  u = u - t;
  printf("u: %d\n", u);  
  t = t << 9;
  printf("t: %d\n", t);  
  return (int16_t) (u+t);
}

int16_t lazy_mont(int64_t T) {

  int64_t t;
  int32_t s, m;
  s = (int32_t) T ;
  printf("s %d\n", s);
  m = (int64_t)T * Q_INV64;
  printf("m mod R %d\n", m);
  t = T - (int64_t)m * KYBER_Q;
  printf("t %ld\n", t);
  t = (int64_t)t >> 32;
  printf("t %ld\n", t);
  return (int16_t)t;
}


void poly_reduce(poly64 *r)
{
  unsigned int i;
  for(i=0;i<KYBER_N;i++)
    // r->coeffs[i] = barrett_reduce2_64(r->coeffs[i]);
    r->coeffs[i] = barrett_reduce2(r->coeffs[i]);
}

static int16_t fqmul(int16_t a, int16_t b) {
  int64_t t = (int32_t)a*b;  
  return montgomery_reduce32(t);
  // printf("-------%ld \n", t);
}



void ntt(int64_t r[256]) {
  unsigned int len, start, j, k;
  int16_t zeta;
  int64_t t=0;

  k = 1;
  for(len = 128; len >= 64; len >>= 1) {
    for(start = 0; start < 256; start = j + len) {
      zeta = zetas[k++];
      for(j = start; j < start + len; ++j) {        
         // t = fqmul(zeta, r[j + len]);                
        t = (int64_t)zeta * r[j + len];
        if(len==64) 
        t = montgomery_reduce64(t);
        if(j==0) printf("%ld = %d %ld\n", t, zeta, r[j + len]);                
        // t = fqmul (t ,65536);
        
        r[j + len] = r[j] - t;
        r[j] = r[j] + t;
      }
    }
  }
}

void poly_ntt(poly64 *r)
{
  ntt(r->coeffs);
  poly_reduce(r);
}

int main() 
{   

  // poly64 r;
  // for(i=0; i<KYBER_N; i++) r.coeffs[i] = 3;
	// poly_ntt(&r);
	// for(i=0; i<KYBER_N; i++) r.coeffs[i] = r.coeffs[i] * R_INV % KYBER_Q/395;
	// for(i=0; i<KYBER_N; i++) printf("%ld ", r.coeffs[i]);

  kyber_gpu();
  // opt_cbd();

  return 0;
}
// TV for for(i=0; i<KYBER_N; i++) r.coeffs[i] = 3; 
// 1248 1248 1110 1110 1802 1802 1550 1550 2471 2471 3158 3158 88 88 2118 2118 3154 3154 1256 1256 1540 1540 1922 1922 1853 1853 685 685 1457 1457 2304 2304 1830 1830 85 85 608 608 3299 3299 136 136 779 779 1234 1234 2134 2134 194 194 2234 2234 1242 1242 801 801 539 539 2978 2978 2687 2687 1892 1892 133 133 1770 1770 230 230 399 399 160 160 3014 3014 2089 2089 3024 3024 1183 1183 508 508 42 42 2564 2564 1400 1400 1011 1011 1051 1051 2846 2846 1086 1086 1792 1792 1329 1329 1547 1547 3288 3288 1289 1289 2184 2184 1537 1537 191 191 127 127 875 875 2076 2076 187 187 2735 2735 697 697 3049 3049 286 286 2638 2638 600 600 3148 3148 1259 1259 2460 2460 3208 3208 3144 3144 1798 1798 1151 1151 2046 2046 47 47 1788 1788 2006 2006 1543 1543 2249 2249 489 489 2284 2284 2324 2324 1935 1935 771 771 3293 3293 2827 2827 2152 2152 311 311 1246 1246 321 321 3175 3175 2936 2936 3105 3105 1565 1565 3202 3202 1443 1443 648 648 357 357 2796 2796 2534 2534 2093 2093 1101 1101 3141 3141 1201 1201 2101 2101 2556 2556 3199 3199 36 36 2727 2727 3250 3250 1505 1505 1031 1031 1878 1878 2650 2650 1482 1482 1413 1413 1795 1795 2079 2079 181 181 1217 1217 3247 3247 177 177 864 864 1785 1785 1533 1533 2225 2225 2087 2087
// #define R 100
// #define R_INV 8
// #define Q_INV 47
// #define Q 17
// int16_t montgomery_red(int32_t a)
// {
//   int32_t t, u;
//   int16_t m;
//   printf("input: %d\n", a);
//   u = a*Q_INV;
//   m = u%R;
//   printf("m: %d\n", m);
//   t = (int32_t)m*Q;
//   t = a + t;
//   printf("t: %d\n", t);
//   t = t/R;
//   printf("t/R: %d\n", t);
//   return t;
// }

// int main() 
// {    
//     uint32_t i=0, t=0;
//     int16_t a = 7, b = 15;
//     a = a*R%Q;
//     b = b*R%Q;
//  	t = montgomery_red((int32_t)a*b); 	
//  	printf("Mont form %d\n", t);
//  	t = t*R_INV%Q;
//     printf("Final: %d\n", t);
//     // kyber_gpu();

//     return 0;
// }

// #define R 65536
// #define R_INV 169	// R_INV = -multinv(R) % Q
// #define Q_INV 62209	// 62209 or 3327 // Q*Q_INV % R = -1 ==> Q_INV = -multinv(Q) % R
// #define Q 3329
// int16_t montgomery_red(int32_t a)
// {
//   int32_t t, u;
//   int16_t m;
//   printf("input: %d\n", a);
//   u = a*Q_INV;
//   // m = u%R;
//   m = u;
//   printf("m: %d\n", m);
//   t = (int32_t)m*Q;
//   t = a - t;
//   printf("t: %d\n", t);
//   // t = t/R;
//   t = t >> 16;
//   printf("t/R: %d\n", t);
//   return t;
// }

// int main() 
// {    
//     uint32_t i=0;
//     int16_t a = 3326, b = 3328, t=0;
//     a = a*R%Q;
//     b = b*R%Q;
//     printf("a: %d b: %d\n", a, b);
//  	t = montgomery_red((int32_t)a*b); 	
//  	printf("Mont form %d\n", t);
//  	t = t*R_INV%Q;

//     printf("Final: %d\n", t);
//     // kyber_gpu();

//     return 0;
// }

// #define R 4294967296
// #define R_INV 1929	// R_INV = multinv(R) % Q
// #define Q_INV 1806234369	// or 2488732927 Q*Q_INV % R = -1 ==> Q_INV = -multinv(Q) % R
// //
// #define Q 3329
// int16_t montgomery_red(int64_t a)
// {
//   int64_t t, u;
//   int32_t m;
//   // printf("input: %ld\n", a);
//   u = a*Q_INV;  
//   m = u;
//   // printf("m: %d\n", m);
//   t = (int64_t)m*Q;
//   t = a - t;
//   // printf("t: %ld\n", t);  
//   t = t >> 32;
//   // printf("t/R: %ld\n", t);
//   return t;
// }

// int main() 
// {       
//     int32_t a = 11075580, b = 11075584, t=0;
//     a = a*R%Q;
//     b = b*R%Q;
//     printf("a: %d b: %d\n", a, b);
//  	t = montgomery_red((int64_t)a*b); 	
//  	printf("Mont form %d\n", t);
//  	t = t*R_INV%Q;

//     printf("Final: %d\n", t);
//     // kyber_gpu();

//     return 0;
// }