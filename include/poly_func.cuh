#include <stdio.h>
#include <stdint.h>

__global__ void polyvec_decompress_gpu(int16_t *r, uint8_t *a);
__global__ void poly_frommsg_gpu(int16_t *msgpoly, uint8_t *msg);
__global__ void poly_decompress_gpu(int16_t *r, uint8_t *a);
__global__ void unpack_pk_gpu(int16_t *pk, uint8_t *seed,
                      const uint8_t *packedpk);
__global__  void poly_sub_gpu(int16_t *r, const int16_t *a, const int16_t *b);
__global__ void polyvec_add_gpu3(int16_t *r, const int16_t *a, const int16_t *b);
__global__ void polyvec_add_gpu2(int16_t *r, const int16_t *a, const int16_t *b);
__global__ void polyvec_pointwise_acc_montgomery_gpu(int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b);
__global__ void polyvec_pointwise_acc_montgomery_gpu2
(int16_t *r, int16_t *t, const int16_t *a,  const int16_t *b);
__global__ void poly_invntt_tomont_gpu(int16_t *r) ;
__global__ void polyvec_invntt_tomont_red_gpu(int16_t *r, uint32_t repeat);
__global__ void polyvec_ntt(int16_t *r);
__global__ void polyvec_ntt2(int16_t *r);
__global__ void unpack_pk_gpu(int16_t *pk, uint8_t *seed,
                      const uint8_t *packedpk);
__global__ void cbd_gpu(int16_t *r, const uint8_t *buf);
__global__ void cbd_gpu2(int16_t *r, const uint8_t *buf);
__global__ void rej_uniform_gpu(int16_t *r, unsigned int len, const uint8_t *buf);
__global__ void poly_compress_gpu(uint8_t *r, int16_t *a);
__global__ void polyvec_reduce_gpu(int16_t *r);
__global__ void poly_reduce_g(int16_t *r);
__global__ void polycopy(int16_t *out,  int16_t *in);
__global__ void poly_veccopy(int16_t *out,  int16_t *in);
__global__ void polyvec_compress_gpu(uint8_t *r, int16_t *a);
__global__ void poly_tomsg_gpu(uint8_t *msg, int16_t *a);

