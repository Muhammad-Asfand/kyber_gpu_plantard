// #include "../include/tmp_constants.h"
#include "../include/aes_gpu.cuh"
#include "../include/params.h"
#include <stdint.h>
#include <stdio.h>


__host__ __device__ uint32_t GETU32(const uint8_t *pt)
{
  uint32_t i = *((uint32_t*)pt);
  return  ((i & 0x000000ffU) << 24) ^
    ((i & 0x0000ff00U) << 8) ^
    ((i & 0x00ff0000U) >> 8) ^
    ((i & 0xff000000U) >> 24);
}

__host__ __device__ void PUTU32(uint8_t *ct, uint32_t st)
{
  *((uint32_t*)ct) =  ((st >> 24) ^
      (((st << 8) >> 24 ) << 8) ^
      (((st << 16) >> 24) << 16) ^
      (st << 24));
}

__device__ void next_rk_256(uint32_t* rk,
      const int round,
      const uint32_t Te0[],
      const uint32_t Te1[],
      const uint32_t Te2[],
      const uint32_t Te3[],
      const uint32_t rcon[])
{
  uint32_t temp;
  temp = rk[7];
  rk[0] = rk[0] ^
    (Te2[(temp >> 16) & 0xff] & 0xff000000) ^
    (Te3[(temp >>  8) & 0xff] & 0x00ff0000) ^
    (Te0[(temp      ) & 0xff] & 0x0000ff00) ^
    (Te1[(temp >> 24)       ] & 0x000000ff) ^
    rcon[round];
  rk[1] = rk[1] ^ rk[0];
  rk[2] = rk[2] ^ rk[1];
  rk[3] = rk[3] ^ rk[2];
  temp = rk[3];
  rk[4] = rk[4] ^
    (Te2[(temp >> 24)       ] & 0xff000000) ^
    (Te3[(temp >> 16) & 0xff] & 0x00ff0000) ^
    (Te0[(temp >>  8) & 0xff] & 0x0000ff00) ^
    (Te1[(temp      ) & 0xff] & 0x000000ff);
  rk[5] = rk[5] ^ rk[4];
  rk[6] = rk[6] ^ rk[5];
  rk[7] = rk[7] ^ rk[6];
}

void AESPrepareKey(uint8_t *enc_key, uint32_t key_bits, uint32_t *dec_key32)
{
    //printf("keybits: %d\n", key_bits);
    unsigned int rk_buf[60];
    unsigned int *rk = rk_buf;
    int i = 0;
    unsigned int temp;

    rk[0] = GETU32(enc_key);
    rk[1] = GETU32(enc_key + 4);
    rk[2] = GETU32(enc_key + 8);
    rk[3] = GETU32(enc_key + 12);
    if (key_bits == 128) {
        for (;;) {
            temp = rk[3];
            rk[4] = rk[0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >> 8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp)& 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)] & 0x000000ff) ^
                rcon_host[i];
            rk[5] = rk[1] ^ rk[4];
            rk[6] = rk[2] ^ rk[5];
            rk[7] = rk[3] ^ rk[6];
            if (++i == 10) {
                //rk += 4;
                rk -= 36;
                goto end;
            }
            rk += 4;
        }
    }
    rk[4] = GETU32(enc_key + 16);
    rk[5] = GETU32(enc_key + 20);
    if (key_bits == 192) {
        for (;;) {
            temp = rk[5];
            rk[6] = rk[0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >> 8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp)& 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)] & 0x000000ff) ^
                rcon_host[i];
            rk[7] = rk[1] ^ rk[6];
            rk[8] = rk[2] ^ rk[7];
            rk[9] = rk[3] ^ rk[8];
            if (++i == 8) {
                rk += 6;
                goto end;

            }
            rk[10] = rk[4] ^ rk[9];
            rk[11] = rk[5] ^ rk[10];
            rk += 6;
        }
    }
    rk[6] = GETU32(enc_key + 24);
    rk[7] = GETU32(enc_key + 28);

    if (key_bits == 256) {
        for (;;) {
            temp = rk[7];
            rk[8] = rk[0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >> 8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp)& 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)] & 0x000000ff) ^
                rcon_host[i];
            rk[9] = rk[1] ^ rk[8];
            rk[10] = rk[2] ^ rk[9];
            rk[11] = rk[3] ^ rk[10];
            if (++i == 7) {
                //rk += 8;
                rk -= 48;
                goto end;
            }
            temp = rk[11];
            rk[12] = rk[4] ^
                (Te4[(temp >> 24)] & 0xff000000) ^
                (Te4[(temp >> 16) & 0xff] & 0x00ff0000) ^
                (Te4[(temp >> 8) & 0xff] & 0x0000ff00) ^
                (Te4[(temp)& 0xff] & 0x000000ff);
            rk[13] = rk[5] ^ rk[12];
            rk[14] = rk[6] ^ rk[13];
            rk[15] = rk[7] ^ rk[14];

            rk += 8;
        }
    }
end:    
    for(int i=0; i<15*4; i++)
    {
        // PUTU32(dec_key + i*4, rk[i]);
        dec_key32[i] = rk[i];
    }    
    // printf("Expanded Keys\n");   
    // for (int i = 0; i < 15*4; i ++)  {
    //     if(i%4==0) printf("\n");
    //     printf("%08x", dec_key32[i]);
    // }
    // printf("\n");
}


__global__ void AES_256_encrypt(uint8_t *out, uint8_t *seed_gpu, uint8_t nonce)
{
  uint32_t s0, s1, s2, s3, t0, t1, t2, t3;

  uint32_t rk[8];  
  uint32_t tid = threadIdx.x;
  uint32_t Idx = blockIdx.x * KYBER_K*KYBER_K*64*16 + tid*16;
  // if(threadIdx.x==0)
  // {
  // printf("block: %u seed0: %u\n", blockIdx.x, seed_gpu[0]);
  // printf("block: %u seed1: %u\n", blockIdx.x, seed_gpu[1]);
  // printf("block: %u seed2: %u\n", blockIdx.x, seed_gpu[2]);
  // printf("block: %u seed3: %u\n", blockIdx.x, seed_gpu[3]);
  // printf("block: %u seed4: %u\n", blockIdx.x, seed_gpu[4]);
  // printf("block: %u seed5: %u\n", blockIdx.x, seed_gpu[5]);
  // printf("block: %u seed6: %u\n", blockIdx.x, seed_gpu[6]);
  // printf("block: %u seed7: %u\n", blockIdx.x, seed_gpu[7]);
  // }
  rk[0] = GETU32(seed_gpu);
  rk[1] = GETU32(seed_gpu + 4);
  rk[2] = GETU32(seed_gpu + 8);
  rk[3] = GETU32(seed_gpu + 12);
  rk[4] = GETU32(seed_gpu + 16);
  rk[5] = GETU32(seed_gpu + 20);
  rk[6] = GETU32(seed_gpu + 24);
  rk[7] = GETU32(seed_gpu + 28);
  // if(threadIdx.x==0)
  // {
  //  printf("\n %u rk\n", blockIdx.x);
  //  for(int i=0; i<8; i++)
  //    printf("%08x ", rk[i]);
  //  // printf("\n Byte rk\n");
  //  // for(int i=0; i<32; i++)
  //  //  printf("%x ", key[i]);
  // }

  s0 = tid ^ rk[0];
  s1 = nonce ^ rk[1];
  s2 = 0 ^ rk[2];
  s3 = 0 ^ rk[3];
  // s0 = GETU32(in     ) ^ rk[0];
  // s1 = GETU32(in +  4) ^ rk[1];
  // s2 = GETU32(in +  8) ^ rk[2];
  // s3 = GETU32(in + 12) ^ rk[3];
  // if(tid==0)  printf("block: %u %08x %08x %08x %08x\n", blockIdx.x, s0, s1, s2, s3);
  // if(threadIdx.x==0)
  // {
  //  printf("\nr0\n");
  //  printf("%08x %08x %08x %08x",  s0, s1, s2, s3);
  //  printf("\n");
  // }
  /* round 1: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
  // printf("\nr1\n");
  // for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
  // printf("\n");
  // if(blockDim.x * blockIdx.x + threadIdx.x==0)
  // {      
  //  printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
  //  printf("\n");
  // }
  next_rk_256(rk, 0, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
  /* round 2: */
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];
// if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr2\n");
//   for(int i=0; i<4; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  s0, s1, s2, s3);
//   printf("\n");
// }
  /* round 3: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
// if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr3\n");
//   for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
//   printf("\n");
// }
  /* round 4: */
 next_rk_256(rk, 1, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];
  /* round 5: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
// if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr5\n");
//   for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
//   printf("\n");
// }
  /* round 6: */
 next_rk_256(rk, 2, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];

  /* round 7: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
// if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr7\n");
//   for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
//   printf("\n");
// }
  /* round 8: */
 next_rk_256(rk, 3, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];
  /* round 9: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
// if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr9\n");
//   for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
//   printf("\n");
// }
 next_rk_256(rk, 4, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
  /* round 10: */
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];
  /* round 11: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];
 next_rk_256(rk, 5, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);
//  if(blockDim.x * blockIdx.x + threadIdx.x==0)
// {
//   printf("\nr11\n");
//   for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
//   printf("\n");
//   printf("%08x %08x %08x %08x",  t0, t1, t2, t3);
//   printf("\n");
// }
  /* round 12: */
  s0 = Te0_ConstMem[t0 >> 24] ^ Te1_ConstMem[(t1 >> 16) & 0xff] ^ Te2_ConstMem[(t2 >>  8) & 0xff] ^ Te3_ConstMem[t3 & 0xff] ^ rk[ 0];
  s1 = Te0_ConstMem[t1 >> 24] ^ Te1_ConstMem[(t2 >> 16) & 0xff] ^ Te2_ConstMem[(t3 >>  8) & 0xff] ^ Te3_ConstMem[t0 & 0xff] ^ rk[ 1];
  s2 = Te0_ConstMem[t2 >> 24] ^ Te1_ConstMem[(t3 >> 16) & 0xff] ^ Te2_ConstMem[(t0 >>  8) & 0xff] ^ Te3_ConstMem[t1 & 0xff] ^ rk[ 2];
  s3 = Te0_ConstMem[t3 >> 24] ^ Te1_ConstMem[(t0 >> 16) & 0xff] ^ Te2_ConstMem[(t1 >>  8) & 0xff] ^ Te3_ConstMem[t2 & 0xff] ^ rk[ 3];

  /* round 13: */
  t0 = Te0_ConstMem[s0 >> 24] ^ Te1_ConstMem[(s1 >> 16) & 0xff] ^ Te2_ConstMem[(s2 >>  8) & 0xff] ^ Te3_ConstMem[s3 & 0xff] ^ rk[ 4];
  t1 = Te0_ConstMem[s1 >> 24] ^ Te1_ConstMem[(s2 >> 16) & 0xff] ^ Te2_ConstMem[(s3 >>  8) & 0xff] ^ Te3_ConstMem[s0 & 0xff] ^ rk[ 5];
  t2 = Te0_ConstMem[s2 >> 24] ^ Te1_ConstMem[(s3 >> 16) & 0xff] ^ Te2_ConstMem[(s0 >>  8) & 0xff] ^ Te3_ConstMem[s1 & 0xff] ^ rk[ 6];
  t3 = Te0_ConstMem[s3 >> 24] ^ Te1_ConstMem[(s0 >> 16) & 0xff] ^ Te2_ConstMem[(s1 >>  8) & 0xff] ^ Te3_ConstMem[s2 & 0xff] ^ rk[ 7];

 next_rk_256(rk, 6, Te0_ConstMem, Te1_ConstMem, Te2_ConstMem, Te3_ConstMem, rcon);

  s0 =
    (Te2_ConstMem[(t0 >> 24)       ] & 0xff000000) ^
    (Te3_ConstMem[(t1 >> 16) & 0xff] & 0x00ff0000) ^
    (Te0_ConstMem[(t2 >>  8) & 0xff] & 0x0000ff00) ^
    (Te1_ConstMem[(t3      ) & 0xff] & 0x000000ff) ^
    rk[0];
  PUTU32(out+Idx     , s0);  // Each thread write 4 bytes

  s1 =
    (Te2_ConstMem[(t1 >> 24)       ] & 0xff000000) ^
    (Te3_ConstMem[(t2 >> 16) & 0xff] & 0x00ff0000) ^
    (Te0_ConstMem[(t3 >>  8) & 0xff] & 0x0000ff00) ^
    (Te1_ConstMem[(t0      ) & 0xff] & 0x000000ff) ^
    rk[1];
  PUTU32(out+Idx + 4, s1);// Another 4 bytes

  s2 =
    (Te2_ConstMem[(t2 >> 24)       ] & 0xff000000) ^
    (Te3_ConstMem[(t3 >> 16) & 0xff] & 0x00ff0000) ^
    (Te0_ConstMem[(t0 >>  8) & 0xff] & 0x0000ff00) ^
    (Te1_ConstMem[(t1      ) & 0xff] & 0x000000ff) ^
    rk[2];
  PUTU32(out+Idx + 8, s2);

  s3 =
    (Te2_ConstMem[(t3 >> 24)       ] & 0xff000000) ^
    (Te3_ConstMem[(t0 >> 16) & 0xff] & 0x00ff0000) ^
    (Te0_ConstMem[(t1 >>  8) & 0xff] & 0x0000ff00) ^
    (Te1_ConstMem[(t2      ) & 0xff] & 0x000000ff) ^
    rk[3];
  // if( threadIdx.x==0)
  //   {
  //     printf("\nr14\n");
  //     // for(int i=4; i<8; i++)   printf("%08x ", rk[i]);
  //     // printf("\n");
  //     printf("%08x %08x %08x %08x",  s0, s1, s2, s3);
  //     printf("\n");
  //   }
  PUTU32(out+Idx +12, s3); // Each thread produced 16 bytes  
  // if(tid==255)  printf("block: %u %08x %08x %08x %08x\n", blockIdx.x, s0, s1, s2, s3);
}


__global__ void encGPUsharedFineGrain(uint8_t *out, const unsigned int *roundkey)
{
    unsigned int tid = threadIdx.x;
    unsigned int counter = tid/4;   // counter within a block
    __shared__ unsigned int s[1024], t[1024];
    __shared__ unsigned int shared_Te0[256];
    __shared__ unsigned int shared_Te1[256];
    __shared__ unsigned int shared_Te2[256];
    __shared__ unsigned int shared_Te3[256];
    __shared__ unsigned int rk[60];
    
    /* initialize T boxes, #threads in block should be larger than 256.
       Thread 0 - 255 cooperate to copy the T-boxes from constant mem to shared mem*/
    if(tid<256)
    {
        shared_Te0[tid] = Te0_ConstMem[tid];
        shared_Te1[tid] = Te1_ConstMem[tid];
        shared_Te2[tid] = Te2_ConstMem[tid];
        shared_Te3[tid] = Te3_ConstMem[tid];        
    }
    if(tid <60)
    {
        rk[tid] = roundkey[tid];
    }
    /* make sure T boxes have been initialized. */
    __syncthreads();    

    if(tid%4 < 3){  // For t0-t2, t4-t6, t8-t10...
        s[tid] = 0 ^ rk[tid%4];
    }
    else{   // For t3, t7, t11, ..., we use tid as a counter value, but maximum only 32-bit
        // Different block needs to generate different counter value
        // s[tid] = (counter + (blockIdx.x * blockDim.x)/4) ^ rk[tid%4];
        s[tid] = counter ^ rk[tid%4];
    }
    __syncthreads();
    // if(tid<8)  printf("round 0 block %u t: %x\n", blockIdx.x, s[tid]);
    ///* round 1: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+4];    
    // if(tid<8)  printf("round 1 block %u t: %x\n", blockIdx.x, t[tid]);
    // if(tid==0)  printf("block %u round 1 t: %x rk: %x\n", blockIdx.x, t[tid], rk[(tid%4)+8]); 
    ///* round 2: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[(tid%4)+8];    
    // if(tid==0)  printf("block %u round 2 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
    /* round 3: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+12];    
        ///* round 4: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+16];
    // if(tid==0)  printf("block %u round 4 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
        ///* round 5: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+20];    
    //if(blockIdx.x == 0 && tid<8)  printf("round 5 t: %x\n", t[tid]);
        ///* round 6: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+24];

    // if(tid==0)  printf("block %u round 6 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
        ///* round 7: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+28];
        ///* round 8: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+32];
        ///* round 9: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+36];
            ///* round 10: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+40];
    // if(tid==0)  printf("block %u round 10 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
        ///* round 11: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+44];
            ///* round 12: */
    s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+48];

        ///* round 13: */
    t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+52];

    s[tid] =(shared_Te2[(t[tid] >> 24)       ] & 0xff000000) ^
            (shared_Te3[(t[(tid+1)%4+counter*4] >> 16) & 0xff] & 0x00ff0000) ^
            (shared_Te0[(t[(tid+2)%4+counter*4] >>  8) & 0xff] & 0x0000ff00) ^
            (shared_Te1[(t[(tid+3)%4+counter*4]      ) & 0xff] & 0x000000ff) ^
            rk[(tid%4)+56];
    // __syncthreads();
    // if(tid==0)  printf("block %u round 14 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
    // out[blockIdx.x * blockDim.x + tid] = s[tid];
    PUTU32(out + blockIdx.x * blockDim.x * 4 + 4*tid, s[tid]);
    // if(tid==0) printf("%08x, %x\n", s[tid], out[blockIdx.x * blockDim.x +4*tid]);
    
}

// poly_gennoise
__global__ void encGPUsharedFineGrain2(uint8_t *out, const unsigned int *roundkey)
{
    unsigned int tid = threadIdx.x;
    unsigned int counter = tid/4;   // counter within a block
    __shared__ unsigned int s[1024], t[1024];
    __shared__ unsigned int shared_Te0[256];
    __shared__ unsigned int shared_Te1[256];
    __shared__ unsigned int shared_Te2[256];
    __shared__ unsigned int shared_Te3[256];
    __shared__ unsigned int rk[60];
    
    /* initialize T boxes, #threads in block should be larger than 256.
       Thread 0 - 255 cooperate to copy the T-boxes from constant mem to shared mem*/
    if(tid<256)
    {
        shared_Te0[tid] = Te0_ConstMem[tid];
        shared_Te1[tid] = Te1_ConstMem[tid];
        shared_Te2[tid] = Te2_ConstMem[tid];
        shared_Te3[tid] = Te3_ConstMem[tid];        
    }
    if(tid <60)
    {
        rk[tid] = roundkey[tid];
    }
    /* make sure T boxes have been initialized. */
    __syncthreads();    

    if(tid%4 < 3){  // For t0-t2, t4-t6, t8-t10...
        s[tid] = 0 ^ rk[tid%4];
    }
    else{   // For t3, t7, t11, ..., we use tid as a counter value, but maximum only 32-bit
        // Different block needs to generate different counter value
        // s[tid] = (counter + (blockIdx.x * blockDim.x)/4) ^ rk[tid%4];
        s[tid] = counter ^ rk[tid%4];
    }
    __syncthreads();
#if KYBER_K == 2    
    if(tid < 160)
    {
#elif KYBER_K == 3
    if(tid < 224)
    {  
#elif KYBER_K == 4
    if(tid < 288)
    {       
#endif      
      // if(tid<8)  printf("round 0 block %u t: %x\n", blockIdx.x, s[tid]);
      ///* round 1: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+4];      
      // if(tid<8)  printf("round 1 block %u t: %x\n", blockIdx.x, t[tid]);
      // if(tid==0)  printf("block %u round 1 t: %x rk: %x\n", blockIdx.x, t[tid], rk[(tid%4)+8]); 
      ///* round 2: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[(tid%4)+8];
      // if(tid==0)  printf("block %u round 2 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
      /* round 3: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+12];
          ///* round 4: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+16];
      // if(tid==0)  printf("block %u round 4 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
          ///* round 5: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+20];
      //if(blockIdx.x == 0 && tid<8)  printf("round 5 t: %x\n", t[tid]);
          ///* round 6: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+24];      
      // if(tid==0)  printf("block %u round 6 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
          ///* round 7: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+28];
          ///* round 8: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+32];
          ///* round 9: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+36];
              ///* round 10: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+40];
      // if(tid==0)  printf("block %u round 10 s: %x rk: %x\n", blockIdx.x, s[tid], rk[(tid%4)+8]);
          ///* round 11: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+44];
              ///* round 12: */
      s[tid] = shared_Te0[t[tid] >> 24] ^ shared_Te1[(t[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(t[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[t[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+48];
  
          ///* round 13: */
      t[tid] = shared_Te0[s[tid] >> 24] ^ shared_Te1[(s[(tid+1)%4+counter*4] >> 16) & 0xff] ^ shared_Te2[(s[(tid+2)%4+counter*4] >>  8) & 0xff] ^ shared_Te3[s[(tid+3)%4+counter*4] & 0xff] ^ rk[ (tid%4)+52];
  
      s[tid] =(shared_Te2[(t[tid] >> 24)       ] & 0xff000000) ^
              (shared_Te3[(t[(tid+1)%4+counter*4] >> 16) & 0xff] & 0x00ff0000) ^
              (shared_Te0[(t[(tid+2)%4+counter*4] >>  8) & 0xff] & 0x0000ff00) ^
              (shared_Te1[(t[(tid+3)%4+counter*4]      ) & 0xff] & 0x000000ff) ^
              rk[(tid%4)+56];
      // out[blockIdx.x * blockDim.x + tid] = s[tid];
#if KYBER_K == 2    
    PUTU32(out + blockIdx.x * 640 + 4*tid, s[tid]);       
#elif KYBER_K == 3
    PUTU32(out + blockIdx.x * 896 + 4*tid, s[tid]);       
#elif KYBER_K == 4
    PUTU32(out + blockIdx.x * 1152 + 4*tid, s[tid]);       
#endif
    // if(tid==0) printf("%08x, %x\n", s[tid], out[blockIdx.x * blockDim.x +4*tid]);
    }
}

__global__ void
encGPUwarpFineGrain(uint8_t *out, const unsigned int *roundkey)
{

    unsigned int tid = threadIdx.x;
    unsigned int counter = tid/4;   // counter within a block
    unsigned int s, t;
    __shared__ unsigned int shared_Te0[256];
    __shared__ unsigned int shared_Te1[256];
    __shared__ unsigned int shared_Te2[256];
    __shared__ unsigned int shared_Te3[256];
    __shared__ unsigned int rk[60];
    
    if(tid<256)
    {
        shared_Te0[tid] = Te0_ConstMem[tid];
        shared_Te1[tid] = Te1_ConstMem[tid];
        shared_Te2[tid] = Te2_ConstMem[tid];
        shared_Te3[tid] = Te3_ConstMem[tid];        
    }
    if(tid <60)
    {
        rk[tid] = roundkey[tid];
    }

    __syncthreads();    

    if(tid%4 < 3){  // For t0-t2, t4-t6, t8-t10...
        s = 0 ^ rk[tid%4];
    }
    else{   // For t3, t7, t11, ..., we use tid as a counter value, but maximum only 32-bit        
        s = (counter + (blockIdx.x * blockDim.x)/4) ^ rk[tid%4];
        //if(tid==3 || tid == 7) printf("counter: %u\n", counter);
    }
    __syncthreads();
    
    //if(blockIdx.x == 0 && tid<8) printf("round 0 s: %x \n", s);
    ///* round 1: */
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+4];
    if(tid<8)printf("round 1 s%u: %x rk: %x\n", tid, t, rk[(tid%4)+4]);
    ///* round 2: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+8];
    // if(tid<8)printf("round 2 s%u: %x rk: %x\n", tid, s, rk[(tid%4)+8]);
    ///* round 3: */
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+12];
    // if(tid<8)printf("round 3 s%u: %x rk: %x\n", tid, t, rk[(tid%4)+12]);
        /////* round 4: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+16];
    // if(tid<8)printf("round 4 s%u: %x rk: %x\n", tid, s, rk[(tid%4)+16]);
        /////* round 5: */
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+20];
    //if(blockIdx.x == 0 && tid<8) printf("round 5 t: %x \n", t);
        /////* round 6: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+24];
    // if(blockIdx.x == 0 && tid<8) printf("round 6 s: %x \n", s);
        /////* round 7: */
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+28];
        /////* round 8: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+32];
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+36];
        /////* round 10: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+40];
    // if(blockIdx.x == 0 && tid<8) printf("round 10 s: %x \n", s);
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+44];
        /////* round 12: */
    s = shared_Te0[t >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+48];
    // if(blockIdx.x == 0 && tid<8) printf("round 12 s: %x %x\n", s, rk[ (tid%4)+48]);
        /////* round 13: */
    t = shared_Te0[s >> 24] ^ shared_Te1[(__shfl_sync(0xffffffff, s, (tid+1)%4+counter*4) >> 16) & 0xff] ^ shared_Te2[(__shfl_sync(0xffffffff, s, (tid+2)%4+counter*4) >>  8) & 0xff] ^ shared_Te3[__shfl_sync(0xffffffff, s, (tid+3)%4+counter*4) & 0xff] ^ rk[ (tid%4)+52];

        s = (shared_Te2[(t >> 24)       ] & 0xff000000) ^
            (shared_Te3[(__shfl_sync(0xffffffff, t, (tid+1)%4+counter*4) >> 16) & 0xff] & 0x00ff0000) ^
            (shared_Te0[(__shfl_sync(0xffffffff, t, (tid+2)%4+counter*4) >>  8) & 0xff] & 0x0000ff00) ^
            (shared_Te1[(__shfl_sync(0xffffffff, t, (tid+3)%4+counter*4) ) & 0xff] & 0x000000ff) ^
            rk[(tid%4)+56];

    // out[blockIdx.x * blockDim.x + tid] = s;  //32-bit implementation
    if(tid==0)
    PUTU32(out + blockIdx.x * blockDim.x * 4 + 4*tid, s);
}

