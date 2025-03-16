// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/constants.h"
#include "../include/params.h"
#include "../include/reduce.h"
#include "../include/tmp_constants.h"
#include "../include/poly.h"
#include "../include/symmetric.h"
#include "../include/aes_gpu.cuh"
#include "../include/incdpa.cuh"
#include "../include/poly_func.cuh"
#include "../include/sha2.cuh"

#include <stdio.h>
#include <stdint.h>

#define GEN_MATRIX_NBLOCKS ((2*KYBER_N*(1U << 16)/(19*KYBER_Q) \
                             + XOF_BLOCKBYTES)/XOF_BLOCKBYTES)

typedef struct{
  poly vec[KYBER_K];
} polyvec;


void kyber_gpu() {
    uint8_t *c, *pk,*msg, *coins;
    uint32_t i, j, k;

    cudaMallocHost((void**) &c, BATCH*KYBER_INDCPA_BYTES * sizeof(uint8_t)); 
    cudaMallocHost((void**) &pk, BATCH*KYBER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t)); 
    cudaMallocHost((void**) &msg, KYBER_INDCPA_MSGBYTES*BATCH* sizeof(uint8_t));
    cudaMallocHost((void**) &coins, KYBER_SYMBYTES*BATCH* sizeof(uint8_t));

    // Assume these are generated from the CPU
#ifdef AESGPU
    for(i=0; i<BATCH; i++) for(j=0; j<KYBER_INDCPA_PUBLICKEYBYTES; j++) 
        pk[i*KYBER_INDCPA_PUBLICKEYBYTES + j] = tmp_pk_aes[j];
#else
    for(i=0; i<BATCH; i++) for(j=0; j<KYBER_INDCPA_PUBLICKEYBYTES; j++) 
        pk[i*KYBER_INDCPA_PUBLICKEYBYTES + j] = tmp_pk[j];
#endif
#ifdef AESGPU
    for(i=0; i<BATCH; i++) for(j=0; j<KYBER_INDCPA_MSGBYTES; j++) 
        msg[i*KYBER_INDCPA_MSGBYTES + j] = tmp_msg_aes[j];  
#else
    for(i=0; i<BATCH; i++) for(j=0; j<KYBER_INDCPA_MSGBYTES; j++) 
        msg[i*KYBER_INDCPA_MSGBYTES + j] = tmp_msg[j];  
#endif
      
    for(i=0; i<BATCH; i++) for(j=0; j<KYBER_INDCPA_MSGBYTES; j++) 
        coins[i*KYBER_INDCPA_MSGBYTES + j] = tmp_coins[j];  

    indcpa_enc_gpu(c, msg, pk, coins);        

    indcpa_dec_gpu(c);

  // uint32_t hlen = KYBER_PUBLICKEYBYTES;
  // uint8_t *h_input, *d_input;  
  // uint8_t *h_output, *d_output;  
  // cudaMallocHost((void**) &h_input, hlen * sizeof(uint8_t));
  // cudaMalloc((void**) &d_input, hlen * sizeof(uint8_t));
  // cudaMallocHost((void**) &h_output, hlen * sizeof(uint8_t));
  // cudaMalloc((void**) &d_output, hlen * sizeof(uint8_t));

  // cudaMemcpy(d_input, h_input, hlen * sizeof(uint8_t), cudaMemcpyHostToDevice);
  // sha256_gpu<<<1,1>>>(d_input, d_output, hlen);
  // cudaMemcpy(h_output, d_output, 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  // for (i = 0;i < 32;++i) {if(i%4==0) printf("\n");printf("%x ", h_output[i]); }
}

