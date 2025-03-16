#include "../include/incdpa.cuh"
#include "../include/poly_func.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/params.h"
#include "../include/tmp_constants.h"
#include "../include/aes_gpu.cuh"


 void indcpa_enc_gpu(uint8_t * c, uint8_t * msg, uint8_t * pk, uint8_t * coins)
 {
    int16_t *b, *at, *r;
    int16_t *pkpv, *msgpoly, *ep, *epp;
    int16_t *d_sp, *d_pk, *d_msgpoly, *d_r, *d_bp, *d_at, *d_t, *d_v, *d_ep, *d_epp;
    uint8_t *d_seed, *d_packedpk, *d_msg, *d_prf, *d_buf;    
    uint8_t *buf, *d_c, *tmp_prf;
    uint8_t *seed;
    uint32_t i, j, k, hh, count, *exp_seed;
    uint32_t *d_exp_seed;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);   cudaEventCreate(&stop);
    // Allocate host memory    
    cudaMallocHost((void**) &at, KYBER_K*KYBER_K*BATCH*KYBER_N * sizeof(int16_t)); 
    cudaMallocHost((void**) &b, (2*KYBER_K+1)*BATCH*KYBER_N * sizeof(int16_t));   
    cudaMallocHost((void**) &r, (2*KYBER_K+1)*BATCH*KYBER_N * sizeof(int16_t));     
    cudaMallocHost((void**) &pkpv, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMallocHost((void**) &msgpoly, KYBER_N*BATCH*  sizeof(int16_t));
    cudaMallocHost((void**) &buf, BATCH*64*16* KYBER_K*KYBER_K* sizeof(uint8_t));
    cudaMallocHost((void**) &ep, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));     
    cudaMallocHost((void**) &epp, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMallocHost((void**) &seed, BATCH*KYBER_SYMBYTES * sizeof(uint8_t)); 
    // AES-256 expands into 14 * 16 bytes of round keys    
    cudaMallocHost((void**) &exp_seed, BATCH*60 * sizeof(uint32_t)); 
    cudaMallocHost((void**) &tmp_prf, (2*KYBER_K+1)*BATCH*KYBER_ETA*KYBER_N * sizeof(uint8_t));  //KYBER_K=2: 5 KYBER_K=3: 7  KYBER_K=4: 9

    // Allocate device memory.
    cudaMalloc((void**) &d_sp, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));   
    cudaMalloc((void**) &d_t, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));    
    cudaMalloc((void**) &d_v, BATCH*KYBER_N * sizeof(int16_t));    
    cudaMalloc((void**) &d_bp, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));   
    cudaMalloc((void**) &d_at, KYBER_K*KYBER_K*BATCH*KYBER_N * sizeof(int16_t));      
    cudaMalloc((void**) &d_seed, KYBER_SYMBYTES * BATCH * sizeof(uint8_t));
    cudaMalloc((void**) &d_packedpk, BATCH*KYBER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t));
    cudaMalloc((void**) &d_pk, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));   
    cudaMalloc((void**) &d_msgpoly, KYBER_N*BATCH * sizeof(int16_t));
    cudaMalloc((void**) &d_msg, KYBER_INDCPA_MSGBYTES*BATCH* sizeof(uint8_t));
    cudaMalloc((void**) &d_ep, KYBER_K*BATCH*KYBER_N * sizeof(int16_t)); 
    cudaMalloc((void**) &d_epp, BATCH*KYBER_N * sizeof(int16_t)); 
    cudaMalloc((void**) &d_r, (2*KYBER_K+1)*BATCH*KYBER_N * sizeof(int16_t));    
    cudaMalloc((void**) &d_c, BATCH*KYBER_INDCPA_BYTES * sizeof(uint8_t)); 
    //KYBER_K=2: 5 KYBER_K=3: 10  KYBER_K=4: 15
    cudaMalloc((void**) &d_prf,(2*KYBER_K+1)*BATCH*KYBER_ETA*KYBER_N * sizeof(uint8_t));
     // Need larger space to accomodate AES implementation
    cudaMalloc((void**) &d_buf, BATCH*4*KYBER_N* KYBER_K*KYBER_K * sizeof(uint8_t));
    cudaMalloc((void**) &d_exp_seed, BATCH*60 * sizeof(uint32_t)); // AES-256 expands into 14 * 16 bytes of round keys
        

    printf("Encryption\n");    
  for(i=0;i<KYBER_SYMBYTES;i++) seed[i] = pk[i+KYBER_POLYVECBYTES];    
  cudaEventRecord(start);   
  for(count=0; count<REPEAT; count++)     
  { 
    AESPrepareKey(seed, 256, exp_seed);
    cudaMemcpy(d_packedpk, pk, KYBER_INDCPA_PUBLICKEYBYTES*BATCH* sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg, KYBER_INDCPA_MSGBYTES*BATCH* sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exp_seed, exp_seed, 60*BATCH * sizeof(uint32_t),   cudaMemcpyHostToDevice);

    unpack_pk_gpu<<<BATCH, KYBER_N/2>>>(d_pk, d_seed, d_packedpk);
    poly_frommsg_gpu<<<BATCH, KYBER_N/8>>>(d_msgpoly, d_msg);       
    
#ifdef AESGPU
    // Generate 1024 random bytes, for 2 rounds of parallel rejection sampling in GPU. KYBER_K*KYBER_K * 1024 = 4096 B  // Each thread encrypt 4 bytes.
    // TO FIX: Generate more AES for different KYBER_K
#if KYBER_K == 2    
    encGPUsharedFineGrain<<<BATCH, 1024>>> (d_buf, d_exp_seed);
#elif KYBER_K == 3
    encGPUsharedFineGrain<<<BATCH, 1024>>> (d_buf, d_exp_seed);
    // encGPUsharedFineGrain<<<BATCH, 1024>>> (d_buf, d_exp_seed);
#elif KYBER_K == 3
    encGPUsharedFineGrain<<<BATCH, 1024>>> (d_buf, d_exp_seed);
#endif
    // encGPUwarpFineGrain<<<BATCH, 1024>>> (d_buf, d_exp_seed);// errors, need to fix
    // cudaMemcpy(buf, d_buf, BATCH*1024*4* sizeof(uint8_t), cudaMemcpyDeviceToHost); 
    for(k=0; k<KYBER_K*KYBER_K; k++)                  
      rej_uniform_gpu<<<BATCH, KYBER_N>>>(d_at+k*KYBER_N, KYBER_N, d_buf+k*4*KYBER_N);
    // cudaMemcpy(at, d_at, KYBER_K*KYBER_K*BATCH*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    // for(i=0; i<1; i++)
    // {
    //   printf("\nbatch %u\n", i);
    //   for(j=0; j<KYBER_K*KYBER_K*KYBER_N; j++)
    //   {
    //     printf("%u,  ", at[i*KYBER_K*KYBER_K*KYBER_N + j]);
    //   }  
    // } 
#else
    // Use this to verify results in decapsulation (Keccak)
    // for(i=0; i<BATCH; i++)  for(k=0; k<KYBER_K; k++)
    //   for(hh=0; hh<KYBER_K; hh++)
    //     for(j=0; j<KYBER_N; j++) at[i*KYBER_K*KYBER_K*KYBER_N + k*KYBER_K*KYBER_N + hh*KYBER_N + j] =256; 
    for(i=0; i<KYBER_K*KYBER_K*BATCH*KYBER_N ; i++) at[i] = 256;
    cudaMemcpy(d_at, at, KYBER_K*KYBER_K*BATCH*KYBER_N * sizeof(int16_t), cudaMemcpyHostToDevice);
#endif

#ifdef AESGPU
     // KYBER_K=2, 0-31: sp, 32-63: ep, 64-79: epp
  // Generate 128*5 random bytes; however minimum 256 threads
  encGPUsharedFineGrain2<<<BATCH, 288>>> (d_prf, d_exp_seed);
  // AES_256_encrypt<<<BATCH, 40>>>(d_prf, d_seed, 0) ; // 40 for KYBER_K=2, 72 for KYBER_K=3
  // cudaMemcpy(tmp_prf, d_prf, (2*KYBER_K+1)*BATCH*KYBER_ETA*KYBER_N/4* sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // for(i=0; i<BATCH; i++)
  //   for(j=0; j<5*KYBER_ETA*KYBER_N/4; j++)
  //   {
  //     if(tmp_prf[j] != tmp_prf[i*5*KYBER_ETA*KYBER_N/4 + j])
  //       printf("wrong at batch %u no. %u: %d %d\n", i, j, tmp_prf[j], tmp_prf[i*5*KYBER_ETA*KYBER_N/4+ j]);
  //   }      
    // for(i=0; i<2; i++)
    // {
    //   printf("\nbatch %u\n", i);
    //   for(j=0; j<(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4; j++)
    //   {
    //     printf(" %u", tmp_prf[i*(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4 + j]);
    //   }  
    // }   
#else  
    for(i=0; i<BATCH; i++) 
      for(j=0; j<(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4; j++) 
      tmp_prf[i*(2*KYBER_K+1)*KYBER_ETA*KYBER_N/4 +j] = 128;   
    // for(i=0; i<BATCH; i++) for(j=0; j<1024; j++) 
      // tmp_prf[i*1024 +j] = j;   
    cudaMemcpy(d_prf, tmp_prf, (2*KYBER_K+1)*BATCH*KYBER_ETA*KYBER_N/4* sizeof(uint8_t), cudaMemcpyHostToDevice);
#endif
    for(i=0; i<2*KYBER_K+1; i++)
      cbd_gpu2<<<BATCH, KYBER_N>>>(d_r+i*KYBER_N, d_prf+i*KYBER_ETA*KYBER_N/4);     // fast
      // cbd_gpu<<<BATCH, KYBER_N/8>>>(d_r+i*KYBER_N, d_prf+i*KYBER_ETA*KYBER_N/4); // slow    
    
    poly_veccopy<<<BATCH, KYBER_N>>>(d_sp, d_r);
      // polyvec_ntt<<<BATCH, KYBER_N/2>>>(d_sp); //slow
    polyvec_ntt2<<<BATCH, KYBER_N/4>>>(d_sp);   //fast
    polyvec_pointwise_acc_montgomery_gpu<<<BATCH, KYBER_N/4>>>(d_bp, d_t, d_at, d_sp);    
    polyvec_pointwise_acc_montgomery_gpu2<<<BATCH, KYBER_N/4>>>(d_v, d_t, d_pk, d_sp);    
    polyvec_invntt_tomont_red_gpu<<<BATCH, KYBER_N/2>>>(d_bp, KYBER_K);
    poly_invntt_tomont_gpu<<<BATCH, KYBER_N/2>>>(d_v);            
    poly_veccopy<<<BATCH, KYBER_N>>>(d_ep, d_r+2*KYBER_N);
    polycopy<<<BATCH, KYBER_N>>>(d_epp, d_r+4*KYBER_N);
    polyvec_add_gpu3<<<BATCH, KYBER_N>>>(d_bp, d_ep, d_bp);// poly_vec_add
    polyvec_reduce_gpu<<<BATCH, KYBER_N>>>(d_bp);      
    polyvec_add_gpu2<<<BATCH, KYBER_N>>>(d_v, d_v, d_epp);// poly_add
    polyvec_add_gpu2<<<BATCH, KYBER_N>>>(d_v, d_v, d_msgpoly);
    poly_reduce_g<<<BATCH, KYBER_N>>>(d_v);
#if KYBER_K == 4
    polyvec_compress_gpu<<<BATCH, KYBER_N/8>>>(d_c, d_bp);
#else
    polyvec_compress_gpu<<<BATCH, KYBER_N/4>>>(d_c, d_bp);
#endif
    poly_compress_gpu<<<BATCH, KYBER_N/8>>>(d_c+KYBER_POLYVECCOMPRESSEDBYTES, d_v);

    // cudaMemcpy(b, d_r, BATCH*(2*KYBER_K+1)*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(b, d_bp, BATCH*KYBER_K*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(b, d_v, BATCH*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, BATCH*KYBER_INDCPA_BYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }    
  cudaEventRecord(stop);     
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\n\nGPU KYBER-128 Encrypt time: %.6f ms average: %.6f ms\n", milliseconds, milliseconds/REPEAT/BATCH);
// ***********************************************************

#ifdef DEBUG
    for(i=0; i<BATCH; i++)
      for(j=0; j<KYBER_INDCPA_BYTES; j++)
      {
        if(c[j] != c[i*KYBER_INDCPA_BYTES + j])
            printf("-wrong at batch %u no. %u: %d %d\n", i, j, c[j],c[i*KYBER_INDCPA_BYTES + j]);          
      }
    // printf("\nc: \n"); 

    // for(i=0;i< BATCH;i++)
    // {
    //   printf("\nbatch: %d\n", i);
    //   for(j=0; j<(2*KYBER_K+1)*KYBER_N; j++) printf("%d, ", b[i*(2*KYBER_K+1)*KYBER_N + j]);
    // }
    // for(j=0; j<KYBER_INDCPA_BYTES; j++) printf("%u ", c[j]);
    // for(j=0; j<KYBER_K*KYBER_N; j++) printf("%d ", b[j]);
      // for(j=0; j<KYBER_N; j++) printf("%d ", b[j]);
#endif

    // for(i=0; i<1; i++)
    // {
    //   printf("\nbatch %u\n", i);
    //   for(j=0; j<KYBER_K*KYBER_N; j++)
    //   {
    //     printf("%d ", b[i* KYBER_K*KYBER_N + j]);
    //   }  
    // }
    // for(i=0; i<BATCH; i++)
    //   for(j=0; j<KYBER_N; j++)
    //   {
    //     if(b[j] != b[i* KYBER_N + j])
    //        printf("wrong at batch %u no. %u: %d %d\n", i, j, b[j], b[i*KYBER_N + j]);
    //   } 
    // for(i=0; i<BATCH; i++)
    //   for(j=0; j<KYBER_K*KYBER_N; j++)
    //   {
    //     if(b[j] != b[i* KYBER_K*KYBER_N + j])
    //        printf("wrong at batch %u no. %u: %d %d\n", i, j, b[j], b[i*KYBER_K*KYBER_N + j]);
    //   } 
 }

 void indcpa_dec_gpu(uint8_t * c)
 {
 	  uint8_t *d_c;
 	  uint32_t i, j, k, hh, count;
 	// variables for decapsulation
    int16_t *d_bp_dec, *d_v_dec, *d_sk_poly, *d_mp, *d_t_dec;
    int16_t *bp_dec, *v_dec, *mp, *sk_poly_dec;
    uint8_t *d_m_dec, *m_dec;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);   cudaEventCreate(&stop);

    cudaMallocHost((void**) &bp_dec, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMallocHost((void**) &v_dec, BATCH*KYBER_N * sizeof(int16_t)); 
    cudaMallocHost((void**) &sk_poly_dec, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMallocHost((void**) &mp, BATCH*KYBER_N * sizeof(int16_t));
    cudaMallocHost((void**) &m_dec, BATCH*KYBER_INDCPA_MSGBYTES * sizeof(uint8_t));

    // Allocate device memory.
     cudaMalloc((void**) &d_bp_dec, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMalloc((void**) &d_v_dec, BATCH*KYBER_N * sizeof(int16_t)); 
    cudaMalloc((void**) &d_sk_poly, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMalloc((void**) &d_mp, BATCH*KYBER_N * sizeof(int16_t));
    cudaMalloc((void**) &d_t_dec, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));
    cudaMalloc((void**) &d_m_dec, BATCH*KYBER_INDCPA_MSGBYTES * sizeof(uint8_t));
 	  cudaMalloc((void**) &d_c, BATCH*KYBER_INDCPA_BYTES * sizeof(uint8_t)); 
	  cudaMalloc((void**) &d_bp_dec, KYBER_K*BATCH*KYBER_N * sizeof(int16_t));

 	  printf("\n\nDecryption\n");
#ifdef AESGPU
    for(i=0; i<BATCH; i++) for(hh=0; hh<KYBER_K; hh++)
      for(j=0; j<KYBER_N; j++) sk_poly_dec[i*KYBER_N*KYBER_K + hh * KYBER_N + j] = sk_poly_aes[hh * KYBER_N + j] ;
#else
    for(i=0; i<BATCH; i++) for(hh=0; hh<KYBER_K; hh++)
      for(j=0; j<KYBER_N; j++) sk_poly_dec[i*KYBER_N*KYBER_K + hh * KYBER_N + j] = sk_poly[hh * KYBER_N + j] ;
#endif
  cudaEventRecord(start);   
  for(count=0; count<REPEAT; count++)     
  { 
 	  cudaMemcpy(d_c, c, BATCH*KYBER_INDCPA_BYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk_poly, sk_poly_dec, BATCH*KYBER_K*KYBER_N * sizeof(int16_t), cudaMemcpyHostToDevice);
#if KYBER_K == 4
  	polyvec_decompress_gpu<<<BATCH, KYBER_N/8>>>(d_bp_dec, d_c);
#else
    polyvec_decompress_gpu<<<BATCH, KYBER_N/4>>>(d_bp_dec, d_c);
#endif     
#if KYBER_K == 3 
  	poly_decompress_gpu<<<BATCH, KYBER_N/2>>>(d_v_dec, d_c);  	
#else
    poly_decompress_gpu<<<BATCH, KYBER_N/8>>>(d_v_dec, d_c);      
#endif
    // polyvec_ntt<<<BATCH, KYBER_N/2>>>(d_bp_dec);// not used
    polyvec_ntt2<<<BATCH, KYBER_N/4>>>(d_bp_dec);
  	polyvec_pointwise_acc_montgomery_gpu2<<<BATCH, KYBER_N/4>>>(d_mp, d_t_dec, d_sk_poly, d_bp_dec);
  	poly_invntt_tomont_gpu<<<BATCH, KYBER_N/2>>>(d_mp);  
  	poly_sub_gpu<<<BATCH, KYBER_N>>>(d_mp, d_v_dec, d_mp);
  	poly_reduce_g<<<BATCH, KYBER_N>>>(d_mp);
  	poly_tomsg_gpu<<<BATCH, KYBER_N>>>(d_m_dec, d_mp);
  	cudaMemcpy(m_dec, d_m_dec, BATCH*KYBER_INDCPA_MSGBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
  }
    
  cudaEventRecord(stop);     
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("\n\nGPU KYBER-128 Decrypt time: %.6f ms average: %.6f\n", milliseconds, milliseconds/REPEAT/BATCH);
  	// cudaMemcpy(bp_dec, d_bp_dec, BATCH*KYBER_K*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(mp, d_mp, BATCH*KYBER_N * sizeof(int16_t), cudaMemcpyDeviceToHost);
    for(i=0; i<BATCH; i++)
    {
        for(j=0; j<KYBER_N; j++)
        {
         if(mp[j] != mp[i*KYBER_N + j])
           printf("wrong at batch %u no. %u: %d %d\n", i, j, mp[j], mp[i*KYBER_N + j]);        
       }
    } 
    // for(j=0; j<KYBER_N; j++) printf("%d ", mp[j]);
    // for(i=0; i<1; i++)
    // {
    //   printf("\nbatch %u\n", i);
    //   for(j=0; j<KYBER_K* KYBER_N; j++)  printf("%d ", bp_dec[i*KYBER_K * KYBER_N + j]);
    // }
  	// for(i=0; i<BATCH; i++)
  	// {
   //  	// printf("\nbatch %u\n", i);    
   //  	// for(j=0; j<KYBER_INDCPA_MSGBYTES; j++)  printf("%d, ", m_dec[i*KYBER_INDCPA_MSGBYTES + j]);
   //    for(j=0; j<KYBER_INDCPA_MSGBYTES; j++)
   //    {
   //      if(m_dec[j] != m_dec[i*KYBER_INDCPA_MSGBYTES  + j])
   //        printf("dec wrong at batch %u no. %u: %d %d\n", i, j, m_dec[j], m_dec[i*KYBER_INDCPA_MSGBYTES  + j]);  
   //    }
  	// }
    printf("\n");
  for(i=0; i<2; i++)
  {
      printf("\nbatch %u\n", i);
	   for(j=0; j<KYBER_INDCPA_MSGBYTES; j++)  printf("%d, ", m_dec[i*KYBER_INDCPA_MSGBYTES + j]);
  }
printf("\n");
printf("Batch Size %d", BATCH);
printf("\n");
}