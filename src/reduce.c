#include <stdint.h>
#include <stdio.h>
#include "../include/params.h"
#include "../include/reduce.h"


/*************************************************
* Name:        montgomery_reduce
*
* Description: Montgomery reduction; given a 32-bit integer a, computes
*              16-bit integer congruent to a * R^-1 mod q,
*              where R=2^16
*
* Arguments:   - int32_t a: input integer to be reduced;
*                           has to be in {-q2^15,...,q2^15-1}
*
* Returns:     integer in {-q+1,...,q-1} congruent to a * R^-1 modulo q.
**************************************************/
int16_t montgomery_reduce(int32_t a)
{
  int32_t t;
  int16_t u;

  u = a*QINV;
  t = (int32_t)u*KYBER_Q;
  t = a - t;
  t >>= 16;
  return t;
}

/*************************************************
* Name:        barrett_reduce
*
* Description: Barrett reduction; given a 16-bit integer a, computes
*              16-bit integer congruent to a mod q in {0,...,q}
*
* Arguments:   - int16_t a: input integer to be reduced
*
* Returns:     integer in {0,...,q} congruent to a modulo q.
**************************************************/
int16_t barrett_reduce(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q/2)/KYBER_Q;

  t  = (int32_t)v*a >> 26;
  t *= KYBER_Q;
  return a - t;
}

/*************************************************
* Name:        csubq
*
* Description: Conditionallly subtract q
*
* Arguments:   - int16_t x: input integer
*
* Returns:     a - q if a >= q, else a
**************************************************/
int16_t csubq(int16_t a) {
  a -= KYBER_Q;
  a += (a >> 15) & KYBER_Q;
  return a;
}

void unpack_pk(int16_t *pk,
                      uint8_t seed[KYBER_SYMBYTES],
                      const uint8_t packedpk[KYBER_INDCPA_PUBLICKEYBYTES])
{
  uint32_t i = 0, j = 0;
  
  for(i=0;i<KYBER_K;i++)
  {    
    for(j=0;j<KYBER_N/2;j++) {
      pk[i * KYBER_N + 2*j]   = ((packedpk[i * KYBER_POLYBYTES + 3*j+0] >> 0) | ((uint16_t)packedpk[i * KYBER_POLYBYTES + 3*j+1] << 8)) & 0xFFF;
      pk[i * KYBER_N + 2*j+1] = ((packedpk[i * KYBER_POLYBYTES + 3*j+1] >> 4) | ((uint16_t)packedpk[i * KYBER_POLYBYTES + 3*j+2] << 4)) & 0xFFF;
    }
    // printf("\nUnpack: %u\n", i); for(j=0;j<KYBER_N;j++) printf("%u ", pubk[i * KYBER_N + j]);    
  }

  // polyvec_frombytes(pubk, packedpk);
  for(i=0;i<KYBER_SYMBYTES;i++)
    seed[i] = packedpk[i+KYBER_POLYVECBYTES];  
}

unsigned int rej_uniform(int16_t *r, unsigned int len, const uint8_t *buf, unsigned int buflen)
{
  unsigned int ctr, pos, i;
  uint16_t val;
  ctr = pos = 0;
  while(ctr < len && pos + 2 <= buflen) {
    val = buf[pos] | ((uint16_t)buf[pos+1] << 8);
    pos += 2;

    if(val < 19*KYBER_Q) {
      val -= (val >> 12)*KYBER_Q; // Barrett reduction
      r[ctr++] = (int16_t)val;      
    }
  }  

  // Iterative version
  // for(pos=0; pos<buflen; pos+=2)
  // {
  //   val = buf[pos] | ((uint16_t)buf[pos+1] << 8);
  //   if(val < 19*KYBER_Q) {
  //     val -= (val >> 12)*KYBER_Q; // Barrett reduction
  //     r[ctr++] = (int16_t)val;      
  //   }
  //   if(ctr>5) break;
  // }

  return ctr;
};

// void shake256(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen)
// {
//   unsigned int i;
//   size_t nblocks = outlen/SHAKE256_RATE;
//   uint8_t t[SHAKE256_RATE];
//   keccak_state state;

//   shake256_absorb(&state, in, inlen);
//   shake256_squeezeblocks(out, nblocks, &state);

//   out += nblocks*SHAKE256_RATE;
//   outlen -= nblocks*SHAKE256_RATE;

//   if(outlen) {
//     shake256_squeezeblocks(t, 1, &state);
//     for(i=0;i<outlen;i++)
//       out[i] = t[i];
//   }
// }

// void kyber_shake256_prf(uint8_t *out,
//                         size_t outlen,
//                         const uint8_t key[KYBER_SYMBYTES],
//                         uint8_t nonce)
// {
//   unsigned int i;
//   uint8_t extkey[KYBER_SYMBYTES+1];

//   for(i=0;i<KYBER_SYMBYTES;i++)
//     extkey[i] = key[i];
//   extkey[i] = nonce;
  
//   shake256(out, outlen, extkey, sizeof(extkey));
// }