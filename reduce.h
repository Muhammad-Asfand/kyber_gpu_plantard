#ifndef REDUCE_H
#define REDUCE_H

#include <stdint.h>
#include "../include/params.h"

#define MONT 2285 // 2^16 mod q
#define QINV 62209 // q^-1 mod 2^16
#define R64 4294967296	// For lazy reduction
#define R64_INV 1929	// R64_INV = multinv(R64) % Q
#define Q_INV64 1806234369	// or 2488732927 Q*Q_INV64 % R64 = -1 ==> Q_INV64 = -multinv(Q) % R64

// #define montgomery_reduce KYBER_NAMESPACE(_montgomery_reduce)
int16_t montgomery_reduce(int32_t a);

// #define barrett_reduce KYBER_NAMESPACE(_barrett_reduce)
int16_t barrett_reduce(int16_t a);

// #define csubq KYBER_NAMESPACE(_csubq)
int16_t csubq(int16_t x);
void unpack_pk(int16_t *pk, uint8_t seed[KYBER_SYMBYTES],
                      const uint8_t packedpk[KYBER_INDCPA_PUBLICKEYBYTES]);
unsigned int rej_uniform(int16_t *r, unsigned int len, const uint8_t *buf,
                                unsigned int buflen);

#endif
