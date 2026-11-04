#pragma once
#include <array>
#include <cstdint>

namespace sts::search {

// XXH64 over a stream of little-endian uint64 words, with no byte buffer or
// allocation. This is an in-memory search fingerprint, not a cryptographic
// digest or a persisted NOSL table identity. All fields and delimiters are
// supplied by the existing publicStateKey visitor.
// Algorithm: Yann Collet, xxHash v0.8.2, doc/xxhash_spec.md (XXH64).
class WordHash64 {
    using U = std::uint64_t;
    static constexpr U P1=0x9E3779B185EBCA87ULL, P2=0xC2B2AE3D27D4EB4FULL;
    static constexpr U P3=0x165667B19E3779F9ULL, P4=0x85EBCA77C2B2AE63ULL;
    static constexpr U P5=0x27D4EB2F165667C5ULL;
    U seed_, count_=0;
    std::array<U,4> lanes_, pending_{};
    static U rotate(U x, unsigned n) { return (x<<n)|(x>>(64-n)); }
    static U round(U accumulator, U word) {
        return rotate(accumulator+word*P2,31)*P1;
    }
public:
    WordHash64(U seed):seed_(seed),lanes_{seed+P1+P2,seed+P2,seed,seed-P1} {}
    void add(U word) {
        pending_[count_++ & 3]=word;
        if((count_ & 3)==0) {
            lanes_[0]=round(lanes_[0],pending_[0]);
            lanes_[1]=round(lanes_[1],pending_[1]);
            lanes_[2]=round(lanes_[2],pending_[2]);
            lanes_[3]=round(lanes_[3],pending_[3]);
        }
    }
    operator U() const {
        U h=seed_+P5;
        if(count_>=4) {
            h=rotate(lanes_[0],1)+rotate(lanes_[1],7)+rotate(lanes_[2],12)+rotate(lanes_[3],18);
            for(auto lane:lanes_) h=(h^round(0,lane))*P1+P4;
        }
        h+=count_*8;
        for(U i=0;i<(count_&3);++i) h=rotate(h^round(0,pending_[i]),27)*P1+P4;
        h^=h>>33;h*=P2;h^=h>>29;h*=P3;h^=h>>32;
        return h;
    }
};
}
