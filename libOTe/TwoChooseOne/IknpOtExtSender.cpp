#include "IknpOtExtSender.h"
#ifdef ENABLE_IKNP
#include "libOTe/Tools/Tools.h"
#include <cryptoTools/Common/Log.h>
#include <cryptoTools/Crypto/Commit.h>
#include <cryptoTools/Network/Channel.h>

#include "TcoOtDefines.h"

#include <immintrin.h>

#include <iostream>

namespace osuCrypto
{
    using namespace std;


    IknpOtExtSender IknpOtExtSender::splitBase()
    {
        std::array<block, gOtExtBaseOtCount> baseRecvOts;

        if (!hasBaseOts())
            throw std::runtime_error("base OTs have not been set. " LOCATION);

        for (u64 i = 0; i < mGens.size(); ++i)
            baseRecvOts[i] = mGens[i].get<block>();

        return IknpOtExtSender(baseRecvOts, mBaseChoiceBits);
    }

    std::unique_ptr<OtExtSender> IknpOtExtSender::split()
    {
        std::array<block, gOtExtBaseOtCount> baseRecvOts;

        for (u64 i = 0; i < mGens.size(); ++i)
            baseRecvOts[i] = mGens[i].get<block>();

        return std::make_unique<IknpOtExtSender>(baseRecvOts, mBaseChoiceBits);
    }

    void IknpOtExtSender::setBaseOts(span<block> baseRecvOts, const BitVector & choices)
    {
        if (baseRecvOts.size() != gOtExtBaseOtCount || choices.size() != gOtExtBaseOtCount)
            throw std::runtime_error("not supported/implemented");

        mBaseChoiceBits = choices;
        for (u64 i = 0; i < gOtExtBaseOtCount; i++)
        {
            mGens[i].SetSeed(baseRecvOts[i]);
        }
    }

    __attribute__((target("avx512f,vaes,avx512vl")))
    void IknpOtExtSender::send(
        span<std::array<block, 2>> messages,
        PRNG& prng,
        Channel& chl)
    {

        if (hasBaseOts() == false)
            genBaseOts(prng, chl);

        // round up
        u64 numOtExt = roundUpTo(messages.size(), 128);
        u64 numSuperBlocks = (numOtExt / 128 + superBlkSize - 1) / superBlkSize;
        //u64 numBlocks = numSuperBlocks * superBlkSize;

        // a temp that will be used to transpose the sender's matrix
        std::array<std::array<block, superBlkSize>, 128> t;
        std::vector<std::array<block, superBlkSize>> u(128 * commStepSize);

        std::array<block, 128> choiceMask;
        block delta = *(block*)mBaseChoiceBits.data();

        for (u64 i = 0; i < 128; ++i)
        {
            if (mBaseChoiceBits[i]) choiceMask[i] = AllOneBlock;
            else choiceMask[i] = ZeroBlock;
        }

        auto mIter = messages.begin();

        block * uIter = (block*)u.data() + superBlkSize * 128 * commStepSize;
        block * uEnd = uIter;

        for (u64 superBlkIdx = 0; superBlkIdx < numSuperBlocks; ++superBlkIdx)
        {


            block * tIter = (block*)t.data();
            block * cIter = choiceMask.data();

            if (uIter == uEnd)
            {
                u64 step = std::min<u64>(numSuperBlocks - superBlkIdx, (u64)commStepSize);

                chl.recv((u8*)u.data(), step * superBlkSize * 128 * sizeof(block));
                uIter = (block*)u.data();
            }

            const __m512i ctr_offset = _mm512_set_epi64(0, 3, 0, 2, 0, 1, 0, 0);
            const __m512i add_offset = _mm512_set_epi64(0, 4, 0, 0, 0, 4, 0, 4);

            // transpose 128 columns at at time. Each column will be 128 * superBlkSize = 1024 bits long.
            for (u64 colIdx = 0; colIdx < 128; colIdx+=2)
            {
                // TODO: Only do 8 blocks, but from two subsequent iterations!
                __m512i ctData[2];
                __m512i ntData[2];
                __m512i ctKeys[11];
                __m512i ntKeys[11];
                __m512i ctctr = _mm512_set_epi64(0, mGens[colIdx].mBlockIdx, 0, mGens[colIdx].mBlockIdx, 0, mGens[colIdx].mBlockIdx, 0, mGens[colIdx].mBlockIdx);
                __m512i ntctr = _mm512_set_epi64(0, mGens[colIdx+1].mBlockIdx, 0, mGens[colIdx + 1].mBlockIdx, 0, mGens[colIdx + 1].mBlockIdx, 0, mGens[colIdx + 1].mBlockIdx);
                ctctr = _mm512_add_epi64(ctctr, ctr_offset);
                ntctr = _mm512_add_epi64(ntctr, ctr_offset);

                for (size_t i = 0; i < 11; ++i) {
                    ctKeys[i] = _mm512_broadcast_i32x4(mGens[colIdx].mAes.mRoundKey[i]);
                    ntKeys[i] = _mm512_broadcast_i32x4(mGens[colIdx+1].mAes.mRoundKey[i]);
                }

                for (size_t w = 0; w < 2; ++w)
                {
                    ctData[w] = ctctr;
                    ntData[w] = ntctr;
                    ctctr = _mm512_add_epi64(ctctr, add_offset);
                    ntctr = _mm512_add_epi64(ntctr, add_offset);

                    ctData[w] = _mm512_xor_si512(ctData[w], ctKeys[0]);
                    ntData[w] = _mm512_xor_si512(ntData[w], ntKeys[0]);
                }

                for (size_t r = 1; r < 10; ++r)
                    for (size_t w = 0; w < 2; ++w)
                    {
                        ctData[w] = _mm512_aesenc_epi128(ctData[w], ctKeys[r]);
                        ntData[w] = _mm512_aesenc_epi128(ntData[w], ntKeys[r]);
                    }

                for (size_t w = 0; w < 2; ++w) {
                    ctData[w] = _mm512_aesenclast_epi128(ctData[w], ctKeys[10]);
                    ntData[w] = _mm512_aesenclast_epi128(ntData[w], ntKeys[10]);
                    __m128i ccData = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cIter));
                    __m128i ncData = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cIter+1));
                    __m512i cuData = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(uIter + 4 * w));
                    __m512i nuData = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(uIter + 8 + 4 * w));
                    __m512i widecCData = _mm512_broadcast_i32x4(ccData);
                    __m512i widenCData = _mm512_broadcast_i32x4(ncData);

                    cuData = _mm512_and_si512(cuData, widecCData);
                    nuData = _mm512_and_si512(nuData, widenCData);
                    ctData[w] = _mm512_xor_si512(ctData[w], cuData);
                    ntData[w] = _mm512_xor_si512(ntData[w], nuData);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(uIter + 4 * w), cuData);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(tIter + 4 * w), ctData[w]);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(uIter + 8 + 4 * w), nuData);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(tIter + 8+ 4 * w), ntData[w]);
                }

                // generate the columns using AES-NI in counter mode.
                mGens[colIdx].mBlockIdx += superBlkSize;
                mGens[colIdx+1].mBlockIdx += superBlkSize;

                cIter += 2;
                uIter += 16;
                tIter += 16;
            }

            // transpose our 128 columns of 1024 bits. We will have 1024 rows,
            // each 128 bits wide.
            transpose128x1024(t);


            auto mEnd = mIter + std::min<u64>(128 * superBlkSize, messages.end() - mIter);

            tIter = (block*)t.data();
            block* tEnd = (block*)t.data() + 128 * superBlkSize;

            while (mIter != mEnd)
            {
                while (mIter != mEnd && tIter < tEnd)
                {
                    (*mIter)[0] = *tIter;
                    (*mIter)[1] = *tIter ^ delta;

                    tIter += superBlkSize;
                    mIter += 1;
                }

                tIter = tIter - 128 * superBlkSize + 1;
            }


#ifdef IKNP_DEBUG
            BitVector choice(128 * superBlkSize);
            chl.recv(u.data(), superBlkSize * 128 * sizeof(block));
            chl.recv(choice.data(), sizeof(block) * superBlkSize);

            u64 doneIdx = mStart - messages.data();
            u64 xx = std::min<u64>(i64(128 * superBlkSize), (messages.data() + messages.size()) - mEnd);
            for (u64 rowIdx = doneIdx,
                j = 0; j < xx; ++rowIdx, ++j)
            {
                if (neq(((block*)u.data())[j], messages[rowIdx][choice[j]]))
                {
                    std::cout << rowIdx << std::endl;
                    throw std::runtime_error("");
                }
            }
#endif
        }

#ifdef IKNP_SHA_HASH
        RandomOracle sha;
        u8 hashBuff[20];
        u64 doneIdx = 0;


        u64 bb = (messages.size() + 127) / 128;
        for (u64 blockIdx = 0; blockIdx < bb; ++blockIdx)
        {
            u64 stop = std::min<u64>(messages.size(), doneIdx + 128);

            for (u64 i = 0; doneIdx < stop; ++doneIdx, ++i)
            {
                // hash the message without delta
                sha.Reset();
                sha.Update((u8*)&messages[doneIdx][0], sizeof(block));
                sha.Final(hashBuff);
                messages[doneIdx][0] = *(block*)hashBuff;

                // hash the message with delta
                sha.Reset();
                sha.Update((u8*)&messages[doneIdx][1], sizeof(block));
                sha.Final(hashBuff);
                messages[doneIdx][1] = *(block*)hashBuff;
            }
        }
#else
        __m512i fixed_key[11];
        for (size_t i = 0; i < 11; ++i)
            fixed_key[i] = _mm512_broadcast_i32x4(mAesFixedKey.mRoundKey[i]);

        std::array<block, 8> aesHashTemp;

        u64 doneIdx = 0;
        u64 bb = (messages.size() + 127) / 128;
        for (u64 blockIdx = 0; blockIdx < bb; ++blockIdx)
        {
            u64 stop = std::min<u64>(messages.size(), doneIdx + 128);

            auto length = 2 * (stop - doneIdx);
            auto steps = length / 16; 
            
            block* mIter = messages[doneIdx].data();

            for (u64 i = 0; i < steps; ++i)
            {
                __m512i data[4];
                __m512i whitening[4];

                for (size_t w = 0; w < 4; ++w) {
                    data[w] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(mIter + 4 * w));
                    whitening[w] = data[w];
                    data[w] = _mm512_xor_si512(data[w], fixed_key[0]);
                }

                for (size_t r = 1; r < 10; ++r)
                    for (size_t w = 0; w < 4; ++w)
                        data[w] = _mm512_aesenc_epi128(data[w], fixed_key[r]);

                for (size_t w = 0; w < 4; ++w) {
                    data[w] = _mm512_aesenclast_epi128(data[w], fixed_key[10]);
                    data[w] = _mm512_xor_si512(data[w], whitening[w]);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(mIter + 4 * w), data[w]);
                }

                mIter += 16;
            }

            auto rem = length - steps * 8;
            mAesFixedKey.ecbEncBlocks(mIter, rem, aesHashTemp.data());
            for (u64 i = 0; i < rem; ++i)
            {
                mIter[i] = mIter[i] ^ aesHashTemp[i];
            }

            doneIdx = stop;
        }

#endif

        static_assert(gOtExtBaseOtCount == 128, "expecting 128");
    }


}
#endif