#pragma once
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <cryptoTools/Network/Channel.h>
#include <libOTe/DPF/BgiEvaluator.h>
#include <cryptoTools/Common/Timer.h>
#include <libOTe/DPF/BgicksPprf.h>
#include <libOTe/TwoChooseOne/TcoOtDefines.h>

namespace osuCrypto
{
    void bitShiftXor(span<block> dest, span<block> in, u8 bitShift);
    void modp(span<block> dest, span<block> in, u64 p);

    class BgciksOtExtSender : public TimerAdapter
    {
    public:

        //BgiEvaluator::MultiKey mGenBgi;
        BgicksMultiPprfSender mGen;
        block mDelta;
        u64 mP, mN2, mN, mNumPartitions, mScaler, mSizePer;
		bool mMal;

        bool mDebug = true;
        void checkRT(span<Channel> chls, Matrix<block>& rT);



        //BitVector mS, mC;
		void genBase(u64 n, Channel& chl, PRNG& prng, u64 scaler = 4, u64 secParam = 80, bool mal = false, BgciksBaseType base = BgciksBaseType::None,
			u64 threads = 1);
		//void genBase(u64 n, span<Channel> chls, PRNG& prng, u64 scaler = 4, u64 secParam = 80, BgciksBaseType base = BgciksBaseType::None);

		void configure(const u64& n, const u64& scaler, const u64& secParam, bool mal);

        void send(
            span<std::array<block, 2>> messages,
            PRNG& prng,
            Channel& chl);

		void send(
			span<std::array<block, 2>> messages,
			PRNG& prng,
			span<Channel> chls);

        void randMulNaive(Matrix<block>& rT, span<std::array<block, 2>>& messages);
        void randMulQuasiCyclic(Matrix<block>& rT, span<std::array<block, 2>>& messages, u64 threads);
    };

}