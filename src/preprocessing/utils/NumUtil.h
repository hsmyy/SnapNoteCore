#ifndef GAUSSIAN_SP_DENOISE_SRC_NUMUTIL_H_
#define GAUSSIAN_SP_DENOISE_SRC_NUMUTIL_H_

class FloatBits
{
    struct Bits
    {
        unsigned M : 23;
        unsigned E : 8;
        unsigned S : 1;
    };
private:
    Bits bits;
public:
    FloatBits(float f)
    {
        *((float*)this) = f;
    }

    operator float& ()
    {
        return *reinterpret_cast<float*>(&bits);
    }

    void setS(int s)
    {
        bits.S = s;
    }

    int getS()
    {
        return bits.S;
    }

    void setE(int e)
    {
        bits.E = e + 127;
    }

    int getE()
    {
        return bits.E - 127;
    }

    void setM(int m)
    {
        bits.M = m;
    }

    int getM()
    {
        return bits.M;
    }
};


class DoubleBits
{
    struct Bits
    {
        unsigned ML : 32;
        unsigned MH : 20;
        unsigned E : 11;
        unsigned S : 1;
    };
private:
    Bits bits;
public:
    DoubleBits(double d)
    {
        *((double*)this) = d;
    }

    DoubleBits(unsigned long long l)
    {
        *((unsigned long long*)this) = l;
    }

    operator double& ()
    {
        return *reinterpret_cast<double*>(&bits);
    }

    operator unsigned long long& ()
    {
        return *reinterpret_cast<unsigned long long*>(&bits);
    }

    void setS(int s)
    {
        bits.S = s;
    }

    int getS()
    {
        return bits.S;
    }

    void setE(int e)
    {
        bits.E = e + 1023;
    }

    int getE()
    {
        return bits.E - 1023;
    }

    void setM(unsigned long long m)
    {
        const unsigned long long mask = 0xfff0000000000000;
        *((unsigned long long*)this) &= mask;
        m &= (~mask);
        *((unsigned long long*)this) |= m;
    }

    unsigned long long getM()
    {
        const unsigned long long mask = 0xfff0000000000000;
        return (unsigned long long)(*this) & (~mask);
    }
};


float eps(float innum)
{
    FloatBits fb1(innum), fb2(innum);
    fb1.setM(1);
    fb2.setM(0);
    fb1.setS(0);
    fb2.setS(0);
    return float(fb1) - float(fb2);
}

double eps(double innum)
{
    DoubleBits db1(innum), db2(innum);
    db1.setM(1);
    db2.setM(0);
    db1.setS(0);
    db2.setS(0);
    return double(db1) - double(db2);
}
#endif
