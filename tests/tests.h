
#ifndef __dealii_qc_tests_h
#define __dealii_qc_tests_h

#include <limits>
#include <type_traits>
#include <algorithm>



namespace Testing
{

  // Return true if the two floating point numbers x and y are  close to each
  // other.
  template<typename T>
  typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  almost_equal (const T &x,
                const T &y,
                const unsigned int ulp)
  {
    // The machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= (std::numeric_limits<T>::epsilon()
                              * std::max(std::fabs(x), std::fabs(y))
                              * ulp)
           ||
           std::fabs(x-y) < std::numeric_limits<T>::min();
  }


  // A direct copy from deal.II test.h file
  // for pseudo-random int generation.
  int rand(const bool reseed=false,
           const int seed=1)
  {
    static int r[32];
    static int k;
    static bool inited=false;
    if (!inited || reseed)
      {
        //srand treats a seed 0 as 1 for some reason
        r[0]=(seed==0)?1:seed;

        for (int i=1; i<31; i++)
          {
            r[i] = (16807LL * r[i-1]) % 2147483647;
            if (r[i] < 0)
              r[i] += 2147483647;
          }
        k=31;
        for (int i=31; i<34; i++)
          {
            r[k%32] = r[(k+32-31)%32];
            k=(k+1)%32;
          }

        for (int i=34; i<344; i++)
          {
            r[k%32] = r[(k+32-31)%32] + r[(k+32-3)%32];
            k=(k+1)%32;
          }
        inited=true;
        if (reseed==true)
          return 0;// do not generate new no
      }

    r[k%32] = r[(k+32-31)%32] + r[(k+32-3)%32];
    int ret = r[k%32];
    k=(k+1)%32;
    return (unsigned int)ret >> 1;
  }

} // namespace Testing



#endif // __dealii_qc_tests_h
