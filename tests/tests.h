
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

} // namespace Testing



#endif // __dealii_qc_tests_h
