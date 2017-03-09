
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

// typedefs

// When the number of atoms is large and exceeds maximum value
// allowed with an `unsgined int` (system dependent limit)
//typedef unsigned long int uint_t;

// TODO: Use of correct charge units; Use charge_t for book keeping
// for now just use float (float takes less time to compute)
// (charge of atoms don't need high precision)
typedef float charge_t;


// Some utility functions

/**
 *  Check if a container's first few elements are exactly
 *  the same as another container
 */
template<class Container>
bool begins_with(const Container& input, const Container& match)
{
    return input.size() >= match.size()
        && std::equal(match.begin(), match.end(), input.begin());
}


// TODO: Replace the following with Boost's string algo?
/**
 * Trim string from right and left
 * If the string contains any characters that are in `t`
 * at the beginning or at the end, they are removed.
 * (default `t`: if it contains any of " ", "\t", "\n", "\r", "\f", "\v")
 */
inline void trim(std::string& s, const char* t = " \t\n\r\f\v")
{
  s.erase(0, s.find_first_not_of(t));
  s.erase(s.find_last_not_of(t) + 1);
}


#endif /* __dealii_qc_utility_h */
