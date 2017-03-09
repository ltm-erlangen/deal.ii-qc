
#ifndef __dealii_qc_parse_atom_data_h
#define __dealii_qc_parse_atom_data_h

#include <istream>
#include <vector>

#include <deal.II/base/point.h>

#include <dealiiqc/atom/atom.h>
#include <dealiiqc/utility.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class to parse atom data file.
   * Stores atom attributes
   */
  template< int dim>
  class ParseAtomData
  {
  public:

    /**
     * Constructor takes data
     */
    ParseAtomData();

    DeclException1( ExcIrrelevant,
		    unsigned int,
		    << "Input atom data stream contains atom attributes "
		    << "at line number: " << arg1 << " "
		    << "which are either not supported within QC formulation "
		    << "or not yet implemented");

    DeclException1( ExcReadFailed,
		   unsigned int,
		   << "Could not read atom data stream "
		   << "at line number: " << arg1 << " "
		   << "Either end of stream reached unexpectedly or "
		   << "important atom attributes are not mentioned!");

    DeclException2( ExcInvalidValue,
		   unsigned int,
    		   std::string,
		   << "Could not read " << arg2 << " or"
    		   << "invalid  " << arg2 << " read "
		   << "at line number: " << arg1 );

    /**
     * Parse input stream
     */
    void parse( std::istream &,  std::vector<Atom<dim>>& );

  private:

    /**
     * Skip empty lines and return the latest non-empty line
     * return true for end of stream
     */
    inline bool skip_read( std::istream &, std::string & );

    /**
     * Parse atoms
     * return true if end of stream is reached
     */
    bool parse_atoms( std::istream &, std::vector<Atom<dim>>& );

    /**
     * Parse masses of different atom types
     * return true if end of stream is reached
     */
    bool parse_masses( std::istream &);

    /**
     * Number of atoms read from the LAMMPS atom data file
     */
    unsigned int n_atoms;

    /**
     * Number of atom types
     */
    unsigned int n_atom_types;

    /**
     * Stream line number
     * Used to inform the user at which line number reading failed
     */
    unsigned int line_no;

  };

} /* namespace dealiiqc */

#endif /* __dealii_qc_parse_atom_data_h */
