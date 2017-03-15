
#ifndef __dealii_qc_parse_atom_data_h
#define __dealii_qc_parse_atom_data_h

#include <istream>
#include <vector>
#include <climits>

#include <deal.II/base/point.h>

#include <dealiiqc/atom/atom.h>
#include <dealiiqc/utility.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class to parse atom data stream.
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
                    << "Could not parse " << arg2 << " or "
                    << "invalid " << arg2 << " read "
                    << "at line number: " << arg1 );

    /**
     * Parse @p is input stream and initialize all the atom attributes.
     * The input stream is allowed to have multiple Masses and Atoms
     * keyword section. In such cases the old atom attributes will be overwritten.
     * @param[in] is input stream
     * @param[out] atoms container to store atom attributes
     * @param[out] masses container to store masses of different atom types
     * @param[out] atomtype_to_atoms atom type and atoms association
     */
    void parse( std::istream &is,
                std::vector<Atom<dim>> &atoms,
                std::vector<double> &masses,
                std::multimap<unsigned int,types::global_atom_index> &atomtype_to_atoms);

  private:

    /**
     * Remove from @p input string all comments (content after #) and
     * all standard whitespace characters (including * '<tt>\\t</tt>',
     * '<tt>\\n</tt>', and '<tt>\\r</tt>') at the beginning, and at the end
     * and return the resulting string.
     */
    std::string strip( const std::string &input );

    /**
     * Parse atoms data from @p is input stream under Atoms keyword section.
     * The input stream should be at the line after the keyword Masses.
     * @param[in] is input stream
     * @param[out] atoms container to store atom attributes
     * @param[out] atomtype_to_atoms atom type and atoms association
     */
    void parse_atoms( std::istream &is,
                      std::vector<Atom<dim>> &,
                      std::multimap<unsigned int, types::global_atom_index> &atomtype_to_atoms);

    /**
     * Parse @p is input stream for mass entries under Masses keyword section
     * to obtain @p masses, a vector of masses of different atom types  and write the result into @p masses.
     * The input stream should be at the line after the keyword Masses
     */
    void parse_masses( std::istream &is, std::vector<double> &masses);

    /**
     * Number of atoms read from the input stream.
     */
    types::global_atom_index n_atoms;

    /**
     * Number of atom types read from the input stream.
     */
    size_t n_atom_types;

    /**
     * Line number of the input stream as it is read.
     * Used to inform the user at which line number reading failed.
     */
    unsigned int line_no;

  };

} /* namespace dealiiqc */

#endif /* __dealii_qc_parse_atom_data_h */
