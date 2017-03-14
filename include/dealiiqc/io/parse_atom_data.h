
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
     * Parse input stream and initialize all the atom attributes.
     * @param is input stream
     * @param atoms container to store atom attributes
     * @param masses container to store masses of different atom types
     * @param atom_types atom and atom type association
     */
    void parse( std::istream &, std::vector<Atom<dim>> &,
                std::vector<double> &,
                std::map<unsigned int,types::global_atom_index> &);

  private:

    /**
     * Return a string with all comments (content after #) and
     * all standard whitespace characters
     * (including * '<tt>\\t</tt>', '<tt>\\n</tt>', and '<tt>\\r</tt>') at
     * the beginning and end of @p input removed.
     */
    std::string strip( const std::string &input );

    /**
     * Parse atoms data.
     * @return a vector of Atom class objects
     * We let the input stream contain multiple `Atoms` keyword
     * sections. The old atom attributes will be overwritten.
     */
    void parse_atoms( std::istream &, std::vector<Atom<dim>> &,
                      std::map<unsigned int, types::global_atom_index> &);

    /**
     * Return @param masses, a vector of masses of different atom types
     * read upon parsing mass entries under Masses keyword section of
     * the @param is.
     * We let the input stream contain multiple `Masses` keyword
     * sections.
     */
    void parse_masses( std::istream &is, std::vector<double> &);

    /**
     * Number of atoms read from the input stream.
     */
    types::global_atom_index n_atoms;

    /**
     * Number of atom types read from the input stream.
     */
    unsigned int n_atom_types;

    /**
     * Line number of the input stream as it is read.
     * Used to inform the user at which line number reading failed.
     */
    unsigned int line_no;

  };

} /* namespace dealiiqc */

#endif /* __dealii_qc_parse_atom_data_h */
