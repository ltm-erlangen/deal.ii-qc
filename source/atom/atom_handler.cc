#include <dealiiqc/atom/atom_handler.h>

namespace dealiiqc
{

  template<int dim>
  AtomHandler<dim>::AtomHandler( const ConfigureQC & configure_qc)
  {
    if (!(configure_qc.get_atom_data_file()).empty() )
      {
        const std::string atom_data_file = configure_qc.get_atom_data_file();
        std::fstream fin(atom_data_file, std::fstream::in );
        // TODO: Use atom types to initialize neighbor lists faster
        // TODO: Use masses of different types of atom for FIRE minimization scheme?
        ParseAtomData<dim> atom_parser;
        atom_parser.parse( fin, atoms, charges, masses);
      }
  }

  template<int dim>
  void AtomHandler<dim>::setup( std::istream & is)
  {
    ParseAtomData<dim> atom_parser;
    atom_parser.parse( is, atoms, charges, masses);
    // TODO: Use atom types to initialize neighbor lists faster
    // TODO: Use masses of different types of atom for FIRE minimization scheme?
  }

  template class AtomHandler<1>;
  template class AtomHandler<2>;
  template class AtomHandler<3>;
}
