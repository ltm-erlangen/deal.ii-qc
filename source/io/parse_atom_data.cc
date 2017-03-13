
#include <dealiiqc/io/parse_atom_data.h>

namespace dealiiqc
{
  using namespace dealii;

  template<int dim>
  ParseAtomData<dim>::ParseAtomData()
    :
    n_atoms(0),n_atom_types(0), line_no(0)
  {}

  template<int dim>
  std::vector<Atom<dim>> ParseAtomData<dim>::parse( std::istream &is )
  {
    std::string line;
    std::vector<Atom<dim>> atoms;

    // Some temporary variables
    unsigned int nbonds = 0;
    unsigned int nangles = 0;
    unsigned int ndihedrals = 0;
    unsigned int nimpropers = 0;

    // Dimensions of the simulation box
    double xlo(0.),xhi(0.),
           ylo(0.),yhi(0.),
           zlo(0.),zhi(0.);

    // Tilts of the simulation box
    double xy(0.),xz(0.),yz(0.);

    // Read comment line (first line)
    line_no++;
    if (!std::getline(is,line))
      Assert( false, ExcReadFailed(line_no));

    // Read main header declarations
    while ( true )
      {
        // Skip empty lines
        if ( !skip_read(is,line) )
          Assert( false, ExcReadFailed(line_no));

        // Read and store n_atoms, nbonds, ...
        if (line.find("atoms") != std::string::npos)
          {
            unsigned long long int n_atoms_tmp;
            if (sscanf(line.c_str(), "%llu", &n_atoms_tmp) != 1)
              Assert( false, ExcInvalidValue(line_no,"atoms"));
            if ( n_atoms_tmp <= UINT_MAX )
              n_atoms = static_cast<typedefs::global_atom_index>(n_atoms_tmp);
            else
              Assert( false, ExcMessage("The number of atoms specified "
                                        "is more than what `typedefs::global_atom_index` can work with "
                                        "try building deal.II with 64bit index space"))
            }
        else if (line.find("bonds") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nbonds) != 1)
              Assert( false, ExcInvalidValue(line_no,"bonds"));
            Assert( nbonds==0, ExcIrrelevant(line_no));
          }
        else if (line.find("angles") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nangles) != 1)
              Assert( false, ExcInvalidValue(line_no,"angles"));
            Assert( nangles==0, ExcIrrelevant(line_no));
          }
        else if (line.find("dihedrals") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &ndihedrals) != 1)
              Assert( false, ExcInvalidValue(line_no,"dihedrals"));
            Assert( ndihedrals==0, ExcIrrelevant(line_no));
          }
        else if (line.find("impropers") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nimpropers) != 1)
              Assert( false, ExcInvalidValue(line_no,"impropers"));
            Assert( nimpropers==0, ExcIrrelevant(line_no));
          }
        else if (line.find("atom types") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &n_atom_types) != 1)
              Assert( false , ExcInvalidValue(line_no,"number of atom types"));
          }
        else if (line.find("xlo xhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &xlo, &xhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("ylo yhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &ylo, &yhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("zlo zhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &zlo, &zhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("xy xz yz") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf %lf", &xy, &xz, &yz) != 3)
              Assert( false, ExcInvalidValue(line_no,"simulation box tilts"));
            Assert((((xy==xz)==yz)==0), ExcIrrelevant(line_no));
          }
        else break;
      }

    Assert( xhi >= xlo || yhi >= ylo || zhi >= zlo, ExcMessage("Invalid simulation box size"))
    // TODO: Extra Asserts with dim and simulation box dimensions
    // TODO: copy simulation box dimensions, check for inconsistent mesh?

    // Read sections with keyword headings (Atoms, Masses)
    // For now ignoring other keyword headings // TODO?
    while ( true )
      {
        if ( dealii::Utilities::match_at_string_start(line, "Masses") )
          {
            // TODO: Use masses of different types of atom for FIRE minimization scheme?
            parse_masses(is, line);
            if ( is.eof() )
              break;
            else
              continue;
          }
        else if ( dealii::Utilities::match_at_string_start(line,"Atoms"))
          {
            atoms = parse_atoms(is, line);
            if ( is.eof() )
              break;
            else
              continue;
          }
        else
          break;
      }
    Assert(atoms.size()!=0, ExcMessage("Given atom data doesn't contain atom positions keyword"));
    return atoms;
  }

  template< int dim>
  std::vector<Atom<dim>> ParseAtomData<dim>::parse_atoms( std::istream &is, std::string &line )
  {
    // TODO: Don't know if size_t can handle large unsigned integer values
    std::size_t  i_atom_index;
    unsigned int i_atom_type;
    double i_q, i_x, i_y, i_z;
    // TODO: Use atom types to initialize neighbor lists faster
    std::map<unsigned int,typedefs::global_atom_index> atom_types;

    std::vector<Atom<dim>> atoms(n_atoms);

    // Store atom attributes and positions
    typedefs::global_atom_index i=1;
    while ( skip_read(is,line) && i <= n_atoms)
      {
        // Not reading molecular_id
        if (sscanf(line.c_str(), "%zu %*u %u %lf %lf %lf %lf",
                   &i_atom_index, &i_atom_type, &i_q, &i_x, &i_y, &i_z ) !=6)
          Assert( false, ExcInvalidValue(line_no, "atom attributes under Atom keyword section"));
        atom_types[i_atom_type]   = static_cast<typedefs::global_atom_index> (i_atom_index);
        atoms[i_atom_index-1].q    = static_cast<typedefs::charge_t>(i_q);
        // TODO: 1 and 2 dim cases could be potentially incorrect
        // Possible way to correct this is to ask user axes dimensionality
        if (dim==1)
          atoms[i_atom_index-1].position[0] = i_x;
        else if (dim==2)
          {
            atoms[i_atom_index-1].position[0] = i_x;
            atoms[i_atom_index-1].position[1] = i_y;
          }
        else if (dim==3)
          {
            atoms[i_atom_index-1].position[0] = i_x;
            atoms[i_atom_index-1].position[1] = i_y;
            atoms[i_atom_index-1].position[2] = i_z;
          }

        i++;
      }
    Assert(i-1==n_atoms, ExcInternalError());

    return atoms;
  }

  template< int dim>
  std::vector<double> ParseAtomData<dim>::parse_masses( std::istream &is, std::string &line)
  {
    size_t i_atom_type;
    double i_mass;
    std::vector<double> masses(n_atom_types, 0.);

    unsigned int i=1;
    // Store atom masses for different atom types
    while ( skip_read(is,line) && i<=n_atom_types)
      {
        if (sscanf(line.c_str(), "%zu %lf", &i_atom_type, &i_mass) !=2)
          Assert( false, ExcInvalidValue(line_no,"Mass"));

        // Copy masses for all atom types
        masses[i_atom_type] = i_mass;
        ++i;
      }

    return masses;
  }

  template<int dim>
  bool ParseAtomData<dim>::skip_read(std::istream &is, std::string &line)
  {
    bool eternal = true;
    while ( true )
      {
        // Check end of stream, if it is set end to false and exit loop
        if (is.eof())
          {
            eternal = false;
            break;
          }

        ++line_no;
        // Read new line, if cannot read check if its end of stream
        // if it's not end of stream throw exception
        if (!std::getline(is,line))
          {
            if (is.eof())
              {
                eternal = false;
                break;
              }
            else
              Assert( false, ExcReadFailed(line_no));
          }

        // Ignore empty lines
        if ( line.find_first_not_of(" \t\n\r") == std::string::npos)
          continue;
        else
          break;
      }

    // Find if # is present
    size_t trim_pos = line.find("#");

    // Trim all the stuff after #
    if ( trim_pos != std::string::npos )
      line.resize( trim_pos );

    // Trim line from left and right with " \t\n\r"
    line = Utilities::trim(line);

    return eternal;
  }

  template      ParseAtomData<1>::ParseAtomData();
  template      ParseAtomData<2>::ParseAtomData();
  template      ParseAtomData<3>::ParseAtomData();
  template bool ParseAtomData<1>::skip_read(std::istream &, std::string &);
  template bool ParseAtomData<2>::skip_read(std::istream &, std::string &);
  template bool ParseAtomData<3>::skip_read(std::istream &, std::string &);
  template std::vector<double>  ParseAtomData<1>::parse_masses(std::istream &, std::string &);
  template std::vector<double>  ParseAtomData<2>::parse_masses(std::istream &, std::string &);
  template std::vector<double>  ParseAtomData<3>::parse_masses(std::istream &, std::string &);
  template std::vector<Atom<1>> ParseAtomData<1>::parse_atoms(std::istream &, std::string &);
  template std::vector<Atom<2>> ParseAtomData<2>::parse_atoms(std::istream &, std::string &);
  template std::vector<Atom<3>> ParseAtomData<3>::parse_atoms(std::istream &, std::string &);
  template std::vector<Atom<1>> ParseAtomData<1>::parse(std::istream &);
  template std::vector<Atom<2>> ParseAtomData<2>::parse(std::istream &);
  template std::vector<Atom<3>> ParseAtomData<3>::parse(std::istream &);


} /* namespace dealiiqc */
