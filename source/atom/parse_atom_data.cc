
#include <deal.II-qc/atom/parse_atom_data.h>

namespace dealiiqc
{
  using namespace dealii;

  template<int dim>
  ParseAtomData<dim>::ParseAtomData()
    :
    n_atoms(0),
    n_atom_types(0),
    line_no(0)
  {}

  template<int dim>
  void ParseAtomData<dim>::parse( std::istream &is,
                                  std::vector<Atom<dim>> &atoms,
                                  std::vector<types::charge> &charges,
                                  std::vector<double> &masses)
  {
    AssertThrow (is, ExcIO());

    std::string line;

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
    while ( std::getline(is,line) )
      {
        line_no++;

        // strip all the comments and all white characters
        line = strip(line);

        // Skip empty lines
        if ( line.size()==0)
          continue;

        // Read and store n_atoms, nbonds, ...
        if (line.find("atoms") != std::string::npos)
          {
            unsigned long long int n_atoms_tmp;
            if (sscanf(line.c_str(), "%llu", &n_atoms_tmp) != 1)
              AssertThrow( false, ExcInvalidValue(line_no,"atoms"));
            if ( n_atoms_tmp <= UINT_MAX )
              n_atoms = static_cast<types::global_atom_index>(n_atoms_tmp);
            else
              AssertThrow( false,
                           ExcMessage("The number of atoms specified "
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
            if (sscanf(line.c_str(), "%zu", &n_atom_types) != 1)
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

    Assert( xhi >= xlo || yhi >= ylo || zhi >= zlo,
            ExcMessage("Invalid simulation box size"))
    // TODO: Extra Asserts with dim and simulation box dimensions
    // TODO: copy simulation box dimensions, check for inconsistent mesh?

    // line has been read by previous while loop but wasn't parsed for keyword sections
    --line_no;
    do
      {
        // Read sections with keyword headings (Atoms, Masses)
        // For now ignoring other keyword headings // TODO?
        if ( dealii::Utilities::match_at_string_start(line, "Masses") )
          {
            // parse_masses is allowed to be executed multiple times.
            parse_masses(is, masses);
          }
        else if ( dealii::Utilities::match_at_string_start(line,"Atoms"))
          {
            // parse_atoms is allowed to be executed multiple times.
            parse_atoms(is, atoms, charges);
          }
        ++line_no;
      }
    while ( std::getline(is,line) );

    return;
  }

  template< int dim>
  void ParseAtomData<dim>::parse_atoms( std::istream &is,
                                        std::vector<Atom<dim>> &atoms,
                                        std::vector<types::charge> &charges)
  {
    std::string line;

    atoms.resize(n_atoms);
    charges.resize(n_atom_types);

    // i_atom = i_atom_index-1
    // atom data counts atoms starting from 1 and
    // the vector container with 0
    unsigned long long int  i_atom_index;
    types::global_atom_index i_atom;

    unsigned int molecule_id;

    unsigned char i_atom_type;
    types::atom_type tmp_type;

    // temporary variable to count number of atom types
    std::vector<types::atom_type> unique_types;

    double i_q, i_x, i_y, i_z;

    // Store atom attributes and positions
    types::global_atom_index i=0;
    while ( std::getline(is,line) && i < n_atoms)
      {
        ++line_no;
        line = strip(line);
        if ( line.size()==0)
          continue;

        // Not reading molecular_id
        if (sscanf(line.c_str(), "%llu  %u " UC_SCANF_STR " %lf %lf %lf %lf",
                   &i_atom_index, &molecule_id, &i_atom_type, &i_q, &i_x, &i_y, &i_z ) !=7)
          AssertThrow( false,
                       ExcInvalidValue( line_no,
                                        "atom attributes under Atom keyword section"));
        Assert( i_atom_index<=n_atoms && i_atom_index>0,
                ExcInvalidValue( line_no,
                                 "atom index (> number of atoms or <=0"));

        i_atom = static_cast<types::global_atom_index>(i_atom_index) -1;

        atoms[i_atom].global_index = i_atom;

        tmp_type = static_cast<types::atom_type> (i_atom_type) -1;

        Assert(tmp_type>=0 && tmp_type < 256,
               ExcInvalidValue(line_no, "atom type attribute"));

        atoms[i_atom].type = tmp_type;

        if (std::find(unique_types.begin(), unique_types.end(), tmp_type) == unique_types.end())
          {
            unique_types.push_back( tmp_type);
            charges[tmp_type] = static_cast<types::charge>(i_q);
          }
        else
          Assert( charges[tmp_type]==static_cast<types::charge>(i_q),
                  ExcInvalidValue(line_no,"charge attribute"));

        // TODO: 1 and 2 dim cases could be potentially incorrect
        // Possible way to correct this is to ask user axes dimensionality
        if (dim==1)
          atoms[i_atom].position[0] = i_x;
        else if (dim==2)
          {
            atoms[i_atom].position[0] = i_x;
            atoms[i_atom].position[1] = i_y;
          }
        else if (dim==3)
          {
            atoms[i_atom].position[0] = i_x;
            atoms[i_atom].position[1] = i_y;
            atoms[i_atom].position[2] = i_z;
          }
        ++i;
      }
    Assert( unique_types.size()==n_atom_types,
            ExcInternalError());
    Assert( i==atoms.size(),
            ExcMessage("The number of atoms "
                       "do not match the number of entries "
                       "under Atoms keyword section"));
    return;
  }

  template< int dim>
  void ParseAtomData<dim>::parse_masses( std::istream &is,
                                         std::vector<double> &masses )
  {
    masses.resize(n_atom_types, 0.);
    std::string line;
    size_t i_atom_type;
    double i_mass;

    size_t i=0;
    // Store atom masses for different atom types
    while ( std::getline(is,line) && i<n_atom_types)
      {
        ++line_no;
        line = strip(line);
        if ( line.size()==0)
          continue;

        if (sscanf(line.c_str(), "%zu %lf", &i_atom_type, &i_mass) !=2)
          AssertThrow( false, ExcInvalidValue(line_no,"Mass"));

        Assert( i_atom_type <= n_atom_types && i_atom_type>0,
                ExcInvalidValue( line_no, "atom type index (> number of atom types "
                                 "or <= 0)"));

        // Copy masses for all atom types
        masses[i_atom_type-1] = i_mass;
        ++i;
      }
    line_no++;
    Assert( i==masses.size(),
            ExcMessage("The number of different atom types "
                       "do not match the number of entries "
                       "under Masses keyword section") );
    return;
  }

  template< int dim>
  std::string ParseAtomData<dim>::strip( const std::string &input )
  {
    std::string line(input);

    // Find if # is present
    size_t trim_pos = line.find("#");

    // Trim all the stuff after #
    if ( trim_pos != std::string::npos )
      line.resize( trim_pos );

    // Trim line from left and right with " \t\n\r"
    line = dealii::Utilities::trim(line);

    return line;
  }

  template class ParseAtomData<1>;
  template class ParseAtomData<2>;
  template class ParseAtomData<3>;

} /* namespace dealiiqc */
