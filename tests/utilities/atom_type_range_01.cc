
#include <deal.II-qc/utilities.h>

using namespace dealii;
using namespace dealiiqc;


// Short test to check correctness of
// dealiiqc::Utilities::atom_type_range()


void
test()
{
  const auto first_pair  = dealiiqc::Utilities::atom_type_range("1*", 2);
  const auto second_pair = dealiiqc::Utilities::atom_type_range("*", 4);
  const auto third_pair  = dealiiqc::Utilities::atom_type_range("*2", 5);
  const auto fourth_pair = dealiiqc::Utilities::atom_type_range("1*3", 7);
  const auto sixth_pair  = dealiiqc::Utilities::atom_type_range("5", 6);

  std::cout << (int)first_pair.first << ":" << (int)first_pair.second
            << std::endl
            << (int)second_pair.first << ":" << (int)second_pair.second
            << std::endl
            << (int)third_pair.first << ":" << (int)third_pair.second
            << std::endl
            << (int)fourth_pair.first << ":" << (int)fourth_pair.second
            << std::endl
            << (int)sixth_pair.first << ":" << (int)sixth_pair.second
            << std::endl;
}

int
main()
{
  try
    {
      test();
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" << std::endl;
    }
}
