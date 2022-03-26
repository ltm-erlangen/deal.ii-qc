
#include <memory>
#include <type_traits>
#include <unordered_map>

#include <deal.II-qc/atom/atom.h>

#include <benchmark/benchmark.h>

#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/stream.hpp>

#include <boost/container/flat_map.hpp>

using namespace dealiiqc;
using namespace boost::container;

/*
  Google's micro-benchmark library primer:

  - The micro-benchmark library measures the duration of the work loop:
    `for (auto _ : state) { ... }` or `while(state.KeepRunning()){ ... }`.

  - The timers can be controlled using `state.PauseTiming()` and
    `state.ResumeTiming()`

  - To prevent an expression from being optimized away by the compiler
    the expression can be enclosed inside benchmark::DoNotOptimize(...)
    or benchmark::ClobberMemory(...).

*/

void BM_traverse_multimap(benchmark::State &state)
{
  using size_type = typename std::multimap<int, Atom<3>>::size_type;

  std::multimap<int, Atom<3>> container;
  const size_type container_size = state.range(0);

  const std::vector<Atom<3>> atoms(container_size);

  for (int i = 0; i < (int)container_size; ++i)
    for (const auto &atom : atoms)
      container.insert(std::make_pair(i, atom));

  for(auto _ : state)
  {
    for (const auto & element : container)
      benchmark::DoNotOptimize(element.second);
  }

  state.SetComplexityN(state.range(0));
}

void BM_traverse_unordered_map(benchmark::State &state)
{
  using size_type = typename std::unordered_map<int, std::unique_ptr<std::vector<Atom<3>>>>::size_type;

  std::unordered_map<int, std::unique_ptr<std::vector<Atom<3>>>> container;
  size_type container_size = state.range(0);

  const std::vector<Atom<3>> atoms(container_size);

  for (int i = 0; i < (int) container_size; ++i)
      container.insert(
        std::make_pair(i,
                       std::make_unique<std::vector<Atom<3>>>(atoms))
        );

  for(auto _ : state)
  {
    for (const auto & element : container)
      for (const auto & atom : *element.second)
        benchmark::DoNotOptimize(atom);
  }

  state.SetComplexityN(state.range(0));
}

void BM_traverse_flat_map(benchmark::State &state)
{
  using size_type = typename flat_map<int, std::unique_ptr<std::vector<Atom<3>>>>::size_type;

  flat_map<int, std::unique_ptr<std::vector<Atom<3>>>> container;
  size_type container_size = state.range(0);

  const std::vector<Atom<3>> atoms(container_size);

  for (int i = 0; i < (int) container_size; ++i)
      container.insert(
        std::make_pair(i,
                       std::make_unique<std::vector<Atom<3>>>(atoms))
        );

  for(auto _ : state)
  {
    for (const auto & element : container)
      for (const auto & atom : *element.second)
        benchmark::DoNotOptimize(atom);
  }

  state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_traverse_multimap)->Range(1<<6, 1<< 10)
                               ->Complexity(benchmark::oN);
BENCHMARK(BM_traverse_unordered_map)->Range(1<<6, 1<< 10)
                               ->Complexity(benchmark::oN);
BENCHMARK(BM_traverse_flat_map)->Range(1<<6, 1<< 10)
                               ->Complexity(benchmark::oN);

BENCHMARK_MAIN();