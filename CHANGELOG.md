# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `PotentialBornCutManager` to compute truncated Born-Mayer-Huggins or
Tosi-Fumi pair potential.
- `GridTools::Cache` object to cache computationally intensive information
about Triangulation.
Information in `GridTools::Cache` is used while associating atoms with cells,
more specifically in finding active cells around atoms.
- Automatic installation of git-hooks into `./git/hooks` folder using `cmake`.

### Changed
- Code formatting tool from astyle to `clang-format`.