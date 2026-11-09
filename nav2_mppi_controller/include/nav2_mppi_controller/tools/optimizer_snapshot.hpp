// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0.
#ifndef NAV2_MPPI_CONTROLLER__TOOLS__OPTIMIZER_SNAPSHOT_HPP_
#define NAV2_MPPI_CONTROLLER__TOOLS__OPTIMIZER_SNAPSHOT_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mppi
{
// Opt-in computational snapshot, not a real-time telemetry transport. Records
// are an ASCII name/type/count header, little-endian payload, and newline.
// Buffer until the command has been computed; never modify optimizer arrays.
class OptimizerSnapshot
{
public:
  OptimizerSnapshot()
  {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1) {
      throw std::runtime_error("Optimizer snapshots require a little-endian host");
    }
    stream_ << "MPPI_SNAPSHOT_LE_V1\n";
  }

  void prefix(const std::string & value) {prefix_ = value;}

  template<typename T>
  void values(const std::string & name, const T & values)
  {
    using Value = typename T::value_type;
    static_assert(std::is_same_v<Value, float>|| std::is_same_v<Value, double>||
      std::is_same_v<Value, uint8_t>);
    const char * type = std::is_same_v<Value, float> ? "f32" :
      (std::is_same_v<Value, double> ? "f64" : "u8");
    stream_ << prefix_ << name << ' ' << type << ' ' << values.size() << '\n';
    stream_.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(Value));
    stream_ << '\n';
  }

  template<size_t N>
  void scalars(const std::string & name, const std::array<double, N> & values)
  {
    this->values(name, values);
  }

  bool save(const std::string & filename) const
  {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path(), error);
    if (error) {return false;}
    const std::string bytes = stream_.str();
    std::ofstream file(filename + ".partial", std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), bytes.size());
    file.close();
    if (!file) {return false;}
    std::filesystem::rename(filename + ".partial", filename, error);
    return !error;
  }

private:
  std::ostringstream stream_;
  std::string prefix_;
};
}  // namespace mppi
#endif  // NAV2_MPPI_CONTROLLER__TOOLS__OPTIMIZER_SNAPSHOT_HPP_
