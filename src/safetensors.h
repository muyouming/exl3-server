#pragma once
// Minimal read-only safetensors reader. The whole file is mmap'd; tensor data
// is addressed in place (no copies) and the mapping is dropped when done.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace exl3 {

enum class DT { F16, BF16, I16, U16, I32, F32, UNKNOWN };

const char* dtype_name(DT d);

struct STTensor {
  DT dtype;
  std::vector<int64_t> shape;
  int64_t offset; // byte offset of the data within the file
  int64_t nbytes;
};

struct SafetensorsFile {
  std::string path;
  int64_t header_len = 0; // 8 + header size, i.e. the data section base
  std::map<std::string, STTensor> tensors;

  const uint8_t* map_base = nullptr;
  int64_t map_size = 0;

  SafetensorsFile() = default;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&& other) noexcept { *this = std::move(other); }
  SafetensorsFile& operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
      unmap();
      path = std::move(other.path);
      header_len = other.header_len;
      tensors = std::move(other.tensors);
      map_base = other.map_base;
      map_size = other.map_size;
      other.map_base = nullptr; // ownership transferred
      other.map_size = 0;
    }
    return *this;
  }
  ~SafetensorsFile() { unmap(); }

  // Parse the header and mmap the file.
  static bool load(const std::string& path, SafetensorsFile& out, std::string& err);
  void unmap();

  bool has(const std::string& name) const { return tensors.count(name) != 0; }
  const STTensor& tensor(const std::string& name) const;

  // Pointer to the tensor's data inside the mapping. Throws via err on misuse.
  const uint8_t* data(const std::string& name, std::string& err) const;
  // Typed accessors (validate dtype).
  const uint16_t* f16_data(const std::string& name, std::string& err) const;
  const uint16_t* bf16_data(const std::string& name, std::string& err) const;
  const int16_t* i16_data(const std::string& name, std::string& err) const;
  const int32_t* i32_data(const std::string& name, std::string& err) const;

  int64_t numel(const std::string& name) const;
};

} // namespace exl3
