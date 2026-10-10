#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "nlohmann/json.hpp"
#include "util.h"

using nlohmann::json;

namespace exl3 {

const char* dtype_name(DT d) {
  switch (d) {
    case DT::F16: return "F16";
    case DT::BF16: return "BF16";
    case DT::I16: return "I16";
    case DT::U16: return "U16";
    case DT::I32: return "I32";
    case DT::F32: return "F32";
    default: return "?";
  }
}

static DT dtype_from(const std::string& s) {
  if (s == "F16" || s == "torch.float16") return DT::F16;
  if (s == "BF16" || s == "torch.bfloat16") return DT::BF16;
  if (s == "I16" || s == "torch.int16") return DT::I16;
  if (s == "U16") return DT::U16;
  if (s == "I32" || s == "torch.int32") return DT::I32;
  if (s == "F32" || s == "torch.float32") return DT::F32;
  return DT::UNKNOWN;
}

static int64_t dtype_size(DT d) {
  switch (d) {
    case DT::F16:
    case DT::BF16:
    case DT::I16:
    case DT::U16: return 2;
    case DT::I32:
    case DT::F32: return 4;
    default: return 0;
  }
}

bool SafetensorsFile::load(const std::string& path, SafetensorsFile& out, std::string& err) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { err = "cannot open " + path; return false; }
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); err = "cannot stat " + path; return false; }
  int64_t size = (int64_t)st.st_size;
  if (size < 8) { ::close(fd); err = "truncated file " + path; return false; }
  void* base = ::mmap(nullptr, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) { err = "mmap failed for " + path; return false; }

  uint64_t n = 0;
  std::memcpy(&n, base, 8);
  if ((int64_t)n + 8 > size) {
    ::munmap(base, (size_t)size);
    err = "invalid safetensors header length in " + path;
    return false;
  }
  json j;
  try {
    j = json::parse(std::string((const char*)base + 8, (size_t)n));
  } catch (const std::exception& e) {
    ::munmap(base, (size_t)size);
    err = std::string("bad safetensors header in ") + path + ": " + e.what();
    return false;
  }

  out.path = path;
  out.map_base = (const uint8_t*)base;
  out.map_size = size;
  out.header_len = 8 + (int64_t)n;
  for (auto& [name, t] : j.items()) {
    if (name == "__metadata__") continue;
    STTensor st_{};
    st_.dtype = dtype_from(t["dtype"].get<std::string>());
    if (st_.dtype == DT::UNKNOWN) {
      out.unmap();
      err = "unsupported dtype " + t["dtype"].get<std::string>() + " for " + name;
      return false;
    }
    for (auto& s : t["shape"]) st_.shape.push_back(s.get<int64_t>());
    st_.offset = t["data_offsets"][0].get<int64_t>();
    st_.nbytes = t["data_offsets"][1].get<int64_t>() - st_.offset;
    int64_t numel = 1;
    for (auto d : st_.shape) numel *= d;
    if (dtype_size(st_.dtype) * numel != st_.nbytes || out.header_len + st_.offset + st_.nbytes > size) {
      out.unmap();
      err = "inconsistent size for " + name;
      return false;
    }
    out.tensors[name] = std::move(st_);
  }
  return true;
}

void SafetensorsFile::unmap() {
  if (map_base) {
    ::munmap((void*)map_base, (size_t)map_size);
    map_base = nullptr;
    map_size = 0;
  }
}

const STTensor& SafetensorsFile::tensor(const std::string& name) const {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    std::fprintf(stderr, "exl3: missing tensor %s\n", name.c_str());
    std::abort();
  }
  return it->second;
}

const uint8_t* SafetensorsFile::data(const std::string& name, std::string& err) const {
  auto it = tensors.find(name);
  if (it == tensors.end()) { err = "missing tensor " + name; return nullptr; }
  return map_base + header_len + it->second.offset;
}

const uint16_t* SafetensorsFile::f16_data(const std::string& name, std::string& err) const {
  const uint8_t* p = data(name, err);
  if (!p) return nullptr;
  DT d = tensor(name).dtype;
  if (d != DT::F16) { err = "tensor " + name + " is " + dtype_name(d) + ", expected F16"; return nullptr; }
  return (const uint16_t*)p;
}

const uint16_t* SafetensorsFile::bf16_data(const std::string& name, std::string& err) const {
  const uint8_t* p = data(name, err);
  if (!p) return nullptr;
  if (tensor(name).dtype != DT::BF16) { err = "tensor " + name + " is not BF16"; return nullptr; }
  return (const uint16_t*)p;
}

const int16_t* SafetensorsFile::i16_data(const std::string& name, std::string& err) const {
  const uint8_t* p = data(name, err);
  if (!p) return nullptr;
  if (tensor(name).dtype != DT::I16) { err = "tensor " + name + " is not I16"; return nullptr; }
  return (const int16_t*)p;
}

const int32_t* SafetensorsFile::i32_data(const std::string& name, std::string& err) const {
  const uint8_t* p = data(name, err);
  if (!p) return nullptr;
  if (tensor(name).dtype != DT::I32) { err = "tensor " + name + " is not I32"; return nullptr; }
  return (const int32_t*)p;
}

int64_t SafetensorsFile::numel(const std::string& name) const {
  int64_t n = 1;
  for (auto d : tensor(name).shape) n *= d;
  return n;
}

} // namespace exl3
