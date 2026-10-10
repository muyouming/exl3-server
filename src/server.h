#pragma once
// OpenAI-compatible HTTP server: /v1/models, /v1/chat/completions,
// /v1/completions (streaming SSE and non-streaming), /health.

#include <atomic>
#include <string>

#include "engine.h"

namespace exl3 {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  bool verbose = false;
};

// Blocks until the server stops. Returns a process exit code.
int run_server(Engine& engine, const ServerOptions& opt, std::atomic<bool>& stop);

} // namespace exl3
