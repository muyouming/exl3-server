#pragma once
// Command line modes: one-shot generation and an interactive chat REPL.

#include <atomic>
#include <string>
#include <vector>

#include "engine.h"

namespace exl3 {

struct CliOptions {
  std::vector<std::string> prompts; // -p/--prompt, chat-formatted unless raw
  bool interactive = false;
  bool raw = false; // treat prompts as raw text completions
  std::string system;
  int thinking = 0;
  bool show_reasoning = false;
  bool stats = true;
  GenerateOptions gen;
};

int run_cli(Engine& engine, const CliOptions& opt, std::atomic<bool>& interrupted);

} // namespace exl3
