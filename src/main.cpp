#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "cli.h"
#include "engine.h"
#include "server.h"
#include "util.h"

namespace {

std::atomic<bool> g_interrupt{false};

void on_sigint(int) { g_interrupt.store(true); }

void usage() {
  std::fputs(
      "exl3-server - EXL3 (exllamav3) inference engine with an OpenAI-compatible API\n"
      "\n"
      "usage: exl3-server --model DIR [mode] [options]\n"
      "\n"
      "modes:\n"
      "  -p, --prompt TEXT     generate from a prompt (chat-formatted unless --raw)\n"
      "  -i, --interactive     interactive chat REPL\n"
      "      --serve           run the OpenAI-compatible HTTP server\n"
      "\n"
      "server:\n"
      "      --host HOST       bind address (default 127.0.0.1)\n"
      "      --port N          port (default 8080)\n"
      "\n"
      "sampling:\n"
      "      --max-tokens N           maximum tokens to generate (default 512)\n"
      "      --temp F                 temperature, 0 = greedy (default 0)\n"
      "      --top-p F                nucleus sampling (default 1.0)\n"
      "      --top-k N                top-k sampling (default 0 = off)\n"
      "      --min-p F                min-p sampling (default 0)\n"
      "      --repetition-penalty F   (default 1.0)\n"
      "      --presence-penalty F     (default 0)\n"
      "      --frequency-penalty F    (default 0)\n"
      "      --seed N                 RNG seed (default random)\n"
      "      --stop TEXT              stop string, repeatable\n"
      "      --ignore-eos             do not stop on EOS\n"
      "\n"
      "chat:\n"
      "      --system TEXT     system prompt\n"
      "      --think           start the reply inside a <think> block\n"
      "      --no-think        render an empty think block (default)\n"
      "      --show-reasoning  print reasoning text dimmed\n"
      "      --raw             treat --prompt as a raw text completion\n"
      "\n"
      "other:\n"
      "      --threads N       threads for weight reconstruction (default: all cores)\n"
      "      --model-name NAME model id reported by the API\n"
      "      --quiet           no per-request stats\n"
      "  -h, --help            this help\n",
      stdout);
}

bool need_value(int i, int argc) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "exl3-server: option %s requires a value\n", "");
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_sigint);
  std::signal(SIGPIPE, SIG_IGN);

  std::string model, host = "127.0.0.1", model_name, system;
  int port = 8080;
  int threads = 0;
  bool serve = false;
  exl3::CliOptions cli;
  bool verbose = true;

  auto arg = [&](int& i, const char* name) -> std::string {
    std::string a = argv[i];
    std::string prefix = std::string(name) + "=";
    if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    if (i + 1 < argc) return argv[++i];
    std::fprintf(stderr, "exl3-server: %s requires a value\n", name);
    std::exit(2);
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (a == "--model" || a.rfind("--model=", 0) == 0) {
      model = arg(i, "--model");
    } else if (a == "-p" || a == "--prompt" || a.rfind("--prompt=", 0) == 0) {
      cli.prompts.push_back(arg(i, a.rfind("--prompt=", 0) == 0 ? "--prompt" : a.c_str()));
    } else if (a == "-i" || a == "--interactive" || a == "--chat") {
      cli.interactive = true;
    } else if (a == "--serve") {
      serve = true;
    } else if (a == "--host" || a.rfind("--host=", 0) == 0) {
      host = arg(i, "--host");
    } else if (a == "--port" || a.rfind("--port=", 0) == 0) {
      port = std::atoi(arg(i, "--port").c_str());
    } else if (a == "--threads" || a.rfind("--threads=", 0) == 0) {
      threads = std::atoi(arg(i, "--threads").c_str());
    } else if (a == "--model-name" || a.rfind("--model-name=", 0) == 0) {
      model_name = arg(i, "--model-name");
    } else if (a == "--system" || a.rfind("--system=", 0) == 0) {
      system = arg(i, "--system");
    } else if (a == "--think") {
      cli.thinking = 1;
    } else if (a == "--no-think") {
      cli.thinking = 0;
    } else if (a == "--show-reasoning") {
      cli.show_reasoning = true;
    } else if (a == "--raw") {
      cli.raw = true;
    } else if (a == "--max-tokens" || a.rfind("--max-tokens=", 0) == 0) {
      cli.gen.max_tokens = std::atoi(arg(i, "--max-tokens").c_str());
    } else if (a == "--temp" || a.rfind("--temp=", 0) == 0) {
      cli.gen.sample.temperature = (float)std::atof(arg(i, "--temp").c_str());
    } else if (a == "--top-p" || a.rfind("--top-p=", 0) == 0) {
      cli.gen.sample.top_p = (float)std::atof(arg(i, "--top-p").c_str());
    } else if (a == "--top-k" || a.rfind("--top-k=", 0) == 0) {
      cli.gen.sample.top_k = std::atoi(arg(i, "--top-k").c_str());
    } else if (a == "--min-p" || a.rfind("--min-p=", 0) == 0) {
      cli.gen.sample.min_p = (float)std::atof(arg(i, "--min-p").c_str());
    } else if (a == "--repetition-penalty" || a.rfind("--repetition-penalty=", 0) == 0) {
      cli.gen.sample.repetition_penalty = (float)std::atof(arg(i, "--repetition-penalty").c_str());
    } else if (a == "--presence-penalty" || a.rfind("--presence-penalty=", 0) == 0) {
      cli.gen.sample.presence_penalty = (float)std::atof(arg(i, "--presence-penalty").c_str());
    } else if (a == "--frequency-penalty" || a.rfind("--frequency-penalty=", 0) == 0) {
      cli.gen.sample.frequency_penalty = (float)std::atof(arg(i, "--frequency-penalty").c_str());
    } else if (a == "--seed" || a.rfind("--seed=", 0) == 0) {
      cli.gen.sample.seed = (uint64_t)std::strtoull(arg(i, "--seed").c_str(), nullptr, 10);
    } else if (a == "--stop" || a.rfind("--stop=", 0) == 0) {
      cli.gen.stop.push_back(arg(i, "--stop"));
    } else if (a == "--ignore-eos") {
      cli.gen.ignore_eos = true;
    } else if (a == "--quiet") {
      verbose = false;
    } else {
      std::fprintf(stderr, "exl3-server: unknown option %s (try --help)\n", a.c_str());
      return 2;
    }
  }

  if (model.empty()) {
    std::fprintf(stderr, "exl3-server: --model is required\n\n");
    usage();
    return 2;
  }

  exl3::Engine engine;
  std::string err;
  if (!engine.load(model, err, threads)) {
    std::fprintf(stderr, "exl3-server: %s\n", err.c_str());
    return 1;
  }
  if (!model_name.empty()) engine.set_model_name(model_name);
  cli.system = system;
  cli.stats = verbose;

  if (serve) {
    exl3::ServerOptions so;
    so.host = host;
    so.port = port;
    so.verbose = verbose;
    return exl3::run_server(engine, so, g_interrupt);
  }
  if (!cli.prompts.empty() || cli.interactive) {
    return exl3::run_cli(engine, cli, g_interrupt);
  }
  std::fprintf(stderr, "exl3-server: nothing to do; pass -p/--prompt, -i/--interactive or --serve\n");
  return 2;
}
