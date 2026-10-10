#!/usr/bin/env python3
"""Ground truth for tests/chat_template_test.cpp.

Renders the checkpoint's chat_template.jinja with jinja2 (the `break` loops
need the loopcontrols extension, which transformers enables too) so the C++
renderer can be compared byte-for-byte against the real template:

    python3 -m pip install --user jinja2
    python3 tools/render_chat_reference.py <model_dir_with_chat_template.jinja>
"""
import sys
import jinja2

T = open(sys.argv[1] + "/chat_template.jinja" if len(sys.argv) > 1 else
         "/Volumes/External/Models/MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3/chat_template.jinja").read()
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
cases = [
    ("plain", [{"role": "user", "content": "Hello"},
               {"role": "assistant", "content": "Hi there"}], {}),
    ("low_no_gen", [{"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}],
     {"reasoning_effort": "low", "add_generation_prompt": False}),
    ("reasoning", [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "<think>\nstep 1\n</think>\nanswer"}], {}),
    ("tools_msgs", [{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "call"},
                    {"role": "tool", "content": '{"a":1}'},
                    {"role": "tool", "content": '{"b":2}'}], {}),
    ("clear", [{"role": "user", "content": "a"},
               {"role": "assistant", "content": "<think>r1</think>x"},
               {"role": "user", "content": "b"},
               {"role": "assistant", "content": "<think>r2</think>y"}],
     {"clear_thinking": True}),
]
for name, msgs, kw in cases:
    kw = dict(kw)
    kw.setdefault("add_generation_prompt", True)
    print(f"{name}: {env.from_string(T).render(messages=msgs, **kw)!r}")
