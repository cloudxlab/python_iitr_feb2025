# cot_action_agent.py
import os
import re
import shlex
import tempfile
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# ========= System Prompt =========
DEFAULT_SYSTEM_PROMPT = r"""
You are an autonomous reasoning agent running in an iterative loop.
Your job is to think step-by-step and decide the next best Action to move the task forward.
You will be repeatedly invoked until you output the special action STOP.

Rules of the Loop
1) Reason first. Always show your thought process clearly under `Thought:` — break the problem into steps, plan next action, justify your decision.
2) Act next. After reasoning, produce exactly one `Action:` block. One of:
   - PYTHON  – Execute Python code.
   - SHELL   – Run a Linux command.
   - RESPOND – Reply to the user in plain text.
   - STOP    – Terminate the loop when done.

Strict Format
Thought:
- <step-by-step reasoning>
- <why this action>
- <expected result>

Action: <TYPE>
<content for that action>

Examples
Thought:
- I need to list files before choosing one.

Action: SHELL
ls -lh

Thought:
- I must compute a mean, I'll write short Python.

Action: PYTHON
numbers = [1,2,3,4,5]
print(sum(numbers)/len(numbers))

Thought:
- I now have enough info to answer the user.

Action: RESPOND
Here is your summary: ...

Thought:
- All objectives met.

Action: STOP
"""

# ========= Helpers =========

_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n\s*Action:\s*(?P<atype>STOP|PYTHON|SHELL|RESPOND)\s*\n?(?P<body>.*)\Z",
    re.DOTALL | re.IGNORECASE,
)

FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$", re.MULTILINE)

def strip_code_fences(s: str) -> str:
    return re.sub(FENCE_RE, "", s).strip()

def truncate(s: str, limit: int = 6000) -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[truncated {len(s)-limit} chars]"

@dataclass
class AgentConfig:
    model_name: str = "gpt-4o-mini"      # Change as needed
    temperature: float = 0.2
    max_steps: int = 24
    python_timeout_sec: int = 30
    shell_timeout_sec: int = 20
    working_dir: Optional[str] = None    # if None, a temp dir is used
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

LLMFn = Callable[[List[Dict[str, str]]], str]
RunnerFn = Callable[[str], Tuple[int, str, str, float]]  # (exit, stdout, stderr, elapsed_s)

# ========= Default Runners =========

def run_python_subprocess(code: str, timeout: int = 30, cwd: Optional[str] = None) -> Tuple[int, str, str, float]:
    """Runs Python code in a fresh subprocess to avoid main-process pollution."""
    code = strip_code_fences(code)
    start = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=cwd) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            ["python", "-u", path],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return proc.returncode, proc.stdout, proc.stderr, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        return 124, e.stdout or "", (e.stderr or "") + "\n[timeout expired]", elapsed
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def run_shell_subprocess(cmd: str, timeout: int = 20, cwd: Optional[str] = None) -> Tuple[int, str, str, float]:
    """Runs a bash command; feed the exact text from the model."""
    cmd = strip_code_fences(cmd)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,           # intentional: agent can run pipelines
            executable="/bin/bash"
        )
        elapsed = time.time() - start
        return proc.returncode, proc.stdout, proc.stderr, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        return 124, e.stdout or "", (e.stderr or "") + "\n[timeout expired]", elapsed

# ========= LLM Adapter (OpenAI example) =========
# Requires: pip install openai>=1.0.0
def make_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> LLMFn:
    """
    Returns a callable(messages) -> content string.
    Expects OPENAI_API_KEY to be set in environment.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install `openai` package (>=1.0.0).") from e

    client = OpenAI()

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content or ""
    return _call

# ========= Agent =========

class CoTActionAgent:
    def __init__(
        self,
        llm: LLMFn,
        config: AgentConfig = AgentConfig(),
        python_runner: Optional[RunnerFn] = None,
        shell_runner: Optional[RunnerFn] = None,
    ):
        self.llm = llm
        self.cfg = config
        self.python_runner = python_runner or (lambda code: run_python_subprocess(code, self.cfg.python_timeout_sec, self.cfg.working_dir))
        self.shell_runner = shell_runner or (lambda cmd: run_shell_subprocess(cmd, self.cfg.shell_timeout_sec, self.cfg.working_dir))

        if self.cfg.working_dir is None:
            self.cfg.working_dir = tempfile.mkdtemp(prefix="cot_agent_")

    def _parse_action(self, text: str) -> Tuple[str, str, str]:
        """
        Returns (thought, atype, body) where atype in {PYTHON,SHELL,RESPOND,STOP}
        If parsing fails, treat as RESPOND with the whole text as body.
        """
        m = _ACTION_RE.search(text.strip())
        if not m:
            return ("(parser: no Thought/Action matched)", "RESPOND", text.strip())
        thought = m.group("thought").strip()
        atype = m.group("atype").upper().strip()
        body = (m.group("body") or "").strip()
        return thought, atype, body

    def _observation_block(self, label: str, exit_code: int, out: str, err: str, elapsed: float) -> str:
        parts = [
            f"{label} Result:",
            f"exit_code: {exit_code}",
            f"elapsed_sec: {elapsed:.3f}",
            "--- STDOUT ---",
            truncate(out),
            "--- STDERR ---",
            truncate(err),
        ]
        return "\n".join(parts)

    def run(self, user_task: str) -> Dict:
        """
        Execute the loop until STOP. Returns a transcript with steps.
        """
        transcript: List[Dict] = []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_task},
        ]

        final_response: Optional[str] = None

        for step in range(1, self.cfg.max_steps + 1):

            # 1) Ask LLM for next Thought/Action
            reply = self.llm(messages).strip()
            print(f"==== {step} REPLY ===== ")
            print(reply)
            print(f"==== /{step} REPLY ===== ")
            print("")
            thought, atype, body = self._parse_action(reply)

            step_rec = {
                "step": step,
                "model_raw": reply,
                "thought": thought,
                "action_type": atype,
                "action_body": body,
                "observation": None,
            }

            # 2) Execute
            if atype == "PYTHON":
                code = body
                print('About to execute this code: ', code)
                res = input('Do you want to execute[y/n]?')
                if res.lower() == 'y':
                    exit_code, out, err, elapsed = self.python_runner(code)
                    obs = self._observation_block("PYTHON", exit_code, out, err, elapsed)
                    step_rec["observation"] = obs
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": f"Observation:\n{obs}"})
                else:
                    break

            elif atype == "SHELL":
                cmd = body
                exit_code, out, err, elapsed = self.shell_runner(cmd)
                obs = self._observation_block("SHELL", exit_code, out, err, elapsed)
                step_rec["observation"] = obs
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation:\n{obs}"})

            elif atype == "RESPOND":
                # Treat as a user-visible message; loop continues unless model chooses STOP later
                final_response = strip_code_fences(body)
                obs = f"RESPOND acknowledged. Message delivered to user.\n--- MESSAGE ---\n{truncate(final_response, 4000)}"
                step_rec["observation"] = obs
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation:\n{obs}"})
                break

            elif atype == "STOP":
                step_rec["observation"] = "Loop terminated by model."
                transcript.append(step_rec)
                break

            else:
                # Fallback: unknown action => treat as RESPOND
                final_response = body or reply
                obs = "Unknown action; treated as RESPOND."
                step_rec["observation"] = obs
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation:\n{obs}"})

            transcript.append(step_rec)
        return {
            "transcript": transcript,
            "final_response": final_response,
            "working_dir": self.cfg.working_dir,
        }

# ========= Example Usage =========
if __name__ == "__main__":
    # Choose your LLM adapter (OpenAI shown here)
    llm = make_openai_llm(model="gpt-4o-mini", temperature=0.2)

    agent = CoTActionAgent(
        llm=llm,
        config=AgentConfig(
            model_name="gpt-4o-mini",
            temperature=0.2,
            max_steps=12,
            python_timeout_sec=30,
            shell_timeout_sec=20,
            working_dir=None,            # temp dir
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        ),
        # Example of swapping Python runner with your Docker REPL:
        # python_runner=lambda code: your_docker_repl_run(code)
    )

    # Your task for the loop to solve:
    user_task = (
        'Download 100 pages from wikipedia in english. Compute the character frequencies.'
    )

    result = agent.run(user_task)
    print("\n=== FINAL RESPONSE (if any) ===\n", result["final_response"])
    print("\n=== TRANSCRIPT (compact) ===")
    for s in result["transcript"]:
        print(f"\nStep {s['step']}: {s['action_type']}")
        print("Thought:", truncate(s["thought"], 400))
        print("Action body:", truncate(s["action_body"], 400))
        print("Observation:", truncate(s["observation"], 400))

