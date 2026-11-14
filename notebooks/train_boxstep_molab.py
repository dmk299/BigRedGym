import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess

    REPO_URL = "https://github.com/dmk299/BigRedGym.git"
    BRANCH = "go2-box-step"
    REPO = "BigRedGym"

    def sh(cmd, cwd=None):
        """Run a shell command and print its output."""
        p = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(p.stdout)
        return p.returncode

    mo.md(
        """
        # Go2 box step: training on molab

        Turn on the GPU first (notebook specs button in the header).
        Then click each **Run** button from top to bottom.
        """
    )
    return BRANCH, REPO, REPO_URL, mo, os, sh


@app.cell
def _(mo):
    run_setup = mo.ui.run_button(label="1. Check GPU, clone repo, install")
    run_setup
    return (run_setup,)


@app.cell
def _(BRANCH, REPO, REPO_URL, mo, os, run_setup, sh):
    mo.stop(not run_setup.value)
    sh("nvidia-smi")
    if os.path.isdir(REPO):
        sh(f"git fetch && git switch {BRANCH} && git pull", cwd=REPO)
    else:
        sh(f"git clone -b {BRANCH} {REPO_URL} {REPO}")
    sh("pip install uv")
    sh("uv sync --frozen", cwd=REPO)
    sh(
        'uv run --frozen python -c "import torch; '
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
        "torch.cuda.get_arch_list())\"",
        cwd=REPO,
    )
    setup_done = True
    return (setup_done,)


@app.cell
def _(mo):
    run_smoke = mo.ui.run_button(label="2. Smoke test (2 iterations)")
    run_smoke
    return (run_smoke,)


@app.cell
def _(REPO, mo, run_smoke, sh):
    mo.stop(not run_smoke.value)
    sh(
        "uv run --frozen scripts/train.py --task=go2boxstep --device=cuda:0 "
        "--headless --max_iterations 2 --disable_wandb",
        cwd=REPO,
    )
    return


@app.cell
def _(mo):
    iterations = mo.ui.number(
        start=10, stop=20000, step=10, value=1500, label="Training iterations"
    )
    run_name = mo.ui.text(value="boxstep", label="Run name")
    use_wandb = mo.ui.checkbox(label="Log to Weights & Biases")
    wandb_key = mo.ui.text(kind="password", label="W&B API key (if logging)")
    run_train = mo.ui.run_button(label="3. Start full training (background)")
    mo.vstack([iterations, run_name, use_wandb, wandb_key, run_train])
    return iterations, run_name, run_train, use_wandb, wandb_key


@app.cell
def _(REPO, iterations, mo, run_name, run_train, sh, use_wandb, wandb_key):
    mo.stop(not run_train.value)
    extra = f" --max_iterations {int(iterations.value)}"
    if run_name.value.strip():
        extra += f" --run_name {run_name.value.strip()}"
    if use_wandb.value and wandb_key.value:
        sh(f"uv run --frozen wandb login {wandb_key.value}", cwd=REPO)
    else:
        extra += " --disable_wandb"
    sh(
        "nohup uv run --frozen scripts/train.py --task=go2boxstep "
        f"--device=cuda:0 --headless{extra} > train.log 2>&1 &",
        cwd=REPO,
    )
    print("Training started. Use the progress button below to check on it.")
    return


@app.cell
def _(mo):
    run_check = mo.ui.run_button(label="4. Check progress")
    run_check
    return (run_check,)


@app.cell
def _(REPO, mo, run_check, sh):
    mo.stop(not run_check.value)
    sh(
        "grep -E 'Iteration|foot_placement|heading|extra_steps|average episode' "
        "train.log | tail -15; echo; tail -3 train.log",
        cwd=REPO,
    )
    return


@app.cell
def _(mo):
    run_save = mo.ui.run_button(label="5. Zip runs for download")
    run_save
    return (run_save,)


@app.cell
def _(REPO, mo, run_save, sh):
    mo.stop(not run_save.value)
    sh("ls logs", cwd=REPO)
    sh("rm -f runs.zip && zip -qr runs.zip logs train.log", cwd=REPO)
    with open(f"{REPO}/runs.zip", "rb") as f:
        _data = f.read()
    mo.download(data=_data, filename="runs.zip", label="Download runs.zip")
    return


if __name__ == "__main__":
    app.run()
