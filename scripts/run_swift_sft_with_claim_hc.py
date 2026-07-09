from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claim_hc.launcher import maybe_relaunch_distributed, rewrite_sys_argv


def main() -> None:
    rewrite_sys_argv(sys.argv[1:])
    maybe_relaunch_distributed()

    from swift.pipelines import sft_main
    from claim_hc.local_template import register_local_internvl_template
    from claim_hc.runtime_patch import install_claim_hc_runtime_patch

    register_local_internvl_template(REPO_ROOT / "forks" / "claim_hc" / "internvl.py")
    install_claim_hc_runtime_patch()
    sft_main()


if __name__ == "__main__":
    main()
