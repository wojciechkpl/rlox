## Summary

Brief description of what this PR does and why.

## Changes

-

## Test plan

- [ ] All CI gates pass locally (`bash scripts/check-ci-local.sh`)
- [ ] New tests added for new functionality
- [ ] If you touched `crates/rlox-sandbox/`: the Linux suite passes
      (`WK=1 bash scripts/check-ci-local.sh`, or `bash scripts/wk-sync-test.sh
      'cargo test -p rlox-sandbox --no-fail-fast'`) — macOS cannot build or lint
      that crate, so a local green run there proves nothing about it

## Related issues

Closes #

## Checklist

- [ ] Code follows the project's style and conventions
- [ ] Self-reviewed the diff for correctness
- [ ] Added/updated documentation if needed
- [ ] No breaking API changes (or documented in summary)
