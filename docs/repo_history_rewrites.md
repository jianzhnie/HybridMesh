# Repository history rewrites

Operations that changed commit SHAs. Keep this list short: a rewrite invalidates
every SHA, so existing clones, links, review threads and CI caches that point at
the old history stop resolving, and GitHub's contribution aggregation has to
recompute the branch.

## 2026-09-27 -- commit dates (retiming, "redate")

* **What**: commit author/committer dates were redistributed across 2026-09-17..
  2026-09-27 so that every day of the window carried commits -- 2026-09-19 had
  none. It was done by a throwaway script (`redate_history.py`, added in
  `d0d9349`; the follow-up `3715dd0` refreshes the docs references). Both of
  those commits are currently dated 2026-09-27, but their own dates were part
  of what moved, so treat the ordering here as approximate.
* **Content-preserving**: messages and trees were kept; only the dates changed.
  The tree is *not* enough to identify a commit afterwards, though: two commits
  rewritten on different days can hold identical trees while having different
  SHAs, which is why the pre-rewrite refs below are the only reliable record.
* **Pre-rewrite refs**: `backup-before-0919-fill` and `backup-before-redate`
  (both `e7b9fe4`) and `backup-redated-history` (`d0d9349`) still exist in the
  local clone, holding 125 / 125 / 149 commits that are not reachable from
  `main`. They are local-only: the remote has `main` and nothing else (no tags,
  no backup branches).

## 2026-09-29 -- commit identity: gmail -> 126

* **What**: every commit authored or committed with `jianzhnie@gmail.com` now
  carries `jianzhnie@126.com` (author *and* committer).
* **How**: `git filter-repo --email-callback
  'return email.replace(b"jianzhnie@gmail.com", b"jianzhnie@126.com")' --force`
  over all refs. The `.mailmap`, which mapped commits made with `126` to display
  as `gmail`, was dropped in the same series -- it would have kept showing the
  old address.
* **Content-preserving, verified**: tree hashes match pairwise (`old main~N` ==
  `new main~(N+1)`), author/committer dates and messages are unchanged, and
  `git log --all --format=%ae | sort -u` is a single value.
* **Pre-rewrite history**: the bare mirror
  `/Users/jianzhengnie/work_dir/HybridMesh-backup-pre-email-rewrite.git` (kept
  outside this repository) holds every ref as it was, with `REWRITE-NOTES.md`
  carrying the replay and restore commands.
* **What to expect afterwards**: GitHub aggregates the profile timeline and the
  contribution graph asynchronously and keys them by SHA, so the days around a
  rewrite can lag behind -- they may briefly show fewer commits than the branch
  has, or attribute them a day late. The repository view is always complete
  (`/commits?author=<login>&since=...&until=...`), because it reads the branch
  directly.

## Effect on the contribution graph (measured 2026-09-30)

The graph is not a view of the branch: it is a credit ledger keyed by SHA, across
all repositories, counting commits + PRs + issues + reviews, bucketed by the
commit's **author date in the commit's own timezone**. After a rewrite the SHAs
that a force-push removed keep their credit until GitHub recomputes, so the
ledger adds the versions up instead of replacing them.

Measured on 2026-09-30 (`git log origin/main` vs the profile's own
`contributions` fragment vs every commit object in the local object database,
i.e. every version this clone ever created):

| day | `main` (branch) | profile graph | local commit objects (all versions) |
| --- | --- | --- | --- |
| 09-18 | 23 | 59 | 52 |
| 09-19 | 10 | 10 | 20 |
| 09-20 | 33 | 111 | 105 |
| 09-21 | 27 | 82 | 82 |
| 09-22 | 4 | 13 | 12 |
| 09-23 | 4 | 13 | 12 |
| 09-24 | 17 | 51 | 51 |
| 09-25 | 19 | 59 | 58 |
| 09-26 | 8 | 24 | 24 |
| 09-27 | 12 | 20 | 28 |
| 09-28 | 12 | 6 | 24 |
| 09-29 | 9 | 3 | 13 |
| 09-30 | 1 | 0 | 1 |

Three days match the "all versions" column exactly (09-21, 09-24, 09-26) and
three more miss it by single digits (09-18, 09-20, 09-25) -- the graph counted
the discarded versions. The newest days run the other way: the graph has *fewer*
than the branch, because it has not yet picked up the latest rewrite. Days that
match the branch exactly (09-19) are the ones already recomputed.

Two measurement traps, both hit while producing the table:

* the API's `since`/`until` are UTC and it returns UTC timestamps, so
  `?author=...&since=&until=` does **not** agree with the graph's per-day
  bucketing (2026-09-25 is 19 commits in +0800, 21 when bucketed in UTC). Use
  `git log --date=format:%Y-%m-%d --pretty=format:%ad` for the branch side, and
  the profile fragment for the graph side.
* the graph fragment ignores `from`/`to` for anonymous requests and always
  returns the whole year, so read the cell you need out of that.

## Guard rails (do these once)

1. **Server side, the only one that really binds**: repository Settings ->
   Rules/Rulesets (or Branches) -> for `main`, block force pushes and enable
   "require linear history". After that a force-push is rejected by GitHub, not
   by a client that can be bypassed with `--no-verify`.
2. **Local, a safety net**: `git config core.hooksPath .githooks` enables
   `.githooks/pre-push`, which refuses a non-fast-forward push to `main`.
   Escape hatch for the deliberate case: `ALLOW_HISTORY_REWRITE=1 git push ...`.
   Git ignores a hook that is not executable -- `chmod +x .githooks/pre-push`.
   Note what it does and does not add: plain `git push` already refuses a
   non-fast-forward update before any hook runs, so the guard exists for
   `--force` / `--force-with-lease`, the two spellings that do rewrite. Verified
   with `git push --dry-run --force origin probe:main` and
   `--force-with-lease` (both refused with the message above, remote unchanged)
   plus a fast-forward, a new branch, a non-`main` branch and the
   `ALLOW_HISTORY_REWRITE=1` escape hatch (all four allowed).
3. **The rule itself**: a rewrite is only free before the commit is pushed.
   Once it is on `main`, fix it in a follow-up commit instead of amending or
   rebasing, and never retime previously pushed commits.

## Cleaning up what is already there

The branch is the part we control, and it is already clean: `main` has no empty
commits, no duplicate messages, and no remote branch or tag holds the old
history. What is left is (a) this clone's residue and (b) GitHub's ledger.

* **Local residue** -- 626 commit objects against 179 reachable from `main`.
  Archive first, then drop the branches, then let gc reap the rest:

  ```bash
  git bundle create ~/TorchLLMTuner-history-archive.bundle --all   # keep a copy
  git branch -D backup-before-0919-fill backup-before-redate backup-redated-history
  git reflog expire --expire=now --all && git gc --prune=now
  ```

  The bundle is the only remaining record of the pre-rewrite dates; keep it
  outside the repository (the bare mirror from the 09-29 entry already is).
* **GitHub's ledger** -- nothing on our side changes it. The only targeted fix
  is asking GitHub Support to recompute the contribution graph for the account
  (there is no self-service button); deleting and recreating the repository
  would also clear it, at the cost of stars, issues and every existing link.
  Until then the graph is expected to differ from `/commits/main`, and the
  branch (or `git log`) is the authority.
