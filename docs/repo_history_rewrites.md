# Repository history rewrites

Operations that changed commit SHAs. Keep this list short: a rewrite invalidates
every SHA, so existing clones, links, review threads and CI caches that point at
the old history stop resolving, and GitHub's contribution aggregation has to
recompute the branch.

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
