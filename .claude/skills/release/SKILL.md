---
name: release
description: Cuts a versioned release of gridfinity-shadowbox (jarvick257/gridfinity-shadowbox on GitHub). Finds the latest v* tag, reviews what changed on main since then, proposes the next semver tag, and after the user agrees creates and pushes the tag (which builds and publishes the Docker image) and a GitHub release via gh. Use when the user wants to cut a release, tag a version, or publish a new image.
---

# Release

`main` is the release branch. Pushing to `main` builds nothing: the Docker image
is built **only** when a `v*` tag is pushed (`.github/workflows/docker.yml`). That
run smoke-tests an amd64 build, then builds `linux/amd64` + `linux/arm64` and pushes
`ghcr.io/jarvick257/gridfinity-shadowbox` tagged from the git tag:

| git tag | image tags |
|---|---|
| `v1.2.3` | `1.2.3`, `1.2`, `latest`, `sha-<short>` |
| `v1.2.3-rc.1` | `1.2.3-rc.1`, `sha-<short>` (no `latest`) |

So a tag is the release: it is what users pull. Never push a tag the user has not
agreed to, and never move or delete a pushed tag without asking (see *Mistakes*).

## Workflow

1. **Preconditions.** Stop and tell the user if any of these fail:
   ```bash
   git fetch origin --tags --prune-tags
   git status --short                    # must be clean
   git rev-parse --abbrev-ref HEAD       # must be main
   git rev-list --left-right --count origin/main...HEAD   # must be "0 0"
   gh auth status
   ```
   Local commits not on `origin/main` are not part of a release. Ask the user to
   push (or push for them if they say so) before continuing. Releases are cut
   from `origin/main`, never from a working tree.

2. **Find the latest tag.**
   ```bash
   git -c versionsort.suffix=- tag --list 'v*' --sort=-v:refname | head -1
   ```
   `versionsort.suffix=-` sorts `v1.2.0-rc.1` below `v1.2.0` and `rc.10` above
   `rc.2`. If the newest tag is a release candidate, review against the newest
   *final* tag (`grep -v -- '-rc\.'`), and offer both the next candidate
   (`-rc.N+1`) and the final version of that candidate. With no tag at all,
   this is the first release: propose the version in `pyproject.toml`
   (`project.version`) as `v<that>` and review the whole history in step 3.

3. **Review what changed** since that tag (`<prev>`):
   ```bash
   git log --oneline --no-merges <prev>..origin/main
   git diff --stat <prev>..origin/main
   ```
   If there are no commits, stop: there is nothing to release. Don't classify by
   commit subjects alone. Read the diff of the files that define the public
   surface (below) to spot breaking changes that aren't named in the subjects.

4. **Propose the next version.** Format is `v<major>.<minor>.<patch>`, per
   [semver](https://semver.org/). What counts as this project's public interface:

   - **CLI:** subcommands, flags, defaults (`gridfinity_shadowbox/cli.py`, the
     generated `--bin-*` flags from `BinParams`)
   - **params.json:** schema and meaning (`gridfinity_shadowbox/params.py`). An
     old params.json must still load *and* replay to the same STL.
   - **SVG contract** between `outline` and `extrude` (mm units, y flip, accepted
     SVG features)
   - **STL output:** bin coordinates, pocket placement, and bin geometry for the
     same inputs (`extrude.placement_matrix`, `bins/`)
   - **Reference sheet:** marker IDs and geometry (`sheet.py`). A change means
     old printed sheets stop working.
   - **Docker image:** entrypoint, default command, `/data` workdir, uid 1000,
     `SHADOWBOX_UI_HOST`, port 7860

   | Change | Bump (≥ 1.0) | Bump (0.x) |
   |---|---|---|
   | Removed/renamed flag or params.json key, old params.json rejected or replays differently, old printed sheets unusable, image interface changed | major | minor |
   | New knob/feature, UI improvement, new option with a default that keeps old output | minor | minor |
   | Bug fix, docs, tests, CI, dependency bumps without behaviour change | patch | patch |

   A geometry *fix* that changes the STL for an unchanged params.json is still a
   reproducibility change. Bump at least minor and call it out in the notes.
   Offer `-rc.N` (dotted, `v1.3.0-rc.1`; the dot keeps `rc.10` sorting above
   `rc.2`) when the user wants to try the image before `latest` moves.

5. **Draft release notes** from the commits, grouped for users, not as a commit
   list: *Breaking changes* (with what to do), *New*, *Fixes*, *Other*. Leave out
   empty groups. End with how to get it:
   ```
   docker pull ghcr.io/jarvick257/gridfinity-shadowbox:X.Y.Z
   ```

6. **Confirm with the user.** Use AskUserQuestion to present the proposed tag (with
   a one-line reason), the alternatives worth considering (e.g. patch vs minor),
   and the draft notes. Continue only on an explicit yes. Apply any edits they
   make to the version or notes.

7. **Sync the package version.** If `pyproject.toml` `project.version` isn't already
   the new version without the `v` (e.g. `1.3.0`, `1.3.0rc1` for `-rc.1` per
   PEP 440), update it and the lockfile, then commit and push to `main`:
   ```bash
   # edit pyproject.toml: version = "X.Y.Z"
   uv lock
   git add pyproject.toml uv.lock
   git commit -m "Release vX.Y.Z"
   git push origin main
   ```
   Skip this step when it already matches (e.g. the first release `v0.1.0`).

8. **Tag and push.** An annotated tag on the commit now at `origin/main`:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z" origin/main
   git push origin vX.Y.Z
   ```

9. **Create the GitHub release:**
   ```bash
   gh release create vX.Y.Z --repo jarvick257/gridfinity-shadowbox \
     --verify-tag --title vX.Y.Z --notes-file - <<'EOF'
   <notes from step 6>
   EOF
   ```
   Add `--prerelease` for `-rc.N` tags (and `--latest=false`).

10. **Watch the image build.** The tag push started it:
    ```bash
    gh run list --repo jarvick257/gridfinity-shadowbox --workflow docker.yml --limit 3
    gh run watch <run-id> --repo jarvick257/gridfinity-shadowbox --exit-status
    ```
    Run the watch in the background. It takes several minutes, and arm64 runs
    under QEMU. If it fails, show the failing step
    (`gh run view <run-id> --log-failed`). The release exists but has no image
    yet, so fix forward with a patch release. Don't re-point the tag.

11. **Report** the tag, the release URL (`gh release view vX.Y.Z --json url -q .url`),
    the workflow result, and the image tags that were published.

## Mistakes

- **Wrong version or notes, image not pulled by anyone yet:** fix the notes with
  `gh release edit`. For a wrong tag, ask the user before
  `gh release delete vX.Y.Z --cleanup-tag`, and also delete the pushed image
  versions in the GHCR package settings.
- **Broken release that is already out:** release a patch with the fix. Don't
  reuse a version number: people and caches may already have that image.

## First-time setup notes

- A new GHCR package is **private**. After the first successful run, the owner
  makes it public under *GitHub → Packages → gridfinity-shadowbox → Package
  settings*, or `docker pull` needs a login.
- GitHub Actions caches are scoped per ref, so each tag build starts cold. That's
  expected.
