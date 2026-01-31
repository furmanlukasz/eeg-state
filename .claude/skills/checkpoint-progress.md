# Checkpoint Progress Skill

When significant progress is made, this skill creates a context bus artifact and commits changes to git.

## When to Use (Proactively)

Use this skill automatically when:
- A new feature is implemented and working
- A bug/issue is fixed
- A new analysis script is created
- Training configs are added or modified
- Paper text is significantly updated
- An ablation study is set up
- Test results show meaningful findings

## When to Ask First

Ask the user before checkpointing when:
- Uncertain if the change is significant enough
- Multiple unrelated changes are mixed together
- The work is experimental/exploratory and might be reverted

## Context Bus Configuration

- **Project slug:** `phd-eeg-mci-biomarkers`
- **Source:** `claude`

## Artifact Types to Use

| Type | When to Use |
|------|-------------|
| `note` | Feature implementations, bug fixes, script additions |
| `decision` | Architectural choices, methodology decisions |
| `plan` | Implementation roadmaps, experiment designs |
| `summary` | Session summaries, analysis results |

## Procedure

### 1. Create Context Bus Artifact

Use `mcp__context-bus__context_upsert_artifact` with:
- `project_slug`: `phd-eeg-mci-biomarkers`
- `type`: appropriate type from above
- `title`: concise description (e.g., "Implemented phase-only ablation config")
- `content_md`: detailed markdown description including:
  - What was changed
  - Why it matters
  - Key files modified
  - How to use/run it
- `tags`: 2-6 relevant tags (e.g., `["ablation", "training", "amplitude-control"]`)
- `source`: `claude`

### 2. Git Commit (No Push)

After creating the artifact, commit all changes:

```bash
git add -A && git commit -m "$(cat <<'EOF'
<commit message>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

**Important:** Do NOT push. Just commit locally.

## Example Usage

After implementing the phase-only ablation:

1. Create artifact:
```
mcp__context-bus__context_upsert_artifact(
  project_slug="phd-eeg-mci-biomarkers",
  type="note",
  title="Implemented phase-only ablation for amplitude control",
  content_md="""
## Summary
Added phase-only training config and analysis script to address the "speed = amplitude proxy" critique.

## Changes
- `configs/model/transformer_phase_only.yaml` - Model config without amplitude
- `configs/experiment/ablation_phase_only.yaml` - Full experiment config
- `scripts/local_analysis/amplitude_ablation_analysis.py` - Three-way ablation analysis

## Why It Matters
This is the most powerful control against amplitude-proxy critique. If expert/novice speed difference persists without amplitude features, the effect reflects phase coordination dynamics.

## How to Run
```bash
# On RunPod
python -m eeg_biomarkers.training.train --config-name=experiment/ablation_phase_only paths.data_dir=data/ds001787
```
  """,
  tags=["ablation", "amplitude-control", "phase-only", "training"],
  source="claude"
)
```

2. Commit changes:
```bash
git add -A && git commit -m "feat: add phase-only ablation for amplitude control

- Add transformer_phase_only.yaml model config
- Add ablation_phase_only.yaml experiment config
- Add amplitude_ablation_analysis.py script
- Update paper with amplitude control framing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

## Tag Vocabulary

Use consistent tags:
- `training`, `ablation`, `analysis`, `paper`, `config`
- `amplitude-control`, `phase-only`, `contrastive`
- `meditation`, `greek-resting`, `mci`
- `bug-fix`, `feature`, `refactor`, `docs`
- `intrinsic-metrics`, `projection-invariance`, `velocity`
