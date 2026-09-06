# Cully Hermes upstream mirror

**Repository:** `cullyk/hermes-agent`  
**Upstream:** `NousResearch/hermes-agent`

This repository tracks the upstream Hermes Agent source. The root README and
translated documentation are upstream product documentation. They do not
represent the deployment, model policy, Azure admission, credentials, hosts or
production-acceptance state of Cully Systems.

## Repository roles

| Repository | Responsibility |
|---|---|
| `cullyk/hermes-agent` | Upstream source mirror and upstream update provenance |
| `cullyk/hermes-runtime` | Cully Azure inference broker, admission, Codex supervisor and production-acceptance integration |
| `cullyk/hermes-extensions` | Private credential-free extensions, policies and remote-node bootstrap packages |

Do not implement Cully-specific Azure, credential, host, customer or production
logic directly on the upstream mirror branch. Use the owning runtime or
extensions repository so upstream synchronisation remains reviewable and
rollback boundaries stay clear.

## Upstream synchronisation

Automated pull-bot branches and pull requests are upstream provenance only.
Before accepting an upstream sync:

1. read the exact upstream and local commits;
2. inspect changes affecting providers, model discovery, updates, migrations,
   terminal backends, tools, skills, gateways, credentials and state;
3. verify whether `hermes-runtime` or `hermes-extensions` carries a dependent
   patch or contract;
4. run the upstream repository tests selected by the change;
5. run the Cully runtime and extension compatibility suites separately;
6. keep source merge, installed update and production rollout as distinct
   decisions.

A closed pull-bot PR does not prove the installed Cully runtime was updated.
An upstream test result does not prove the Cully Azure or production boundary.

## Local change policy

Local-only mirror documentation should be small and isolated. Avoid changing the
upstream README merely to describe Cully deployment state because the next
upstream sync would repeatedly conflict with it.

When a local patch to upstream code is unavoidable:

- create a focused issue and branch;
- identify the missing upstream contract;
- include deterministic tests;
- record whether the patch is intended for upstream submission, runtime-only
  carry or retirement;
- keep credentials, private endpoints, host identifiers and customer data out
  of the repository;
- require human merge.

## Current non-claims

This repository does not claim:

- that `hermes-runtime` integration PR #4 is merged or production accepted;
- that the installed coordinator matches this mirror head;
- that an upstream update has passed Cully-specific migration or rollback;
- that Azure quota, cost, certificates, live canaries or cutover are accepted;
- that private extensions are installed or enabled.

Use the current PRs, runbooks and acceptance ledgers in the owning repositories
for those facts.
