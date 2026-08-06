# INC-2210: Search Results Served from Stale Index for 11 Days

**Date:** 2026-04-22
**Severity:** SEV-2
**Duration:** 11 days (silent)
**Services affected:** product-search, catalog-indexer DAG

## Summary
The nightly catalog-indexer Airflow DAG had been silently writing to a
retired index alias for 11 days after a search-cluster migration. The DAG
runs completed successfully and reported green every night, but the alias it
wrote to was no longer the one product-search read from, so customers saw
product data that was 11 days stale, including sold-out items still shown as
available.

## Root cause
During a search-cluster migration, the read alias was cut over to the new
cluster, but the indexer DAG's `write_alias` variable was a hardcoded string
rather than one resolved from the same service-discovery source the
read-side used. The DAG's success criteria only checked that the bulk index
API returned HTTP 200, not that the document count in the target index grew
as expected or that the target index was the one actually serving traffic.

## Detection
Detected by a customer complaint about a discontinued product appearing
in search results, escalated to the catalog team, who traced it back to the
indexer writing to the orphaned alias. No internal monitoring caught it.

## Contributing factors
- The DAG's "success" was defined purely by task completion, not by any
  check on the freshness or destination of the output.
- Read and write paths for the search alias were configured independently,
  with no shared source of truth and no automated drift check between them.
- There was no row-count or staleness assertion on the search index despite
  this being a well-understood failure mode for indexing pipelines.

## Remediation
1. The indexer DAG now resolves the write alias from the same
   service-discovery config the read path uses, removing the possibility of
   divergence.
2. Added a post-index validation task that checks document count delta,
   sample-document freshness (comparing indexed `updated_at` to source
   `updated_at`), and confirms the target index matches the currently active
   read alias before marking the DAG successful.
3. Added a daily staleness dashboard comparing catalog source-of-truth row
   counts to what is queryable through product-search.

## Lessons learned
A green DAG only proves the tasks ran without raising an exception; it says
nothing about whether the data produced is correct, complete, or reaching
the place users actually read from. Any pipeline whose failure mode is
"quietly writes to the wrong place" needs an explicit destination and
freshness check, not just a status-code check.
