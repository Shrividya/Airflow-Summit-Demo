# INC-2350: Support Ticket Repro Steps Pasted Credentials into Postmortem

**Date:** 2026-02-19
**Severity:** SEV-3
**Duration:** N/A (documentation hygiene incident, not a service outage)
**Services affected:** none directly -- internal postmortem authoring process

## Summary
While writing up the repro steps for a support escalation, the on-call
engineer pasted the raw curl command they used to reproduce the customer's
issue directly into this postmortem, including a live bearer token and the
customer's contact email. A second engineer pasted a debug AWS access key
from a scratch script used to inspect the affected S3 bucket. Neither
credential should ever have been committed to a document that gets indexed
into the searchable incident knowledge base.

## Root cause
No guidance or tooling existed to stop engineers from pasting raw
credentials into postmortem documents, and the ingestion pipeline that
indexes postmortems into the RAG assistant had no scan step to catch it
before embedding.

## Repro steps (as originally pasted, before this line was flagged)
```
curl -H "Authorization: Bearer [redacted]" \
  https://internal-api.example.com/v1/customers/acct_5521/replay
```
Customer contact for follow-up: jane.doe@example-customer.com

Debug credentials used to inspect the S3 bucket (scratch script, since
deleted): access key.

## Detection
Caught by a documentation review before publishing, not by any automated
check -- which is exactly the gap this incident exists to close.

## Contributing factors
- No automated secret/PII scanning on postmortem ingestion.
- No cultural norm of using placeholder values (`<TOKEN>`, `<EMAIL>`) in
  repro steps shared outside a ticketing system with its own access
  controls.

## Remediation
1. Added an ingest-time scan that redacts emails, bearer tokens, and
   cloud provider access keys before any postmortem text is chunked or
   embedded, so raw secrets never reach the vector index.
2. Treat cloud provider access keys and private key material as a hard
   block on promotion to the production index, even after redaction --
   staging can still be inspected, but a human must review and re-ingest
   a cleaned copy before it reaches prod.

