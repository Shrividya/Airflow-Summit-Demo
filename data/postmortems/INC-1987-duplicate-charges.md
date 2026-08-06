# INC-1987: Duplicate Charges from Payment Webhook Replay

**Date:** 2025-11-03
**Severity:** SEV-1
**Duration:** 2 hours 15 minutes
**Services affected:** payment-webhook-consumer, billing-ledger

## Summary
A backlog-drain event in the message queue caused 3 days of already-processed
payment webhooks to be redelivered. Because the webhook consumer's
idempotency check relied on an in-memory cache with a 15-minute TTL rather
than a persisted idempotency key, roughly 4,800 payments were charged a
second time before the issue was caught.

## Root cause
The webhook consumer used an in-memory LRU cache keyed on webhook event ID to
prevent double-processing. This was sufficient for normal traffic but was
never designed to survive a queue replay of historical messages, since events
older than the 15-minute TTL were treated as new. A separate incident
(unrelated queue rebalance) triggered the broker to redeliver 3 days of
retained messages starting at 09:10 UTC.

## Detection
Caught by the finance reconciliation job that runs nightly, which flagged an
anomalous spike in refund requests the following morning. There was no
real-time detection during the incident itself.

## Contributing factors
- Idempotency was implemented in application memory instead of a persisted
  store (e.g., a dedicated idempotency-key table), so it did not survive
  either a redeploy or a replay.
- The message broker's retention window (3 days) was longer than anyone on
  the payments team realized; the replay scenario had never been tested.
- No real-time anomaly detection existed on "charges per minute," only on
  end-of-day reconciliation.

## Remediation
1. Replaced the in-memory idempotency cache with a persisted idempotency-key
   table, keyed on webhook event ID, checked before any charge is applied,
   with no TTL.
2. Added a real-time anomaly detector on charges-per-minute that pages
   on-call if the rate exceeds 3x the trailing 1-hour average.
3. Ran a game-day exercise simulating a full queue replay against staging to
   validate the fix.
4. All 4,800 affected customers were automatically refunded within 24 hours
   and notified.

## Lessons learned
Idempotency guarantees must be evaluated against the full space of delivery
semantics the upstream system can produce, including replay and redelivery,
not just normal steady-state traffic. In-memory idempotency caches are not a
substitute for a persisted idempotency key when money is involved.
