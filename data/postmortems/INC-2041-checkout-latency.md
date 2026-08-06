# INC-2041: Checkout Latency Spike

**Date:** 2026-02-11
**Severity:** SEV-2
**Duration:** 47 minutes
**Services affected:** checkout-api, payment-gateway-proxy

## Summary
Between 14:02 and 14:49 UTC, p99 latency on the checkout API rose from ~220ms
to over 6s. Roughly 9% of checkout attempts timed out and were retried by the
client, and 1.3% of sessions abandoned the cart during the incident window.

## Root cause
A connection pool exhaustion in `payment-gateway-proxy`. A deploy earlier that
day reduced the proxy's outbound connection pool from 200 to 40 as part of an
unrelated config cleanup. Under normal traffic this was sufficient, but a
scheduled marketing push at 14:00 UTC roughly tripled checkout volume, and the
proxy began queuing requests behind the exhausted pool.

## Detection
Detected by a customer support ticket spike, not by automated alerting. The
existing latency alert threshold (2s p99) was correctly configured but the
on-call engineer had it muted due to an unrelated noisy alert the previous
week.

## Contributing factors
- The connection pool size change was not covered by a load test.
- The marketing push calendar is not integrated with the on-call/capacity
  planning process, so traffic spikes are not flagged to the infra team.
- Alert muting has no expiry; the latency alert had been muted for 9 days.

## Remediation
1. Restored the outbound connection pool to 200 and added a floor value with
   a code review requirement for any change below it.
2. Added an automatic 72-hour expiry to all muted alerts.
3. Marketing is now required to file a traffic-impact ticket 5 business days
   before any push expected to exceed 2x baseline checkout volume.
4. Added a synthetic checkout probe that runs every 60 seconds and pages
   on-call directly, independent of the support-ticket path.

## Lessons learned
Config changes to connection pools and resource limits need the same load
testing rigor as code changes. Alert muting without an expiry is a recurring
root cause across several past incidents and should be disabled platform-wide.
