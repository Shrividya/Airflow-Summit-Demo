# INC-1765: Full Auth Service Outage from Expired Certificate

**Date:** 2025-08-06
**Severity:** SEV-1
**Duration:** 38 minutes
**Services affected:** auth-service, all downstream authenticated APIs

## Summary
The internal mTLS certificate used by auth-service to authenticate to the
credentials database expired at 03:14 UTC. Every login and token-refresh
request began failing immediately, effectively taking down all authenticated
traffic platform-wide until a manually rotated certificate was deployed.

## Root cause
The certificate had a 90-day validity period and auto-rotation was intended
to run 14 days before expiry via a scheduled job. That job depended on a
secrets-manager API whose credentials had themselves been rotated the prior
week; the rotation job started failing silently with 403 errors that were
logged but not alerted on, so the certificate was never renewed and expired
on schedule.

## Detection
Detected immediately by uptime monitoring and customer reports the moment
the certificate expired, since the failure mode was a hard outage rather
than a gradual one. Time to detect was under 2 minutes; time to root-cause
was the majority of the 38-minute duration.

## Contributing factors
- The certificate auto-rotation job's own failures were not monitored;
  logging existed but no alert was attached to it.
- There was no expiry-countdown alert on the certificate itself as a
  backstop independent of the rotation job succeeding.
- The rotation job's dependency on secrets-manager credentials created a
  single point of failure that wasn't covered by its own health check.

## Remediation
1. Added a certificate-expiry alert firing at 30, 14, and 3 days before
   expiry, independent of whether the rotation job reports success.
2. Added alerting on the rotation job's own failure/error rate, not just
   its completion status.
3. Moved to short-lived (7-day) certificates with more frequent, lower-risk
   rotation, reducing the blast radius of any single rotation failure.

## Lessons learned
Any automated remediation or rotation job needs its own independent
backstop alert on the condition it exists to prevent, not just monitoring
of the job's own execution status. A job that silently stops running is
functionally the same as never having built it.
