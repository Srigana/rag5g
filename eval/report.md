# Retrieval evaluation

- Questions scored: **20** (skipped: 0)
- **Right chunk in top 5: 75%** (15/20)
- Right chunk ranked first: 60%
- MRR@5: 0.654

| Spec | n | hit@5 |
|---|---|---|
| TS 23.501 | 6 | 83% |
| TS 33.501 | 8 | 62% |
| TS 38.300 | 6 | 83% |

| # | Question | Gold rank | Top hit |
|---|---|---|---|
| q01 | What is the purpose of the N9 interface in a PDU Session? | miss (>5) | TS 23.501 §5.6.1, pp.107-108 |
| q02 | What is the purpose of the KDF input parameter when deriving CK' and IK'? | miss (>5) | TS 33.501 §A.3, pp.237-238 |
| q03 | What is the purpose of the HANDOVER REQUEST ACKNOWLEDGE message in the context of handover preparation? | miss (>5) | TS 38.300 §9.2.3.4.2, pp.98-99 |
| q04 | What is the purpose of the Binding Indication in the subscription request? | 2 | TS 23.501 §6.3.1.0, pp.360-361 |
| q05 | What happens to the AS integrity protection algorithm if the AMF selects NIA0 as the NAS integrity protection algorithm? | 1 | TS 33.501 §6.7.3.6, p.85 |
| q06 | What type of signalling is used to provide multicast MRB configuration(s) to a UE in RRC_CONNECTED state? | 1 | TS 38.300 §16.10.1, p.179 |
| q07 | What happens to the UDR flag in the HPLMN when the subscribed S-NSSAIs change? | 1 | TS 23.501 §5.15.4.2, p.204 |
| q08 | What happens to the AMF's locally kept NCC value after a successful N2 Resume Request message is received from the ng-eNB? | 1 | TS 33.501 §6.16.2.4, pp.129-130 |
| q09 | What happens to the PDCP SN and HFN in the target gNB during a Full Configuration handover for RLC-AM bearers? | 1 | TS 38.300 §9.2.3.2.2, p.94 |
| q10 | What happens to the AMF's stored NH and NCC value when a new 5G NAS security context is activated? | miss (>5) | TS 33.501 §8.2, p.152 |
| q11 | What is the role of the IAB-donor-CU in the establishment and release of redundant routes for IAB-nodes operating in SA-mode? | 1 | TS 38.300 §4.7.4.3, p.36 |
| q12 | What happens to the UE's S-NSSAIs when it moves from EPS to 5GS? | 4 | TS 23.501 §5.17.2.4, pp.239-240 |
| q13 | Where is NF selection performed in the scenario where NF Service Consumer and NF Service Producer are in different PLMNs? | miss (>5) | TS 23.501 §6.3.17, p.386 |
| q14 | What type of notification may be sent by the SMF to the AF about a UP path change for a PDU Session? | 3 | TS 23.501 §5.6.7.1, pp.122-123 |
| q15 | What happens when the MN cannot allocate an unused radio bearer identity for a new radio bearer in the SN due to radio bearer identity space exhaustion? | 1 | TS 33.501 §6.10.2.1, pp.106-107 |
| q16 | What happens to the UE when no slice-based cell reselection information is provided for any NSAG that was determined to be considered during cell reselection? | 1 | TS 38.300 §16.3.3a, pp.157-158 |
| q17 | What is the purpose of verifying the signature of the access token using NRF's public key? | 1 | TS 33.501 §13.4.1.1.2, p.203 |
| q18 | What is the definition of A2X services? | 1 | TS 38.300 §3.1, p.21 |
| q19 | What is the purpose of setting up two redundant PDU Sessions over the 5G network? | 1 | TS 23.501 §5.33.2.1, pp.329-330 |
| q20 | What attributes are confidentiality protected when sent over the N32-f interface, irrespective of the Data-type Encryption Policy? | 1 | TS 33.501 §5.9.3.3, pp.45-46 |
