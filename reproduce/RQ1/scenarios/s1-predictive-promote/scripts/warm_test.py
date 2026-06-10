"""Verify max_new_tokens=0 (prefill-only) warms an evicted prefix into HBM."""
import requests, time, statistics, sys, json as J
B="http://127.0.0.1:30000"; V=129000; VICT=50000; FLOOD_N=6; FLOOD_LEN=50000; N=3
def seq(s,n): return [(s+i)%V for i in range(n)]
def gen(ids,pid,mx=4):
    requests.post(B+"/generate",json={"input_ids":ids,"sampling_params":{"temperature":0,"max_new_tokens":mx,"ignore_eos":True},"program_id":pid},timeout=300).raise_for_status()
def warm(ids,pid):  # the CLEAN path: prefill-only, no decode
    r=requests.post(B+"/generate",json={"input_ids":ids,"sampling_params":{"temperature":0,"max_new_tokens":0},"program_id":pid},timeout=300)
    return r.status_code, r.text[:80]
def ttft(ids,pid):
    body={"input_ids":ids,"sampling_params":{"temperature":0,"max_new_tokens":6,"ignore_eos":True},"program_id":pid,"stream":True}
    t0=time.perf_counter(); tt=None; c=0
    with requests.post(B+"/generate",json=body,timeout=300,stream=True) as r:
        for line in r.iter_lines():
            if not line: continue
            if tt is None: tt=(time.perf_counter()-t0)*1000
            s=line.decode() if isinstance(line,bytes) else line
            if s.startswith("data:"): s=s[5:].strip()
            if s and s!="[DONE]":
                try: c=int((J.loads(s).get("meta_info") or {}).get("cached_tokens") or c)
                except: pass
            if tt and c: break
    return tt,c
def flood(tag):
    for k in range(FLOOD_N): gen(seq((90000+k*4001+tag*131)%V,FLOOD_LEN),f"FL{tag}_{k}",mx=2)
b_t,o_t=[],[]
for t in range(N):
    vid=f"W{t}"; vids=seq((t*7001+1)%V,VICT)
    gen(vids,vid,mx=4); time.sleep(0.5)
    flood(2*t)
    tb,cb=ttft(vids,vid)                          # B: resume directly
    flood(2*t+1)
    sc,txt=warm(vids,vid)                          # CLEAN WARM: max_new=0 prefill-only
    to,co=ttft(vids,vid)                           # ours: resume after warm
    b_t.append(tb); o_t.append(to)
    print(f"trial {t}: warm_status={sc}  B={tb:.0f}ms(cached={cb})  ours={to:.0f}ms(cached={co})  saved={tb-to:.0f}ms")
print(f"\nB mean={statistics.mean(b_t):.0f}ms  ours(warm max_new=0) mean={statistics.mean(o_t):.0f}ms")
sv=statistics.mean(b_t)-statistics.mean(o_t)
print(f"=> clean warm saves {sv:.0f}ms ({100*sv/statistics.mean(b_t):.0f}%) — prefill-only stages prefix to HBM" if sv>500 else f"=> warm did NOT help (saved {sv:.0f}ms) — investigate")
